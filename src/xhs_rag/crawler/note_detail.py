"""笔记详情采集 —— 解析 SSR 的 __INITIAL_STATE__ + 嗅探视频真实 URL。

★ 实测结论（2026-08-31，M2 实勘）：
  1. 详情页 HTML 里有 window.__INITIAL_STATE__={...}，但它是 JS 字面量
     而非合法 JSON —— 内含 undefined（如 "addDesktopPrompt":undefined）。
     json.loads 直接崩。解法：把原始字符串交给页面上下文 eval（JS 语义
     天然认 undefined），再 JSON.stringify 转成合法 JSON 交给 Python。
  2. 页面渲染完成后 __INITIAL_STATE__ 会被前端清空，page.evaluate 读不到
     window 对象 —— 必须从 page.content() 的 HTML 字符串里正则提取。
  3. 视频笔记的 video.media.stream.EF*（各清晰度档位）在 SSR 注入时是
     空对象，拿不到真实 mp4 URL。但详情页打开后视频会自动播放，播放器
     会请求 sns-video-v3.xhscdn.com/stream/..._261.mp4?sign=...&t=...
     —— 用 response 嗅探抓这个 URL（带 sign 时效签名，需尽快下载）。
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from playwright.sync_api import BrowserContext, Page, Response

from ..auth.qrcode_login import CAPTCHA_MARK
from ..auth.session import CaptchaRequired

# 视频流域名（播放器嗅探用）
# ★ 2026-08-31 实测发现两种路径：/stream/（多数）和 /pre_post/（部分笔记走 MSE 分片）
#   只匹配 stream 会漏掉 pre_post 类视频 → 改用域名前缀 sns-video + .mp4 校验
_VIDEO_URL_MARK = "sns-video"
_VIDEO_URL_EXT = ".mp4"

# __INITIAL_STATE__ 的 JS 字面量起点标记
_STATE_MARK = "window.__INITIAL_STATE__="


# ──────────────────────────────────────────────────────────
# 解析工具
# ──────────────────────────────────────────────────────────
def extract_initial_state_raw(html: str) -> str | None:
    """从 HTML 里括号配平提取 __INITIAL_STATE__ 的原始 JS 字面量。

    为什么不用正则 .*?：JSON 嵌套大括号，非贪婪会提前截断。
    这里做括号配平扫描，跳过字符串内的括号和转义。
    """
    i = html.find(_STATE_MARK)
    if i < 0:
        return None
    j = html.find("{", i)
    if j < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    limit = min(len(html), j + 3_000_000)
    for k in range(j, limit):
        c = html[k]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return html[j:k + 1]
    return None


def parse_initial_state(page: Page, raw: str) -> dict:
    """把 JS 字面量（含 undefined）交给页面 eval，转成合法 JSON 再解析。"""
    result = page.evaluate(
        """(raw) => {
            const obj = eval('(' + raw + ')');
            return JSON.stringify(obj);
        }""",
        raw,
    )
    return json.loads(result)


def _pick_img_url(im: Any) -> str:
    """从详情页 imageList 单张图里挑可下载 URL。

    实测字段：url / urlDefault / urlPre / infoList[].url，按优先级取。
    """
    if isinstance(im, str):
        return im
    if not isinstance(im, dict):
        return ""
    for key in ("url", "urlDefault", "urlPre"):
        v = im.get(key)
        if isinstance(v, str) and v:
            return v
    info = im.get("infoList") or im.get("info_list") or []
    if isinstance(info, list):
        for it in info:
            if isinstance(it, dict) and it.get("url"):
                return str(it["url"])
    return ""


def _pick_video_meta(n: dict) -> dict:
    """从 note 对象提取视频元数据（时长/宽高/video_id/md5/stream_types）。"""
    video = n.get("video") or {}
    meta: dict[str, Any] = {}

    capa = video.get("capa") or {}
    meta["duration_sec"] = capa.get("duration")

    v2_raw = video.get("mediaV2") or ""
    if isinstance(v2_raw, str) and v2_raw:
        try:
            v2 = json.loads(v2_raw)
        except Exception:
            v2 = {}
    else:
        v2 = v2_raw if isinstance(v2_raw, dict) else {}
    vv = v2.get("video") or {}
    if not meta.get("duration_sec"):
        meta["duration_sec"] = vv.get("duration")
    meta["width"] = vv.get("width")
    meta["height"] = vv.get("height")
    meta["video_id"] = v2.get("video_id") or vv.get("video_id")
    meta["md5"] = vv.get("md5")
    st = vv.get("stream_types")
    if st:
        meta["stream_types"] = json.dumps(st, ensure_ascii=False)

    # media 里的兜底宽高
    if not meta.get("width"):
        meta["width"] = (video.get("media") or {}).get("width")
    if not meta.get("height"):
        meta["height"] = (video.get("media") or {}).get("height")
    return meta


@dataclass
class DetailResult:
    note: dict | None = None
    images: list[str] = field(default_factory=list)   # 图片 URL 列表
    video: dict = field(default_factory=dict)          # 视频元数据
    video_url: str = ""                                 # 播放器嗅探到的真实 URL
    raw_state: dict = field(default_factory=dict)
    stopped_reason: str = ""


class DetailCrawler:
    """逐条抓详情：goto 详情页 → 提取 __INITIAL_STATE__ → 嗅探视频 URL。"""

    def __init__(self, ctx: BrowserContext, cfg: Any):
        self.ctx = ctx
        self.cfg = cfg

    def fetch(self, note_id: str, xsec_token: str,
              wait_video: int = 12) -> DetailResult:
        url = f"https://www.xiaohongshu.com/explore/{note_id}"
        if xsec_token:
            url += f"?xsec_token={xsec_token}&xsec_source=pc_feed"

        res = DetailResult()
        video_urls: list[str] = []

        def on_response(resp: Response):
            u = resp.url or ""
            if _VIDEO_URL_MARK in u and _VIDEO_URL_EXT in u and u not in video_urls:
                video_urls.append(u)

        self.ctx.on("response", on_response)

        page = self.ctx.pages[0] if self.ctx.pages else self.ctx.new_page()
        try:
            page.goto(url, wait_until="load", timeout=60_000)
            page.wait_for_timeout(4000)

            if CAPTCHA_MARK in (page.url or ""):
                res.stopped_reason = "触发验证码风控"
                return res

            html = page.content()
            raw = extract_initial_state_raw(html)
            if not raw:
                res.stopped_reason = "HTML 里没有 __INITIAL_STATE__（可能被跳转/风控）"
                return res

            try:
                state = parse_initial_state(page, raw)
            except Exception as e:
                res.stopped_reason = f"__INITIAL_STATE__ 解析失败: {e}"
                return res

            res.raw_state = state
            note_map = (state.get("note") or {}).get("noteDetailMap") or {}
            nd = note_map.get(note_id)
            if not nd:
                # 拿不到当前 note_id 时尝试第一条（通常是它）
                nd = next(iter(note_map.values()), None)
            if not nd:
                res.stopped_reason = "noteDetailMap 里没有该笔记"
                return res

            n = nd.get("note") or {}
            res.note = self._normalize(n, note_id)

            # 图片 URL 列表
            for im in n.get("imageList") or []:
                u = _pick_img_url(im)
                if u:
                    res.images.append(u)

            # 视频元数据
            if res.note.get("note_type") == "video":
                res.video = _pick_video_meta(n)

                # 等播放器自动播放，嗅探真实 mp4 URL
                # 首帧就绪通常 <5s，最多等 wait_video 秒
                deadline = time.time() + wait_video
                while not video_urls and time.time() < deadline:
                    page.wait_for_timeout(1000)
                if video_urls:
                    res.video_url = video_urls[-1]
                else:
                    res.stopped_reason = "视频 URL 未嗅探到（播放器未加载？）"
            else:
                res.stopped_reason = ""

        except CaptchaRequired as e:
            res.stopped_reason = f"验证码风控: {e}"
        except Exception as e:
            res.stopped_reason = f"异常: {type(e).__name__}: {e}"
            logger.exception("详情抓取异常")

        self.ctx.remove_listener("response", on_response)
        return res

    def _normalize(self, n: dict, note_id: str) -> dict:
        """标准化为 notes 表字段（与 collect._extract_note 输出对齐）。"""
        user = n.get("user") or {}
        inter = n.get("interactInfo") or n.get("interact_info") or {}

        # desc 清洗：小红书正文带 [话题] 标签，保留原文即可
        desc = n.get("desc") or ""

        published = n.get("time") or n.get("lastUpdateTime") or 0
        try:
            published = int(published)
            if 0 < published < 10**11:
                published *= 1000
        except (TypeError, ValueError):
            published = 0

        def cnt(v):
            return _parse_count(v)

        note_type = n.get("type") or ("video" if n.get("video") else "normal")

        tags = []
        for t in n.get("tagList") or []:
            if isinstance(t, dict):
                name = t.get("name") or t.get("tag_name") or ""
                if name:
                    tags.append(name)
            elif isinstance(t, str) and t:
                tags.append(t)

        return {
            "note_id": note_id,
            "title": str(n.get("title") or ""),
            "author_id": str(user.get("userId") or "") or None,
            "author_name": str(user.get("nickname") or "") or None,
            "note_type": str(note_type),
            "desc": str(desc),
            "published_at": published,
            "liked_count": cnt(inter.get("likedCount")),
            "collect_count": cnt(inter.get("collectedCount")),
            "comment_count": cnt(inter.get("commentCount")),
            "tags": tags,
        }


def _parse_count(v: Any) -> int:
    """详情页互动数也是数字或 '10万+' 字符串，同 M1 的解析逻辑。"""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return int(v)
    s = str(v or 0).strip()
    if not s:
        return 0
    try:
        return int(s)
    except ValueError:
        pass
    mult = 1
    if s.endswith("万+"):
        s, mult = s[:-2], 10000
    elif s.endswith("万"):
        s, mult = s[:-1], 10000
    elif s.endswith("千+"):
        s, mult = s[:-2], 1000
    elif s.endswith("千"):
        s, mult = s[:-1], 1000
    elif s.endswith("+"):
        s = s[:-1]
    try:
        return int(float(s) * mult)
    except ValueError:
        return 0


def _jitter() -> int:
    """详情页间隔高斯抖动：1.5~3 秒（config detail_interval 默认）。"""
    return int(random.gauss(2.2, 0.5) * 1000 + 300)
