"""收藏列表采集 —— 浏览器代发请求，我们只监听响应。

★ 为什么必须这样：小红书前端用 axios 拦截器注入 x-s / x-s-common 签名，
  外部用 requests/curl 重放同样的 URL 只会拿到 {"code": -1}。
  正确做法是让页面自己发请求（前端 JS 自己算签名自己带），
  我们用 page.on("response") 抓响应体。这条路被 MediaCrawler 工程化验证过。

★ 风控纪律（来自 M0 踩坑记录第 5 条）：
  - 同一时间只有一个浏览器上下文在跑，绝不并发多个采集进程
  - 翻页用真实滚动触发（前端检测到滚动才发下一页请求），不用循环请求
  - 每页之间随机延时（高斯抖动），像人浏览而不是脚本
  - 一旦 looks_like_captcha 命中，立即停下并抛出 CaptchaRequired，
    不要重试、不要重新扫码 —— 越扫风控越严
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from loguru import logger
from playwright.sync_api import BrowserContext, Page, Response

from ..auth.qrcode_login import CAPTCHA_MARK, SESSION_COOKIE, get_login_user_id
from ..auth.session import CaptchaRequired

# 收藏列表接口（★ 实测路径，网上流传的 note_collect_page 是 404）
COLLECT_API_MARK = "/api/sns/web/v2/note/collect/page"

# 收藏 tab 的 profile URL —— 打开后页面会自动请求上面的接口
COLLECT_TAB_URL = "https://www.xiaohongshu.com/user/profile/{uid}?tab=collect"

# 滚动到底触发前端加载下一页（小红书是滚动懒加载）
# ★ 2026-08-31 实测：收藏列表在内部滚动容器里，只滚 document.body 不触发加载。
#   找出所有可滚动容器一起滚到底。
_SCROLL_DOWN = """
() => {
    const scrollables = [...document.querySelectorAll('*')].filter(el => {
        return el.scrollHeight > el.clientHeight + 100 && el.clientHeight > 200;
    });
    scrollables.forEach(el => el.scrollTo({top: el.scrollHeight, behavior: 'instant'}));
    window.scrollTo({top: document.body.scrollHeight, behavior: 'instant'});
    return scrollables.length;
}
"""


@dataclass
class CollectResult:
    note_ids: list[str] = field(default_factory=list)
    pages: int = 0
    stopped_reason: str = ""
    captcha_url: str = ""


def _parse_count(v: Any) -> int:
    """互动数解析：接口返回可能是数字，也可能是 '10万+' '1.2万' '3千' 等中文单位。

    ★ 2026-08-31 实测：collect 接口的 liked_count 是字符串且带单位，
      直接 int() 会崩（ValueError 会把 Playwright 事件回调打挂，后续响应全丢）。
    """
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


def _norm_url(u: str) -> str:
    """响应 URL 可能是 edith.xiaohongshu.com 或 www.xiaohongshu.com，统一成纯路径。

    ★ 2026-08-31 修复：早先版本只去了 query 没去域名，返回完整 URL 与
      COLLECT_API_MARK（纯路径）永远不相等，导致所有响应都被丢掉 ——
      "抓不到收藏接口"的元凶，注释与实现长期不一致。
    """
    try:
        return urlsplit(u or "").path
    except Exception:
        return (u or "").split("?")[0]


def _extract_note(item: dict) -> dict:
    """从收藏列表单条 item 里提取标准化字段。

    ★ 宽容策略：接口字段没有公开文档，随时可能改名。
      这里全部用 .get() 多级兜底，缺字段填空，绝不因缺字段崩溃。
      原始 item 会原样落 jsonl，字段演进时从留档里重放即可。
    """
    note_id = str(item.get("id") or item.get("noteId") or item.get("note_id") or "").strip()
    if not note_id:
        return {}

    # 作者信息：常见两种位置
    user = item.get("user") or {}
    author_id = user.get("userId") or user.get("user_id") or ""
    author_name = user.get("nickname") or user.get("nickName") or ""

    # 互动数据（collect 接口实测是 interact_info 下划线版，且只有 liked_count）
    interact = item.get("interactInfo") or item.get("interact_info") or {}
    liked = interact.get("likedCount") or interact.get("liked_count") or 0
    collected = interact.get("collectedCount") or interact.get("collected_count") or 0
    commented = interact.get("commentCount") or interact.get("comment_count") or 0

    # 封面/图片：imageList 或 cover 对象（collect 接口实测是 cover，带 url_default/info_list）
    imgs = item.get("imageList") or item.get("images") or []
    cover = item.get("cover") or {}
    if not imgs and isinstance(cover, dict):
        imgs = [cover]
    img_urls = []
    for im in imgs:
        if isinstance(im, str):
            img_urls.append(im)
        elif isinstance(im, dict):
            u = (im.get("urlDefault") or im.get("url_default")
                 or im.get("url") or im.get("url_pre")
                 or im.get("info_list") or [{}])
            if isinstance(u, list):
                u = u[0].get("url") if u and isinstance(u[0], dict) else ""
            img_urls.append(u or "")
    img_urls = [u for u in img_urls if u]

    # xsec_token：位置不固定（display 里常见），深挖一层
    xsec_token = ""
    xsec_source = ""
    display = item.get("display") or {}
    if isinstance(display, dict):
        xsec_token = display.get("xsec_token") or display.get("xsecToken") or ""
        xsec_source = display.get("xsec_source") or display.get("xsecSource") or ""
    if not xsec_token:
        xsec_token = item.get("xsec_token") or ""
    if not xsec_source:
        xsec_source = item.get("xsec_source") or "pc_feed"

    # 发布时间：毫秒时间戳，部分接口给的是字符串秒；collect 接口实测无此字段
    published = item.get("time") or item.get("createTime") or 0
    try:
        published = int(published)
        if 0 < published < 10**11:      # 秒 → 毫秒
            published *= 1000
    except (TypeError, ValueError):
        published = 0

    note_type = item.get("type") or ("video" if item.get("video") else "normal")

    # collect 接口实测标题字段是 display_title
    title = item.get("title") or item.get("desc") or item.get("display_title") or ""
    desc = item.get("desc") or item.get("display_title") or ""

    return {
        "note_id": note_id,
        "title": str(title),
        "author_id": str(author_id) if author_id else None,
        "author_name": str(author_name) if author_name else None,
        "note_type": str(note_type),
        "desc": str(desc),
        "published_at": published,
        "liked_count": _parse_count(liked),
        "collect_count": _parse_count(collected),
        "comment_count": _parse_count(commented),
        "image_urls": img_urls,
        "xsec_token": str(xsec_token),
        "xsec_source": str(xsec_source),
        "url": (f"https://www.xiaohongshu.com/explore/{note_id}"
                f"{'?xsec_token=' + xsec_token if xsec_token else ''}"),
    }


class CollectCrawler:
    def __init__(self, ctx: BrowserContext, cfg: Any, user_id: str,
                 jsonl: Path | None = None):
        self.ctx = ctx
        self.cfg = cfg
        self.user_id = user_id
        self.jsonl = jsonl
        self.seen: set[str] = set()
        self.pages = 0
        self.empty_streak = 0
        self.stopped_reason = ""
        self._stop = False          # 状态标志：has_more=false 时置位（事件回调里不能用异常控制流）

    # ── 主流程 ────────────────────────────────────────────
    def sync(self, on_page: Callable[[dict], None] | None = None,
             max_pages: int = 200) -> CollectResult:
        """翻页抓完整收藏列表。on_page 每页回调一次（用于落库）。"""
        page = self.ctx.pages[0] if self.ctx.pages else self.ctx.new_page()

        # 响应监听：挂在 context 级别（CDP 多标签下 page 级可能漏接），
        # 只认收藏列表接口的 200 响应
        def on_response(resp: Response):
            if "xiaohongshu.com/api" in resp.url:
                logger.debug(f"[resp] {resp.url.split('?')[0]}")
            if _norm_url(resp.url) != COLLECT_API_MARK:
                return
            if resp.status != 200:
                logger.debug(f"收藏接口非 200: {resp.status}")
                return
            try:
                body = resp.json()
            except Exception:
                return
            try:
                self._handle_page(body, on_page)
            except Exception:
                # ★ 绝不能 re-raise：pyee 会把回调异常当 error 事件处理，
                #   导致后续所有响应不再分发（表现为"滚 30 次都没新内容"）。
                #   记录日志后忽略，_handle_page 内部已足够宽容。
                logger.exception("收藏响应处理异常（已忽略，防止事件分发中断）")

        self.ctx.on("response", on_response)

        try:
            # 先确认登录态还活着（零额外请求，页面加载时已渲染）
            uid = get_login_user_id(page)
            if not uid:
                # 页面还没加载用户信息，先访问一次首页/收藏页
                pass

            url = COLLECT_TAB_URL.format(uid=self.user_id)
            logger.info(f"打开收藏页 {url}")
            page.goto(url, wait_until="domcontentloaded", timeout=60_000)

            # ★ 2026-08-31 实测：?tab=collect 直连不触发收藏接口（URL 参数被前端忽略），
            #   必须等页面渲染后模拟点击「收藏」tab（点完 URL 会变成 ?tab=fav&subTab=note）
            #   点击后验证 URL 变化，没变则重试（React 事件可能还没挂载）
            page.wait_for_timeout(2500)
            switched = False
            for attempt in range(3):
                if self._click_collect_tab(page):
                    page.wait_for_timeout(2500)
                    if "tab=fav" in page.url:
                        switched = True
                        logger.info("已切换到「收藏」tab（URL 含 tab=fav）")
                        break
                    logger.warning(f"第 {attempt + 1} 次点击后 URL 未切换，重试…")
                else:
                    logger.warning("未找到「收藏」tab（页面结构可能变了）")
                    break
            if not switched:
                logger.warning("收藏 tab 未切换成功，尝试直接等响应…")

            # 等第一页响应
            deadline = time.time() + 25
            while self.pages == 0 and time.time() < deadline:
                page.wait_for_timeout(500)

            if self.pages == 0:
                self.stopped_reason = "未捕获到收藏接口响应"
                return self._result()

            # 滚动翻页：直到 has_more=false 或连续空转
            early_stop = int(self.cfg.get("sync.early_stop_streak", 30))
            max_empty = early_stop
            while self.pages < max_pages and not self._stop:
                if self._check_captcha(page):
                    self.stopped_reason = "触发验证码风控"
                    return self._result()

                prev = len(self.seen)
                page.evaluate(_SCROLL_DOWN)
                page.wait_for_timeout(self._jitter())

                # 更新空转计数（_handle_page 里也会做，这里兜底防止漏算）
                if len(self.seen) == prev:
                    self.empty_streak += 1
                else:
                    self.empty_streak = 0

                if self.empty_streak >= max_empty:
                    self.stopped_reason = f"连续 {max_empty} 次滚动无新内容"
                    break

            if not self.stopped_reason and self._stop:
                pass  # 原因已在 _handle_page 里写
            if not self.stopped_reason and self.pages >= max_pages:
                self.stopped_reason = "达到页数上限"

        except CaptchaRequired:
            self.stopped_reason = "触发验证码风控（CaptchaRequired）"
        except Exception as e:
            self.stopped_reason = f"异常: {type(e).__name__}: {e}"
            logger.exception("收藏采集异常")

        return self._result()

    # ── 内部 ──────────────────────────────────────────────
    def _handle_page(self, body: dict, on_page: Callable[[dict], None] | None) -> None:
        """处理一页响应：提取 items → 落 jsonl → 统计 → 回调。"""
        code = body.get("code")
        if code not in (0, "0", None):
            logger.warning(f"收藏接口业务错误: code={code} msg={body.get('msg')}")
            return

        data = body.get("data") or {}
        # ★ 2026-08-31 实测：列表字段名从 items 变成了 notes（code=0 但 items 恒空）
        items = data.get("items") or data.get("notes") or []
        has_more = data.get("has_more", False)
        cursor = data.get("cursor", "")

        self.pages += 1
        new_ids: list[str] = []
        raw_items: list[dict] = []

        for item in items:
            note = _extract_note(item)
            if not note:
                continue
            raw_items.append(item)
            nid = note["note_id"]
            if nid not in self.seen:
                self.seen.add(nid)
                new_ids.append(nid)

        if self.jsonl:
            DB_APPEND(self.jsonl, {
                "page": self.pages,
                "cursor": cursor,
                "has_more": has_more,
                "items": raw_items,
            })

        if on_page:
            for item in raw_items:
                note = _extract_note(item)
                if note:
                    on_page(note)

        if new_ids:
            self.empty_streak = 0
        else:
            self.empty_streak += 1

        logger.info(
            f"[{self.pages:>3}] 本页 {len(items)} 条，新增 {len(new_ids)}，"
            f"累计 {len(self.seen)}，has_more={has_more}，cursor={str(cursor)[:12]}"
        )

        # has_more=false 时置状态标志，主循环下一轮自然退出
        # （事件回调里不能用异常控制流 —— 异常会丢失或传播到不可预期的位置）
        if not has_more:
            self._stop = True
            self.stopped_reason = "has_more=false，收藏列表已到底"

    @staticmethod
    def _click_collect_tab(page: Page) -> bool:
        """点击 profile 页的「收藏」tab，触发 collect/page 请求。

        实测（2026-08-31）：URL ?tab=collect 直连不生效，页面停在默认 tab，
        必须点一下。失败只 warning 不崩，兼容页面结构变化。
        """
        try:
            ok = page.evaluate(
                """() => {
                    const all = [...document.querySelectorAll('div, span, a, li')];
                    const target = all.find(e => {
                        const t = (e.textContent || '').trim();
                        return (t === '收藏' || t === '我的收藏')
                            && e.getBoundingClientRect().width > 0;
                    });
                    if (target) { target.click(); return true; }
                    return false;
                }"""
            )
            return bool(ok)
        except Exception as e:
            logger.debug(f"点击收藏 tab 失败: {e}")
            return False

    def _check_captcha(self, page: Page) -> bool:
        try:
            url = page.url or ""
        except Exception:
            return False
        if CAPTCHA_MARK in url:
            raise CaptchaRequired(url)
        return False

    @staticmethod
    def _jitter() -> int:
        """翻页间隔高斯抖动：1.2~2.8 秒，像人浏览。"""
        return int(random.gauss(2.0, 0.5) * 1000 + 500)

    def _result(self) -> CollectResult:
        return CollectResult(
            note_ids=sorted(self.seen),
            pages=self.pages,
            stopped_reason=self.stopped_reason,
        )


def DB_APPEND(path: Path, obj: Any) -> None:
    """轻量 jsonl 追加（避免引入 store 依赖导致循环 import）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
