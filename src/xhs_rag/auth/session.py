"""登录态健康检查：每次同步前先跑一遍，别抓到一半才发现掉线。

判定顺序（从便宜到贵）：
  1. storage_state.json 里有没有 web_session      —— 不启浏览器，毫秒级
  2. 启 headless 浏览器带 profile 访问首页复验    —— 慢但准，cookie 可能已过期
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from loguru import logger

from ..core.config import Config
from ..crawler.browser import BrowserSession
from .qrcode_login import (  # 常量与判定都在这边，session 单向依赖它
    CAPTCHA_MARK,
    HOME,
    SESSION_COOKIE,
    get_login_user_id,
    is_logged_in,
    looks_like_captcha,
)


class CaptchaRequired(Exception):
    """触发验证码风控。

    必须单独成类：如果和普通「登录态失效」混为一谈，上层会去自动重新扫码，
    而扫码本身又会加重风控 —— 越扫越严，形成死循环。
    """


def has_stored_session(config: Config) -> tuple[bool, str]:
    """只读本地文件判断，不启动浏览器。"""
    p: Path = config.path("paths.storage_state")
    if not p.exists():
        return False, f"无登录态文件（{p}）"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"登录态文件解析失败: {e}"

    cookies = data.get("cookies") or []
    hit = [c for c in cookies if c.get("name") == SESSION_COOKIE]
    if not hit:
        return False, "登录态文件中没有 web_session"
    return True, "存在 web_session"


def verify_online(config: Config, headless: bool = True) -> tuple[bool, str]:
    """真正启一次浏览器验证 cookie 还有效。

    ★ 2026-08-31 修复：早先版本开 explore 首页等 1.5s 读 __INITIAL_STATE__.user，
      但 PC 首页的登录用户是异步 /user/me 加载的，等不到就误报「登录无效」。
      改为直接探自己的收藏接口（与 M1 采集同一条路）：只有登录者能看到
      自己的收藏，collect/page 返回 code=0 即服务端认可，零误判。

    注意：每跑一次就是一次真实访问。频繁调用本身就是风控诱因，
    正常流程里每天最多跑一两次，别拿它当心跳用。
    """
    from ..crawler.collect import COLLECT_API_MARK  # 常量，函数内延迟 import 避开循环

    uid = str(config.get("auth.user_id", "") or "").strip()
    profile_url = f"https://www.xiaohongshu.com/user/profile/{uid}?tab=collect"
    # 与 collect.py 的点击逻辑保持一致（避免依赖 collect 模块造成循环 import）
    CLICK_FAV_JS = """
    () => {
        const all = [...document.querySelectorAll('div, span, a, li')];
        const target = all.find(e => {
            const t = (e.textContent || '').trim();
            return (t === '收藏' || t === '我的收藏') && e.getBoundingClientRect().width > 0;
        });
        if (target) { target.click(); return true; }
        return false;
    }
    """

    got: dict = {"code": None, "count": 0}

    try:
        with BrowserSession(config, headless=headless, save_state=False) as ctx:
            page = ctx.pages[0] if ctx.pages else ctx.new_page()

            def on_response(resp) -> None:
                try:
                    if resp.url.split("?")[0].endswith(COLLECT_API_MARK) and resp.status == 200:
                        got["code"] = resp.json().get("code")
                except Exception:
                    pass

            ctx.on("response", on_response)
            page.goto(profile_url, wait_until="domcontentloaded", timeout=60_000)
            page.wait_for_timeout(2500)

            # 直连不触发接口（2026-08-31 实测），必须点一下「收藏」tab
            try:
                page.evaluate(CLICK_FAV_JS)
            except Exception:
                pass

            deadline = time.time() + 12
            while got["code"] is None and time.time() < deadline:
                page.wait_for_timeout(500)
                if looks_like_captcha(page.url):
                    break

            if got["code"] == 0:
                return True, "服务端认可登录态（收藏接口 code=0）"
            if looks_like_captcha(page.url):
                return False, (
                    f"触发验证码风控（跳转到 {CAPTCHA_MARK}）\n"
                    "        这不是登录态过期，是访问行为被判定为自动化。\n"
                    "        此时【不要】反复重扫 —— 越扫风控越严。\n"
                    "        处理：用自己日常用的浏览器正常登录并浏览几分钟，冷却后再试。"
                )
            if "/login" in (page.url or ""):
                return False, "被重定向到登录页 —— 登录态已失效，需要重新登录"
            return False, f"收藏接口未返回数据（code={got['code']}），疑似登录态失效"
    except Exception as e:
        return False, f"在线验证异常: {type(e).__name__}: {e}"


def check(config: Config, online: bool = True, headless: bool | None = None) -> tuple[bool, str]:
    """headless 默认跟随 auth.headless（即采集时用的模式）。

    早先固定用 headless=True 做在线验证，等于拿一个「和真实采集不同的环境」去判定
    登录态，结论不可信 —— headless 更容易触发风控，还可能误报成 cookie 过期。
    """
    if headless is None:
        headless = bool(config.get("auth.headless", False))
    ok, reason = has_stored_session(config)
    if not ok or not online:
        return ok, reason
    return verify_online(config, headless=headless)


def report(config: Config, online: bool = True) -> bool:
    """打印检查结果并返回是否可用。"""
    ok, reason = check(config, online=online)
    if ok:
        logger.success(f"登录态有效 —— {reason}")
    else:
        logger.warning(f"登录态无效 —— {reason}")
        logger.info("重新扫码：python -m xhs_rag.cli login")
    return ok
