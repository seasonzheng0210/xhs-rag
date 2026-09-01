"""扫码登录。

两条获取二维码的路，优先走接口嗅探：
  ① 嗅探 /api/sns/web/v1/user/qrcode 的响应 → 拿到二维码内容字符串 → 本地重绘
     （好处：能直接在终端渲染 ASCII 二维码，不必盯着浏览器窗口；二维码过期自动重绘）
  ② 兜底：直接截取页面上二维码元素的图，存成 PNG 让用户扫

登录成功的判定用 cookie 里出现 web_session，比看 DOM/URL 稳 ——
小红书的前端结构经常改，但 session cookie 的名字不会。
"""
from __future__ import annotations

import io
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import qrcode
from loguru import logger
from playwright.sync_api import BrowserContext, Page

HOME = "https://www.xiaohongshu.com/explore"

# ★ 实测抓包得到（2026-08-31）：真正的二维码接口是 login/qrcode/create，
#   网上流传的 /api/sns/web/v1/user/qrcode 早已不存在。
#   用错路径的后果很隐蔽：嗅探永远不命中，只能退化成 DOM 截图，
#   终端就不会渲染 ASCII 二维码 —— 表现为「一直在等扫码但没码可扫」。
#   把这个常量写对，是「能不能在终端直接扫码」的分水岭。
QRCODE_API_MARK = "/api/sns/web/v1/login/qrcode/create"
QRCODE_STATUS_MARK = "/api/qrcode/userinfo"      # 扫码状态轮询接口
SESSION_COOKIE = "web_session"

# 跳到这个路径 = 被风控要求过验证码。
# 常量定义在这里（而不是 session.py）是为了保持单向依赖：
# session → qrcode_login，反过来会形成循环 import。
CAPTCHA_MARK = "/website-login/captcha"


def looks_like_captcha(url: str) -> bool:
    return CAPTCHA_MARK in (url or "")


def _find_qr_content(obj: Any, strict: bool = True) -> str | None:
    """从任意结构的响应体里递归找出二维码内容。

    两轮扫描：先只认 xhslink.com（小红书二维码专用短链，几乎不会误判），
    找不到再放宽到任意 http(s) 链接。
    这么做是因为响应体结构没公开文档，写死字段名的提取方式一遇改名就失效。
    """
    if isinstance(obj, str):
        if "xhslink" in obj:
            return obj
        return obj if (not strict and obj.startswith("http")) else None

    if isinstance(obj, dict):
        for key in ("url", "qr_url", "qrcode", "qr_code", "content", "code"):
            if key in obj:
                hit = _find_qr_content(obj[key], strict)
                if hit:
                    return hit
        for v in obj.values():
            hit = _find_qr_content(v, strict)
            if hit:
                return hit
    elif isinstance(obj, list):
        for v in obj:
            hit = _find_qr_content(v, strict)
            if hit:
                return hit
    return None


def wait_for_captcha(page: Page, timeout: int = 300) -> bool:
    """卡在验证码页时，等着人在浏览器窗口里手动过。

    ★ 为什么必须有这一段：headful 模式下窗口是可见的，验证码本来就能人工过掉。
      缺了它，遇到 captcha 只能退出 → 重跑 → 再撞风控，形成死循环。
      有它，一次弹窗就能解决，还避免了额外请求带来的风控累加。

    返回 True = 已通过（URL 离开了验证码页）。
    """
    if not looks_like_captcha(page.url):
        return True

    print("\n" + "!" * 58)
    print("  ⚠ 触发验证码风控")
    print("")
    print("  请在刚刚弹出的浏览器窗口里手动完成验证（拖滑块/点选）。")
    print("  完成后这里会自动继续，不用重跑命令。")
    print("!" * 58 + "\n")
    logger.warning("等待人工完成验证码…")

    deadline = time.time() + timeout
    while time.time() < deadline:
        page.wait_for_timeout(1000)
        try:
            if not looks_like_captcha(page.url):
                logger.success("验证码已通过，继续登录流程")
                return True
        except Exception:
            continue

    logger.error(f"{timeout} 秒内验证码未通过")
    return False


@dataclass
class LoginResult:
    ok: bool
    user_id: str = ""
    reason: str = ""
    qr_png: Path | None = None


# ── 二维码渲染 ────────────────────────────────────────────
def render_qr(content: str, png_path: Path) -> str:
    """生成二维码，返回终端可打印的 ASCII 串，同时落一份 PNG。"""
    qr = qrcode.QRCode(border=1)
    qr.add_data(content)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")

    png_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(png_path)

    # 同步落一份纯文本：ASCII 图在终端里经常被截断/看不清，
    # 外部工具（比如聊天窗口内联渲染二维码）需要拿到原始内容而不是图片
    png_path.with_suffix(".txt").write_text(content, encoding="utf-8")

    buf = io.StringIO()
    qr.print_ascii(out=buf, invert=True)
    return buf.getvalue()


# ── 主流程 ────────────────────────────────────────────────
def qrcode_login(
    ctx: BrowserContext,
    timeout: int = 180,
    png_path: Path | None = None,
    on_qr: Callable[[str, Path], None] | None = None,
) -> LoginResult:
    """在给定 context 内完成扫码登录。已登录则直接返回。

    ★ 判定「是否已登录」必须问服务端，不能只看 cookie：
      启动时我们会把 storage_state 的 cookie 重新注入（restore_cookies），
      于是 cookie 永远存在 —— 若据此判定，每次 login 都会「秒登录成功」跳过扫码，
      实际上服务端早把这个 session 作废了。这个假阳性坑过一次。
    """
    page = ctx.pages[0] if ctx.pages else ctx.new_page()

    try:
        page.goto(HOME, wait_until="domcontentloaded", timeout=60_000)
        page.wait_for_timeout(2000)
        if not looks_like_captcha(page.url):
            uid = get_login_user_id(page)
            if uid:
                logger.success("已有有效登录态（服务端认可），跳过扫码")
                return LoginResult(ok=True, user_id=uid, reason="复用已有登录态")
    except Exception as e:
        logger.debug(f"登录态预检未通过，进入扫码流程: {e}")

    captured: dict = {}
    # 疑似二维码/登录相关的接口，抓不到时打印出来便于定位接口改名
    candidates: set[str] = set()

    def on_response(resp):
        url = resp.url or ""
        if any(k in url.lower() for k in ("qrcode", "qr_code", "login", "signin")):
            candidates.add(url[:140])
        if QRCODE_API_MARK in url and resp.status == 200:
            try:
                data = resp.json()
            except Exception:
                return
            qr_url = _find_qr_content(data, strict=True) or _find_qr_content(data, strict=False)
            if qr_url:
                captured["url"] = qr_url
                captured["ts"] = time.time()
                logger.debug(f"嗅探到二维码内容: {qr_url[:60]}…")

    page.on("response", on_response)

    logger.info("打开小红书首页…")
    page.goto(HOME, wait_until="domcontentloaded", timeout=60_000)
    time.sleep(2)

    # 一上来就撞验证码的话，先让人在窗口里过掉，别急着放弃重来
    if not wait_for_captcha(page, timeout):
        return LoginResult(ok=False, reason="验证码未通过")
    time.sleep(1)

    if not click_login_entry(page):
        return LoginResult(ok=False, reason="没找到登录入口，页面结构可能变了")

    qr_png = png_path or Path("data/auth/qrcode.png")
    shown_url = ""
    deadline = time.time() + timeout
    last_dom_shot = 0.0
    DOM_REFRESH_SEC = 45      # 二维码有效期通常 3-5 分钟，定期重截避免用户扫到过期的
    SCANNED_HINT = False

    logger.info(f"等待扫码（{timeout} 秒内有效）…")
    while time.time() < deadline:
        # cookie 是快判（本地，零成本），服务端认可是定判 —— 两者都过才算登录成功
        if is_logged_in(ctx) and get_login_user_id(page):
            logger.success("登录成功（服务端已认可）")
            return LoginResult(ok=True, user_id=get_login_user_id(page) or "",
                               qr_png=qr_png)

        # 扫码途中也可能被弹验证码，同样等人过掉后继续
        try:
            if looks_like_captcha(page.url):
                remaining = int(deadline - time.time())
                if remaining <= 0 or not wait_for_captcha(page, min(120, remaining)):
                    break
                continue
        except Exception:
            pass

        # 接口嗅探优先：拿到二维码内容就能本地重绘，还能终端打印
        if captured.get("url") and captured["url"] != shown_url:
            shown_url = captured["url"]
            ascii_art = render_qr(shown_url, qr_png)
            print("\n" + "=" * 56)
            print("  用小红书 App 扫描下方二维码（或打开 PNG）")
            print("=" * 56)
            print(ascii_art)
            print(f"二维码图片：{qr_png.resolve()}")
            print("=" * 56 + "\n")
            if on_qr:
                on_qr(shown_url, qr_png)

        elif not shown_url or shown_url == "<dom>":
            need_shot = (not shown_url) or (time.time() - last_dom_shot > DOM_REFRESH_SEC)
            if need_shot and capture_qr_from_dom(page, qr_png):
                last_dom_shot = time.time()
                if not shown_url:
                    shown_url = "<dom>"
                    logger.info(f"二维码已存为 {qr_png.resolve()}，请打开扫描")
                    logger.info("（接口嗅探未命中，走 DOM 截图；每 45 秒自动刷新一次）")
                else:
                    logger.info("二维码已刷新，请重新扫描最新图片")

        page.wait_for_timeout(1000)

    # 超时：把抓到的候选接口打印出来，方便定位「接口改名」这个最常见的失效原因
    if candidates:
        logger.info("本次抓到的登录相关接口（用于排查二维码接口路径）：")
        for u in sorted(candidates)[:12]:
            logger.info(f"  {u}")
    return LoginResult(ok=False, reason=f"{timeout} 秒内未扫码", qr_png=qr_png)


def click_login_entry(page: Page) -> bool:
    """点开登录弹窗。小红书改版频率高，这里多做几手准备。"""
    selectors = [
        "text=登录",
        ".login-btn",
        ".side-bar .login",
        "button:has-text('登录')",
    ]
    for sel in selectors:
        try:
            loc = page.locator(sel).first
            if loc.count() and loc.is_visible():
                loc.click(timeout=3000)
                page.wait_for_timeout(1500)
                return True
        except Exception:
            continue

    # 登录入口可能在弹窗关闭后需要直接跳登录页
    try:
        page.goto(HOME, wait_until="domcontentloaded")
        page.wait_for_timeout(1000)
        loc = page.locator("text=登录").first
        if loc.count():
            loc.click(timeout=3000)
            return True
    except Exception:
        pass
    return False


def capture_qr_from_dom(page: Page, png_path: Path) -> bool:
    """兜底方案：把页面上的二维码元素截图存盘。"""
    candidates = [
        ".qrcode-img",
        "canvas",
        ".login-container img",
        "img[class*='qrcode']",
        ".reds-mask img",
    ]
    for sel in candidates:
        try:
            loc = page.locator(sel).first
            if loc.count() and loc.is_visible():
                png_path.parent.mkdir(parents=True, exist_ok=True)
                loc.screenshot(path=str(png_path))
                return True
        except Exception:
            continue
    return False


# ── 状态检查 ──────────────────────────────────────────────
def is_logged_in(ctx: BrowserContext) -> bool:
    try:
        return any(c["name"] == SESSION_COOKIE for c in ctx.cookies())
    except Exception:
        return False


USER_ID_RE = re.compile(r"/user/profile/([0-9a-fA-F]{24})")
PROFILE_ME = "https://www.xiaohongshu.com/user/profile/me"

# ★ 判定「服务端认可」的唯一可靠依据：服务端 rendered 出来的用户对象。
#   只查 cookie 会得出假阳性 —— cookie 是我们自己注入的，浏览器端有，
#   但服务端可能早就把这个 session 作废了（实测就撞上了：cookie 在、
#   /user/profile/me 却 302 到 /login）。
#   页面已加载时调用它是零额外请求，不会累加风控。
_JS_LOGIN_STATE = """
() => {
  try {
    const s = window.__INITIAL_STATE__;
    const u = s && s.user;
    if (!u) return null;
    const info = u.userInfo || u.user || u;
    if (info && info.id) return String(info.id);
    if (u.userIdFromMct) return String(u.userIdFromMct);
    if (u.id) return String(u.id);
  } catch (e) {}
  return null;
}
"""


def get_login_user_id(page: Page) -> str | None:
    """在当前已加载的页面上，问服务端「我是谁」。

    返回 None = 未登录（或服务端不认这个 session）。
    页面没加载过时返回 None，调用方需先 goto。
    """
    try:
        return page.evaluate(_JS_LOGIN_STATE)
    except Exception:
        return None


def extract_user_id(ctx: BrowserContext) -> str | None:
    """挖 user_id（24 位十六进制），顺带验证服务端是否认可登录态。

    优先在首页 evaluate —— 首页一定会渲染当前用户，一次访问同时完成
    「判登录」和「取 ID」两件事，不额外增加请求。
    """
    page = ctx.pages[0] if ctx.pages else ctx.new_page()

    try:
        page.goto(HOME, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_timeout(2000)
        uid = get_login_user_id(page)
        if uid:
            return uid
        m = USER_ID_RE.search(page.url)      # 万一停在个人页
        if m:
            return m.group(1)
    except Exception:
        pass

    # 兜底：profile/me 的 302 目标。仅当首页没拿到时尝试
    try:
        page.goto(PROFILE_ME, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_timeout(1500)
        m = USER_ID_RE.search(page.url)
        if m:
            return m.group(1)
    except Exception:
        pass
    return None
