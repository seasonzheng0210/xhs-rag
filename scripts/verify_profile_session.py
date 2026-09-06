"""裸启 profile 验证登录态（不注入 storage_state 旧 cookie）。

背景：BrowserSession.__enter__ 的 _restore_cookies 会把旧的失效 cookie 灌回 profile，
可能覆盖扫码刚种下的新 web_session。本脚本绕过 BrowserSession，直接用 playwright
launch_persistent_context 打开 profile，只依赖 profile 磁盘上已有的 cookie。

用法：
    python scripts/verify_profile_session.py
成功：打印 user_id，可选 --export 导出 storage_state.json
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from playwright.sync_api import sync_playwright  # noqa: E402

from xhs_rag.core.config import load_config, save_user_id  # noqa: E402

HOME = "https://www.xiaohongshu.com/explore"
LAUNCH_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--disable-features=IsolateOrigins,site-per-process",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-infobars",
    "--lang=zh-CN",
]

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


def main() -> int:
    cfg = load_config()
    export = "--export" in sys.argv
    profile_dir: Path = cfg.path("paths.browser_profile")
    state_file: Path = cfg.path("paths.storage_state")

    logger.info(f"裸启 profile（不注入旧 cookie）: {profile_dir}")
    with sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            headless=False,
            args=LAUNCH_ARGS,
            viewport={"width": 1440, "height": 900},
            locale="zh-CN",
            timezone_id="Asia/Shanghai",
            ignore_https_errors=True,
        )
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(HOME, wait_until="domcontentloaded", timeout=60_000)
        page.wait_for_timeout(3000)
        uid = page.evaluate(_JS_LOGIN_STATE)
        if not uid:
            logger.error("profile 里没有服务端认可的登录态 —— 需要重新扫码")
            ctx.close()
            return 1
        logger.success(f"服务端认可登录态，user_id = {uid}")

        if export:
            if state_file.exists():
                bak = state_file.with_name(state_file.name + ".bak")
                shutil.copy2(state_file, bak)
                logger.info(f"旧登录态已备份 → {bak}")
            ctx.storage_state(path=str(state_file))
            logger.success(f"登录态已导出 → {state_file}")
            save_user_id(cfg, uid)
            logger.info(f"user_id 已写入 config.yaml: {uid}")
        ctx.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
