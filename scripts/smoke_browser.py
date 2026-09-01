"""浏览器冒烟测试 —— 不扫码就能验证 M0 的两个关键点：

1. persistent context 能正常启动（user_data_dir 可写、Chromium 可用）
2. stealth 脚本真的生效（navigator.webdriver 已抹掉、window.chrome 已补齐）

第 2 点必须验证：脚本注入失败时 Playwright 不会报错，只会让你在后面
莫名其妙地撞风控，等到发现时已经浪费了一整轮采集。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loguru import logger  # noqa: E402

from xhs_rag.core.config import load_config  # noqa: E402
from xhs_rag.core.logging import setup_logging  # noqa: E402
from xhs_rag.crawler.browser import BrowserSession  # noqa: E402

CHECKS = {
    "navigator.webdriver 已抹除": "() => navigator.webdriver === undefined",
    "window.chrome 已注入": "() => !!window.chrome && !!window.chrome.runtime",
    "plugins 非空": "() => navigator.plugins.length > 0",
    "languages 为中文": "() => (navigator.languages || []).includes('zh-CN')",
    "WebGL 非 SwiftShader": r"""() => {
        const c = document.createElement('canvas');
        const gl = c.getContext('webgl') || c.getContext('experimental-webgl');
        if (!gl) return true;  // 无 WebGL 环境时跳过
        const dbg = gl.getExtension('WEBGL_debug_renderer_info');
        const r = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
        return !!r && !String(r).includes('SwiftShader');
    }""",
}


def main() -> int:
    cfg = load_config()
    setup_logging("INFO", None)

    logger.info("启动 persistent context（headless）…")
    with BrowserSession(cfg, headless=True, save_state=False) as ctx:
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto("about:blank")

        passed = 0
        print("\n反检测自检\n" + "-" * 52)
        for name, expr in CHECKS.items():
            try:
                ok = bool(page.evaluate(expr))
            except Exception as e:
                ok = False
                name = f"{name}（异常: {e}）"
            print(f"[{'PASS' if ok else 'FAIL'}] {name}")
            passed += ok

        ua = page.evaluate("() => navigator.userAgent")
        print("-" * 52)
        print(f"UA: {ua}")
        if "HeadlessChrome" in ua:
            print("注：headless 模式下 UA 含 HeadlessChrome 属正常。")
            print("    正式采集用 auth.headless=false（默认值），UA 即为常规 Chrome。")
        print()

        total = len(CHECKS)
        print(f"结果：{passed}/{total} 通过")
        if passed < total:
            logger.warning("有检查项未通过 —— 说明 stealth 脚本没生效或版本不兼容，别急着往下走")
            return 1

    logger.success("浏览器环境就绪，可以进行扫码登录")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
