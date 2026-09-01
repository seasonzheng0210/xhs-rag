"""浏览器上下文：指纹恒定 + 反自动化检测。

★ 全项目最关键的一个约束：必须用 launch_persistent_context，不能用 new_context。
  a1 / webId 这些设备指纹 Cookie 写在 user_data_dir 里；每次新开 context 指纹都会漂移，
  小红书会把你判定成「换设备登录」，轻则弹验证码，重则整号风控。
  profile 目录必须长期保留，换一次目录等于重新建立信任。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger
from playwright.sync_api import Browser, BrowserContext, Playwright, sync_playwright

from ..core.config import ROOT, Config

STEALTH_SCRIPT = ROOT / "assets" / "stealth.min.js"

# 常见的「我是自动化」痕迹，逐条关掉
LAUNCH_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--disable-features=IsolateOrigins,site-per-process",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-infobars",
    "--lang=zh-CN",
]


class BrowserSession:
    """包装 Playwright 上下文，屏蔽 persistent / CDP 两种模式的差异。

    用法：
        with BrowserSession(cfg) as ctx:
            page = ctx.new_page()
            page.goto(...)
    """

    def __init__(self, config: Config, headless: bool | None = None, save_state: bool = True):
        self.config = config
        self.headless = config.get("auth.headless", False) if headless is None else headless
        self.save_state = save_state
        self._pw: Playwright | None = None
        self._browser: Browser | None = None
        self.context: BrowserContext | None = None
        self._cdp = False

    # ──────────────────────────────────────────────────────
    def __enter__(self) -> BrowserContext:
        self._pw = sync_playwright().start()
        pw = self._pw

        if self.config.get("browser.use_cdp", False):
            endpoint = self.config.get("browser.cdp_endpoint", "http://127.0.0.1:9222")
            logger.info(f"CDP 模式连接 {endpoint}")
            self._browser = pw.chromium.connect_over_cdp(endpoint)
            self.context = self._browser.contexts[0]
            self._cdp = True
            return self.context

        profile_dir: Path = self.config.path("paths.browser_profile")
        profile_dir.mkdir(parents=True, exist_ok=True)

        kwargs: dict[str, Any] = dict(
            user_data_dir=str(profile_dir),
            headless=self.headless,
            args=LAUNCH_ARGS,
            viewport={"width": self.config.get("browser.viewport", [1440, 900])[0],
                      "height": self.config.get("browser.viewport", [1440, 900])[1]},
            locale=self.config.get("browser.locale", "zh-CN"),
            timezone_id=self.config.get("browser.timezone", "Asia/Shanghai"),
            ignore_https_errors=True,
        )
        ua = self.config.get("browser.user_agent", "")
        if ua:
            kwargs["user_agent"] = ua

        self.context = pw.chromium.launch_persistent_context(**kwargs)
        restored = self._restore_cookies(self.context)

        if STEALTH_SCRIPT.exists():
            self.context.add_init_script(path=str(STEALTH_SCRIPT))
        else:
            logger.warning(f"未找到反检测脚本 {STEALTH_SCRIPT}，navigator.webdriver 会暴露")

        logger.debug(f"浏览器就绪（profile={profile_dir}，注入 cookie {restored} 个）")
        return self.context

    # ──────────────────────────────────────────────────────
    def _restore_cookies(self, ctx: BrowserContext) -> int:
        """启动时把 storage_state.json 里的 cookie 重新灌回去。

        ★ 为什么必须有这一步：实测发现小红书会在响应里下发删除指令清掉
          web_session，profile 的 Cookies 库随之被掏空，下次开浏览器就是未登录状态。
          只靠 persistent profile 等于把登录态交给服务端处置 —— 必须有本地副本兜底。
        """
        if not self.config.get("browser.restore_cookies", True):
            return 0
        state_file: Path = self.config.path("paths.storage_state")
        if not state_file.exists():
            return 0
        try:
            data = json.loads(state_file.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"登录态文件读取失败，跳过注入: {e}")
            return 0

        cookies = data.get("cookies") or []
        ok = 0
        for c in cookies:
            try:
                ctx.add_cookies([c])
                ok += 1
            except Exception:
                # 个别字段不合规的 cookie 跳过即可，不影响主体
                continue
        if ok:
            logger.debug(f"已从 {state_file.name} 注入 {ok}/{len(cookies)} 个 cookie")
        return ok

    # ──────────────────────────────────────────────────────
    def save_storage_state(self) -> Path | None:
        """登录态双写之一：导出 cookie/localStorage 到 JSON。

        profile 目录是主，这份 JSON 是备份 —— 换机器迁移、profile 损坏时用得上。
        """
        if not self.context or self._cdp:
            return None
        path: Path = self.config.path("paths.storage_state")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.context.storage_state(path=str(path))
        return path

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if self.save_state and not (exc_type and self._cdp):
                p = self.save_storage_state()
                if p:
                    logger.debug(f"登录态已写入 {p}")
        except Exception as e:
            logger.warning(f"保存登录态失败（不影响已抓数据）: {e}")
        finally:
            if self.context and not self._cdp:
                self.context.close()
            if self._pw:
                self._pw.stop()

    async def __aenter__(self):  # pragma: no cover - 占位，避免误用
        raise NotImplementedError("当前只实现同步接口")
