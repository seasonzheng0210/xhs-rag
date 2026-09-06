"""浏览器上下文：指纹恒定 + 反自动化检测。

★ 全项目最关键的一个约束：必须用 launch_persistent_context，不能用 new_context。
  a1 / webId 这些设备指纹 Cookie 写在 user_data_dir 里；每次新开 context 指纹都会漂移，
  小红书会把你判定成「换设备登录」，轻则弹验证码，重则整号风控。
  profile 目录必须长期保留，换一次目录等于重新建立信任。

★ 登录态写回的事务式规则（2026-09-06 借鉴 workbuddy-switch 的"先备份后写入"）：
  storage_state.json 只在「会话正常结束 + cookie 仍含 web_session」时才允许覆盖，
  且覆盖前先把旧文件备份成 storage_state.json.bak。
  登录失败/超时、会话异常、web_session 已被服务端作废 → 一律跳过覆盖，保留旧文件，
  避免把中间态/空登录态写回（曾致「假登录视图」：首页渲染成已登录、login 找不到入口）。
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from loguru import logger
from playwright.sync_api import Browser, BrowserContext, Playwright, sync_playwright

from ..core.config import ROOT, Config

STEALTH_SCRIPT = ROOT / "assets" / "stealth.min.js"

# 小红书会话 cookie 名（与 auth/qrcode_login.SESSION_COOKIE 同值）。
# 这里不复用该常量，避免 browser.py 反向依赖 auth 模块（browser 是底层，被 auth 依赖）。
SESSION_COOKIE_NAME = "web_session"

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
        # 事务式写回开关：会话正常结束且 web_session 仍在 → 备份旧文件后覆盖。
        # login 失败/超时调用方应置 False，防止中间态 cookie 污染旧登录态。
        self.allow_state_commit = True
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
    def _backup_state(self) -> Path | None:
        """覆盖前把现有 storage_state.json 备份为 .bak（事务式写回的第一步）。

        借鉴 workbuddy-switch「切换前先备份认证文件」：万一新写入的登录态有问题，
        可用 .bak 一键回滚，不必重新扫码。只保留最近一份，滚动覆盖。
        """
        src: Path = self.config.path("paths.storage_state")
        if not src.exists():
            return None
        try:
            bak = src.with_name(src.name + ".bak")
            shutil.copy2(src, bak)
            logger.debug(f"旧登录态已备份 → {bak}")
            return bak
        except Exception as e:
            logger.warning(f"备份旧登录态失败（继续写回）: {e}")
            return None

    def save_storage_state(self) -> Path | None:
        """登录态双写之一：导出 cookie/localStorage 到 JSON。

        profile 目录是主，这份 JSON 是备份 —— 换机器迁移、profile 损坏时用得上。

        ★ 事务式写回（2026-09-06）：覆盖前先备份旧文件到 .bak。
          只有调用方确认会话有效（allow_state_commit=True 且 cookie 含 web_session）
          才走到这里；本方法只负责「备份旧 → 写新」的原子性。
        """
        if not self.context or self._cdp:
            return None
        self._backup_state()
        path: Path = self.config.path("paths.storage_state")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.context.storage_state(path=str(path))
        return path

    def _cookie_has_session(self) -> bool:
        """当前浏览器上下文里是否还有 web_session cookie。

        注意这只是本地快判：cookie 在 ≠ 服务端认可（服务端可随时作废 session）。
        但反过来，cookie 没了 = 登录态必然已丢（被服务端下发删除指令清空），
        此时绝不能把空登录态写回覆盖旧文件。
        """
        if not self.context:
            return False
        try:
            return any(c.get("name") == SESSION_COOKIE_NAME for c in self.context.cookies())
        except Exception:
            return False

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            # CDP 模式：上下文属于外部 Chrome，绝不动它的登录态文件
            if self._cdp:
                return

            # 会话异常退出（抛异常穿过 with 块）：不写回。
            # 异常往往伴随风控/断网/半截状态，写回 = 用坏状态覆盖好状态。
            if exc_type is not None:
                logger.warning("会话异常退出，跳过登录态写回（保留原文件）")
                return

            if not self.save_state:
                return

            # 调用方显式禁止提交（login 失败/超时）：防止中间态 cookie 污染旧登录态
            if not self.allow_state_commit:
                logger.warning(
                    "登录未成功，跳过登录态写回 —— 保留原 storage_state.json "
                    "（避免 09-03「假登录视图」复发：中间态覆盖导致 login 找不到入口）"
                )
                return

            # cookie 里已无 web_session（服务端作废/被清空）：空登录态不覆盖旧文件
            if not self._cookie_has_session():
                logger.warning(
                    "会话中已无 web_session cookie（疑似登录态被服务端作废），"
                    "跳过登录态写回，保留原文件"
                )
                return

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
