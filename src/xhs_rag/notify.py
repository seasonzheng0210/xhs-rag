"""失败告警（P1-1）：激活 config 中长期未落地的 notify_on_expire / notify_email。

设计原则（本地优先 + 零第三方依赖）：
1. **多通道，逐级降级**，任一通道失败都不影响主流程（best-effort）：
   - ① loguru 日志（始终）
   - ② `data/alerts.json` 落盘（保留最近 MAX_KEEP 条）—— Web UI 与 CLI 都读它
   - ③ 桌面通知（Windows 10+ WinRT toast，经 PowerShell 调用；失败静默）
   - ④ 邮件（配置 `auth.notify_email` 时才尝试；本版仅预留，未配置则跳过）
2. **告警只记事实，不做决策**：是否该告警由调用方判定（如同步失败、登录态失效）。
3. 排重：相同 (kind, message) 在 DEDUP_WINDOW 秒内只记一次，避免定时任务反复刷屏。

消费方：
- `xhs_rag.cli alerts list|ack`（CLI）
- Web `GET /api/alerts` + 页面顶部横幅
- `schedule status` 打印最近告警
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

ALERTS_FILE = "alerts.json"
MAX_KEEP = 200          # 落盘保留条数
DEDUP_WINDOW = 3600     # 相同告警去重窗口（秒）


def _path(cfg) -> Path:
    return cfg.path("paths.data_dir") / ALERTS_FILE


def _load(cfg) -> list[dict]:
    try:
        data = json.loads(_path(cfg).read_text(encoding="utf-8"))
        return data.get("alerts", []) if isinstance(data, dict) else []
    except Exception:
        return []


def _save(cfg, alerts: list[dict]) -> None:
    try:
        p = _path(cfg)
        p.parent.mkdir(parents=True, exist_ok=True)
        trimmed = alerts[-MAX_KEEP:]
        p.write_text(json.dumps({"alerts": trimmed}, ensure_ascii=False, indent=2),
                     encoding="utf-8")
    except Exception:
        pass


def _is_dup(alerts: list[dict], kind: str, message: str) -> bool:
    """窗口期内相同 kind+message 视为重复。"""
    try:
        cutoff = datetime.now() - timedelta(seconds=DEDUP_WINDOW)
        for a in reversed(alerts[-20:]):
            if a.get("kind") != kind or a.get("message") != message:
                continue
            ts = datetime.fromisoformat(a.get("ts", "1970-01-01T00:00:00"))
            if ts > cutoff:
                return True
    except Exception:
        pass
    return False


def desktop_notify(title: str, message: str) -> bool:
    """Windows 10+ 桌面 toast（WinRT，经 PowerShell，无需第三方模块）。失败静默返回 False。"""
    if sys.platform != "win32":
        return False
    ps = (
        "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType=WindowsRuntime] > $null;"
        "$t=[Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent("
        "[Windows.UI.Notifications.ToastTemplateType]::ToastText02);"
        "$x=$t.GetElementsByTagName('text');"
        f"$x.Item(0).AppendChild($t.CreateTextNode('{_ps_escape(title)}')) > $null;"
        f"$x.Item(1).AppendChild($t.CreateTextNode('{_ps_escape(message)}')) > $null;"
        "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("
        "'xhs-rag').Show([Windows.UI.Notifications.ToastNotification]::new($t));"
    )
    try:
        subprocess.run(["powershell", "-NoProfile", "-WindowStyle", "Hidden",
                        "-Command", ps],
                       capture_output=True, timeout=15)
        return True
    except Exception:
        return False


def _ps_escape(s: str) -> str:
    return s.replace("'", "''").replace("\r", " ").replace("\n", " ")


def notify(cfg, kind: str, title: str, message: str, level: str = "error",
           desktop: bool | None = None) -> dict:
    """记一条告警。返回写入的记录（重复则返回已存在的那条，且不重复落盘）。

    kind: sync_failed | login_expired | coverage_gap | other
    level: error | warning
    desktop: None 时读配置 notify.desktop（默认 True）
    """
    from .core.logging import logger

    # ① 日志
    log_fn = logger.error if level == "error" else logger.warning
    log_fn(f"[告警/{kind}] {title} —— {message}")

    alerts = _load(cfg)
    if _is_dup(alerts, kind, message):
        logger.debug(f"[告警/{kind}] 窗口期内重复，跳过落盘")
        return next((a for a in reversed(alerts)
                     if a.get("kind") == kind and a.get("message") == message), {})

    rec = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "kind": kind,
        "level": level,
        "title": title,
        "message": message,
        "ack": False,
    }
    alerts.append(rec)
    _save(cfg, alerts)  # ② 落盘

    # ③ 桌面通知
    want_desktop = bool(cfg.get("notify.desktop", True)) if desktop is None else desktop
    if want_desktop:
        rec["desktop_delivered"] = desktop_notify(f"xhs-rag · {title}", message)
        # 落盘更新（把投递结果记上，便于排查"到底弹没弹"）
        alerts[-1] = rec
        _save(cfg, alerts)

    # ④ 邮件（未配置则跳过；本版不做实际投递，只记录意图避免误以为已发）
    to = str(cfg.get("auth.notify_email") or "").strip()
    if to and level == "error":
        logger.warning(f"[告警/{kind}] notify_email={to} 已配置，但邮件通道尚未实现"
                       f"（见 docs/新项目启动交接包.md 技术债表）")
    return rec


def recent(cfg, n: int = 20, only_unacked: bool = False) -> list[dict]:
    alerts = _load(cfg)
    if only_unacked:
        alerts = [a for a in alerts if not a.get("ack")]
    return alerts[-n:][::-1]  # 新的在前


def unacked_count(cfg) -> int:
    return len([a for a in _load(cfg) if not a.get("ack")])


def ack_all(cfg) -> int:
    """全部标记已读，返回标记条数。"""
    alerts = _load(cfg)
    n = 0
    for a in alerts:
        if not a.get("ack"):
            a["ack"] = True
            a["ack_at"] = datetime.now().isoformat(timespec="seconds")
            n += 1
    if n:
        _save(cfg, alerts)
    return n
