"""M7 调度：Windows 任务计划程序（schtasks）封装。

设计见 docs/M7-调度设计.md（ADR-1~5）：
- 调度载体 = 系统任务计划程序，Python 只负责注册/查询/删除，不做常驻调度。
- 执行体 = 当前解释器的 pythonw.exe -m xhs_rag.cli sync，工作目录 = 项目根。
- 运行结果落 data/sync_state.json（由 cli.cmd_sync 写入），区分「未触发」vs「触发但失败」。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

TASK_NAME = "XhsRagSync"
STATE_FILE = "sync_state.json"


# ── 纯函数：拼 schtasks 命令（可单测）─────────────────────
def _pythonw() -> str:
    """当前解释器对应的 pythonw.exe（无控制台窗口）；找不到则回退 python.exe。"""
    exe = Path(sys.executable)
    pyw = exe.with_name("pythonw.exe")
    return str(pyw if pyw.exists() else exe)


def build_create_cmd(at: str) -> list[str]:
    """schtasks /Create 参数列表。/TR 内嵌引号用 \\" 转义。"""
    tr = f'\\"{_pythonw()}\\" -m xhs_rag.cli sync'
    return ["schtasks", "/Create", "/F",
            "/TN", TASK_NAME,
            "/TR", tr,
            "/SC", "DAILY",
            "/ST", at]


def build_delete_cmd() -> list[str]:
    return ["schtasks", "/Delete", "/F", "/TN", TASK_NAME]


def build_query_cmd() -> list[str]:
    return ["schtasks", "/Query", "/TN", TASK_NAME, "/FO", "LIST", "/V"]


def build_run_cmd() -> list[str]:
    """让任务计划程序立即触发一次（验证整条调度链路）。"""
    return ["schtasks", "/Run", "/TN", TASK_NAME]


# ── 执行封装 ──────────────────────────────────────────────
def _run(cmd: list[str]) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="gbk", errors="replace")
    except OSError as e:
        # 沙箱/权限受限（如 WinError 5）：明确报错而非裸 traceback
        return 1, f"[schtasks 无法执行] {e}"
    out = (p.stdout or "").strip()
    err = (p.stderr or "").strip()
    return p.returncode, out or err


def install(at: str = "09:00") -> tuple[int, str]:
    """注册计划任务（/F 幂等覆盖）。返回 (rc, 输出)。"""
    return _run(build_create_cmd(at))


def uninstall() -> tuple[int, str]:
    return _run(build_delete_cmd())


def exists() -> bool:
    rc, _ = _run(["schtasks", "/Query", "/TN", TASK_NAME])
    return rc == 0


def status_detail() -> tuple[int, str]:
    """任务不存在时 rc=1，输出含错误信息。"""
    return _run(build_query_cmd())


def trigger_once() -> tuple[int, str]:
    """立即触发（异步：schtasks /Run 只负责拉起，结果看 sync_state.json）。"""
    return _run(build_run_cmd())


# ── sync 运行结果持久化 ───────────────────────────────────
def state_path(cfg) -> Path:
    return cfg.path("paths.data_dir") / STATE_FILE


def write_sync_state(cfg, steps: dict[str, int], coverage: int,
                     duration_s: float) -> Path | None:
    """cmd_sync 收尾调用。写失败不影响 sync 返回码（best-effort）。"""
    from datetime import datetime

    try:
        p = state_path(cfg)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "duration_s": round(duration_s, 1),
            "steps": steps,
            "coverage": coverage,
        }
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                     encoding="utf-8")
        return p
    except Exception:
        return None


def read_sync_state(cfg) -> dict | None:
    try:
        return json.loads(state_path(cfg).read_text(encoding="utf-8"))
    except Exception:
        return None
