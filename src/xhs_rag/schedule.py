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


def build_create_cmd(script: str, at: str) -> list[str]:
    """schtasks /Create 参数列表。

    /TR 指向包装 cmd 脚本（**绝不用 \\" 转义内嵌引号**——2026-09-16 实测：
    schtasks 会把 \\" 原样存进任务定义，触发时报 0x80070002 文件未找到）。
    脚本路径由 ensure_task_script() 生成，位于项目内且不含空格。
    """
    return ["schtasks", "/Create", "/F",
            "/TN", TASK_NAME,
            "/TR", script,
            "/SC", "DAILY",
            "/ST", at]


def build_delete_cmd() -> list[str]:
    return ["schtasks", "/Delete", "/F", "/TN", TASK_NAME]


def build_query_cmd() -> list[str]:
    return ["schtasks", "/Query", "/TN", TASK_NAME, "/FO", "LIST", "/V"]


def build_run_cmd() -> list[str]:
    """让任务计划程序立即触发一次（验证整条调度链路）。"""
    return ["schtasks", "/Run", "/TN", TASK_NAME]


# ── 包装脚本（解决工作目录 + 引号两个坑）─────────────────
SCRIPT_NAME = "sync_task.cmd"


def task_script_path(cfg) -> Path:
    return cfg.root / "scripts" / SCRIPT_NAME


def ensure_task_script(cfg) -> Path:
    """生成/覆盖包装 cmd 脚本。

    计划任务由 Task Scheduler 拉起时工作目录是 C:\\Windows\\System32，
    直接 `pythonw -m xhs_rag.cli` 会 ImportError（找不到 xhs_rag 包）。
    脚本里先 `cd /d` 到项目根，再调解释器；路径全 ASCII 且无空格，
    因此 /TR 无需任何引号，彻底避开 schtasks 的转义坑。
    """
    p = task_script_path(cfg)
    log = f"{cfg.root}\\data\\logs\\task_sync.log"
    body = (
        "@echo off\r\n"
        "rem M7 计划任务入口，由 xhs_rag.schedule 自动生成，勿手工修改\r\n"
        f'echo [%date% %time%] M7 sync 被触发 >> "{log}"\r\n'
        f'cd /d "{cfg.root}"\r\n'
        f'"{sys.executable}" -m xhs_rag.cli sync >> "{log}" 2>&1\r\n'
        f'echo [%date% %time%] M7 sync 结束, rc=%ERRORLEVEL% >> "{log}"\r\n'
    )
    p.parent.mkdir(parents=True, exist_ok=True)
    # 内容为 ASCII；若项目根含中文路径则退回 GBK（cmd 默认代码页）
    try:
        p.write_text(body, encoding="ascii")
    except UnicodeEncodeError:
        p.write_text(body, encoding="gbk", errors="replace")
    return p


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


def install(cfg, at: str = "09:00") -> tuple[int, str]:
    """生成包装脚本并注册计划任务（/F 幂等覆盖）。返回 (rc, 输出)。"""
    script = ensure_task_script(cfg)
    return _run(build_create_cmd(str(script), at))


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
