"""M7 调度：Windows 任务计划程序（schtasks）封装。

设计见 docs/M7-调度设计.md（ADR-1~5）：
- 调度载体 = 系统任务计划程序，Python 只负责注册/查询/删除，不做常驻调度。
- 执行体 = 当前解释器的 python.exe + 包装脚本 scripts/sync_task.cmd -m xhs_rag.cli sync
  （2026-09-16 实测改 python.exe 而非 pythonw.exe：pythonw 属 GUI 子系统，启动期
   错误无 stdout/stderr，静默失败且日志无痕；python.exe + 重定向可保留全部 traceback）。
  工作目录由包装脚本 cd /d 到项目根，绕开 Task Scheduler 拉起时 C:\\Windows\\System32
  工作目录导致 ImportError 的坑。
- 运行结果落盘：① 主证据 = sync_runs 表（trigger=scheduled/manual，可回溯 N 天，
  见 store/schema.sql）；② 兼容兜底 = data/sync_state.json（由 cli.cmd_sync 写入）。
- 运行语义（ADR-1 用户拍板，2026-09-16）：交互模式登录会话内运行；锁屏仍运行；
  开启「错过则补跑」（StartWhenAvailable=true）—— 重启后未登录，登录后会补跑一次。
  不做「不管是否登录都运行」（/RU SYSTEM），因 SYSTEM 账户下 Playwright + 用户目录
  browser_profile 可用性未验证，风险未知。
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

TASK_NAME = "XhsRagSync"
STATE_FILE = "sync_state.json"


# ── 纯函数：拼 schtasks 命令（可单测）─────────────────────
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
    """生成/覆盖包装 cmd 脚本（幂等）。

    计划任务由 Task Scheduler 拉起时工作目录是 C:\\Windows\\System32，
    直接 `python -m xhs_rag.cli` 会 ImportError（找不到 xhs_rag 包）。
    脚本里先 `cd /d` 到项目根，再调解释器；路径全 ASCII 且无空格，
    因此 /TR 无需任何引号，彻底避开 schtasks 的转义坑。

    头部约定（与 B2/Q5 对应）：
    - chcp 65001：统一脚本内 echo 与 python stdout 同为 UTF-8，避免中文乱码；
    - 大小截断：task_sync.log 超过 1MB 时用覆盖写清空（不用 del，符合项目约定），
      防止日志无限增长。
    """
    p = task_script_path(cfg)
    log = f"{cfg.root}\\data\\logs\\task_sync.log"
    body = (
        "@echo off\r\n"
        "chcp 65001 >nul\r\n"
        "rem M7 计划任务入口，由 xhs_rag.schedule 自动生成，勿手工修改\r\n"
        # B2: 超过 1MB 用覆盖写清空（不用 del），空文件 %%~zA 返回 0 不触发
        f'for %%A in ("{log}") do if %%~zA GTR 1048576 break > "{log}"\r\n'
        f'echo [%date% %time%] M7 sync 被触发 >> "{log}"\r\n'
        f'cd /d "{cfg.root}"\r\n'
        f'"{sys.executable}" -m xhs_rag.cli sync >> "{log}" 2>&1\r\n'
        f'echo [%date% %time%] M7 sync 结束, rc=%ERRORLEVEL% >> "{log}"\r\n'
    )
    p.parent.mkdir(parents=True, exist_ok=True)
    # 含中文 rem/echo 文案，用 UTF-8 BOM 写入，配合 chcp 65001 由 cmd 正确按 UTF-8 解析
    p.write_text(body, encoding="utf-8-sig")
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


def _interactive_note() -> str:
    """B1：交互模式语义提示（用户 2026-09-16 拍板）。"""
    return ("提示：本任务在你的登录会话内运行（锁屏仍运行，非 SYSTEM 账户）。\n"
            "      已开启「错过则补跑」：到点未登录/关机，登录后会自动补跑一次。")


def _query_task_xml(name: str) -> str | None:
    """导出已注册任务的 XML（schtasks /Query /XML）。返回解码后的 XML 文本或 None。"""
    cmd = ["schtasks", "/Query", "/TN", name, "/XML"]
    try:
        p = subprocess.run(cmd, capture_output=True)
    except OSError:
        return None
    raw = p.stdout or b""
    if not raw:
        return None
    # schtasks /Query /XML 多为 UTF-16LE（带 BOM），逐个尝试解码
    for enc in ("utf-16", "utf-8", "gbk"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("utf-8", errors="replace")


def _patch_start_when_available(xml_text: str) -> str:
    """把 <StartWhenAvailable>false</StartWhenAvailable> 改为 true。

    兼容可能的命名空间前缀（ns0: 等）。若元素不存在，则注入到 <Settings> 内。
    """
    # 1) 直接命中 false（含可选前缀）
    pat = re.compile(
        r"<(?P<ns>[\w]+:)?StartWhenAvailable>\s*false\s*</(?P=ns)StartWhenAvailable>",
        re.IGNORECASE,
    )
    if pat.search(xml_text):
        return pat.sub(
            lambda m: f"<{m.group('ns') or ''}StartWhenAvailable>true</{m.group('ns') or ''}StartWhenAvailable>",
            xml_text,
        )
    # 2) 已是 true，无需处理
    if re.search(r"StartWhenAvailable>\s*true\s*</", xml_text, re.IGNORECASE):
        return xml_text
    # 3) 元素缺失：注入到 <Settings> ... </Settings> 内
    if "<Settings>" in xml_text:
        return xml_text.replace(
            "</Settings>",
            "  <StartWhenAvailable>true</StartWhenAvailable>\n  </Settings>",
            1,
        )
    return xml_text


def _enable_start_when_available(cfg, name: str) -> tuple[int, str]:
    """B1：开启「错过则补跑」。导出 XML → 改 StartWhenAvailable → 重新注册。

    幂等：重复 install 仍得到 true；重新注册保留其余字段（MultipleInstancesPolicy /
    触发时间 /TR 等，因它们已在导出的 XML 内）。
    """
    xml_text = _query_task_xml(name)
    if not xml_text:
        return 1, "[B1] 导出任务 XML 失败，未能开启「错过则补跑」"
    patched = _patch_start_when_available(xml_text)
    if patched == xml_text and 'StartWhenAvailable' not in patched:
        # 理论上不会走到这里，保险提示
        return 1, "[B1] 未能注入 StartWhenAvailable"
    tmp = cfg.path("paths.data_dir") / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    xml_path = tmp / f"{name}.xml"
    # 任务计划程序 /Create /XML 读取器按声明解析：schtasks /Query 导出声明为
    # UTF-16，故此处按 UTF-16LE（带 BOM）回写，与声明一致，避免使用 UTF-8 触发
    # "无法切换编码 / 文档语法" 错误。
    xml_path.write_text(patched, encoding="utf-16")
    rc, out = _run(["schtasks", "/Create", "/F", "/TN", name, "/XML", str(xml_path)])
    if rc != 0:
        return rc, f"[B1] 重新注册失败: {out}"
    return 0, "[B1] 已开启「错过则补跑」(StartWhenAvailable=true)"


def install(cfg, at: str = "09:00") -> tuple[int, str]:
    """生成包装脚本并注册计划任务（/F 幂等覆盖）。返回 (rc, 输出)。

    B1：注册后立即开启「错过则补跑」。
    Q2：schtasks 原始非 0 码归一化为 1，但原始码保留在输出文本里便于排查。
    """
    script = ensure_task_script(cfg)
    rc, out = _run(build_create_cmd(str(script), at))
    if rc != 0:
        # Q2：归一化（保留原始码文本）
        return 1, f"注册失败（schtasks 原始码 {rc}）: {out}"
    # B1：开启错过则补跑
    b_rc, b_out = _enable_start_when_available(cfg, TASK_NAME)
    if b_rc != 0:
        # 任务已注册但不含补跑；原始码归一化，提示补跑未生效
        return 1, (f"{out}\n{b_out}（原始码 {b_rc}）\n"
                   f"{_interactive_note()}")
    return 0, f"{out}\n{b_out}\n{_interactive_note()}"


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
    """cmd_sync 收尾调用（兜底通道，主证据已改为 sync_runs 表）。

    写失败不影响 sync 返回码（best-effort）。
    """
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


def _read_json_state(cfg) -> dict | None:
    try:
        return json.loads(state_path(cfg).read_text(encoding="utf-8"))
    except Exception:
        return None


def _last_sync_run(cfg) -> dict | None:
    """读 sync_runs 最近一行（单一事实来源）。DB 不存在/异常返回 None。"""
    import sqlite3

    try:
        db_path = cfg.path("paths.db")
        if not db_path.exists():
            return None
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM sync_runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception:
        return None


def _iso_from_ms(ms) -> str | None:
    if not ms:
        return None
    try:
        return datetime.fromtimestamp(ms / 1000).isoformat(timespec="seconds")
    except Exception:
        return None


def _run_row_to_state(row: dict, cfg) -> dict:
    """把 sync_runs 行归一化为 read_sync_state 返回结构。

    DB 不存 per-step rc，故「各环节/覆盖」兜底读 sync_state.json（finished 行才补）。
    """
    started = row.get("started_at")
    finished = row.get("finished_at")
    status = row.get("status")
    dur = round((finished - started) / 1000, 1) if (started and finished) else None
    running = (status == "running")

    json_st = _read_json_state(cfg) or {}
    if running:
        steps: dict = {}
        coverage = 0
    else:
        steps = json_st.get("steps", {})
        coverage = json_st.get("coverage", 0)

    return {
        "source": "db",
        "running": running,
        "status": status,
        "started_at": _iso_from_ms(started),
        "finished_at": _iso_from_ms(finished),
        "duration_s": dur,
        "steps": steps,
        "coverage": coverage,
        "listed": row.get("listed") or 0,
        "indexed": row.get("indexed") or 0,
        "error_msg": row.get("error_msg") or "",
    }


def read_sync_state(cfg) -> dict | None:
    """优先读 sync_runs 最近一行；DB 读不到时回退 sync_state.json。

    返回结构含：source/running/status/started_at/finished_at/duration_s/
    steps/coverage/listed/indexed/error_msg，供 schedule status 打印。
    """
    row = _last_sync_run(cfg)
    if row is not None:
        return _run_row_to_state(row, cfg)
    # 兜底：仅 sync_state.json（无 DB 或表空）
    js = _read_json_state(cfg)
    if js is None:
        return None
    js.setdefault("source", "json")
    js.setdefault("running", False)
    js.setdefault("status", "unknown")
    js.setdefault("started_at", None)
    js.setdefault("listed", 0)
    js.setdefault("indexed", 0)
    js.setdefault("error_msg", "")
    return js
