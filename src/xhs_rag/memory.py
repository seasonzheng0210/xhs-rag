"""M11 长期记忆 —— 对话摘要 + 用户画像双轨（采集/加工分离）。

架构（详见 docs/M11-长期记忆设计.md）：
- 在线零成本采集：Web /api/answer 每轮把 (session, role, content) 追加 dialog_log，
  纯 sqlite 无 LLM 调用，不拖慢回答流。
- 离线 digest 加工（run_digest）：把未消费轮次按规则分块 →
    ① 摘要轨：每块 LLM 压成 1-2 句 summary → memory_digests
    ② 画像轨：仅 user 消息 → LLM 抽用户稳定画像 → profile_entries（规则去重）
  LLM 失败静默跳过，不影响服务。
- 注入（recent_context）：回答前取最近摘要 + 高频画像拼成「背景注记」，
  仅作个性化参考，不是检索片段，回答事实仍只依据检索结果。

线程模型：server 是 ThreadingHTTPServer(sqlite 连接不可跨线程) → 本模块
所有函数每调用独立开短连接，天然线程安全。个人库规模下开销可忽略。

用法：
    from .qa.answer import Answerer
    from .core.config import load_config
    run_digest(str(cfg.path("paths.db")), Answerer(cfg))   # digest
    txt = recent_context(db_path)                          # 注入文本
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

import requests
from loguru import logger

# ── digest 用的 LLM 提示词 ─────────────────────────────────

_SUMMARY_PROMPT = """你是个人收藏夹问答助手的「记忆整理器」。用户会把他收藏的小红书笔记片段交给助手提问（菜谱/育儿/生活技巧等）。下面是助手与用户的一段问答记录。

任务：把它压缩成 1-2 句「对话摘要」，供未来会话快速回顾。摘要要点：用户在解决什么问题、关心什么、助手给出了什么方向的结论/推荐。不要编造记录里没有的信息，不要逐条复述问答。

输出严格 JSON：{"summary": "..."}。只输出 JSON，不要多余文字。"""

_PROFILE_PROMPT = """你是「用户画像抽取器」。从下面的用户提问里，抽取关于用户本人的稳定事实与偏好。

可抽的类型（示例，不限于此）：
- 身份/家庭：给多大孩子做辅食的家长、独居、帮家里老人查食谱……
- 饮食偏好：口味清淡/偏辣、素食、忌口、常做的菜系（粤菜/家常菜）……
- 关心主题：研究沙茶酱/潮汕风味做法、关注宝宝辅食营养搭配、备孕/月子餐……
- 习惯约束：收藏的都是快手菜、不想买复杂厨具……

规则：
1. 只抽「关于用户本人」的稳定特征；收藏内容本身、助手回答内容一律不抽。
2. 一时性的具体问题（如「冬瓜蒸肉饼怎么做」）→ 归纳成关心主题（如「研究家常蒸菜做法」），不要照抄问题原文。
3. 已有画像里已被覆盖的事实不要重复输出。
4. 拿不准的宁可不抽。

输出格式：每行一条画像事实，不要编号、不要 JSON、不要引号、不要解释。没有可抽的就只输出一行「无」。

用户提问记录：
{quotes}

已有画像（一行一条，供去重参考）：
{existing}"""

_DIGEST_LLM = {
    "max_tokens": 800,
    "temperature": 0.2,
}


def _conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(Path(db_path)))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


# ── 采集（server 每轮问答调用，零 LLM 成本）──────────────────

def log_dialog(db_path: str, role: str, content: str,
               session: str = "") -> None:
    """追加一条对话原文到 dialog_log。失败静默（记忆是增强不是主链路）。"""
    try:
        with _conn(db_path) as c:
            c.execute(
                "INSERT INTO dialog_log (ts, session, role, content) VALUES (?,?,?,?)",
                (int(time.time() * 1000), session or "", role, content))
    except Exception as e:
        logger.warning(f"dialog_log 写入失败: {e}")


# ── digest 加工 ─────────────────────────────────────────────

def _pending_blocks(db_path: str, max_rounds: int = 8,
                    gap_min: float = 10.0,
                    limit: int = 120) -> list[dict]:
    """取未 digest 轮次并按规则分块。

    分块规则（不依赖 LLM，行为可预期）：
      (a) 同 session 且与上一轮间隔 <= gap_min 视为同一对话块
      (b) 块内轮次上限 max_rounds（超出切新块）
    返回 [{ids: [...], session, turns: [{role, content}]}]。
    """
    with _conn(db_path) as c:
        rows = c.execute(
            "SELECT id, ts, session, role, content FROM dialog_log "
            "WHERE digested=0 ORDER BY id LIMIT ?", (limit,)).fetchall()
    blocks: list[dict] = []
    cur: dict | None = None
    for r in rows:
        same_session = cur is not None and r["session"] == cur["session"]
        within_gap = cur is not None and (
            (r["ts"] - cur["last_ts"]) / 1000.0 <= gap_min * 60)
        over_rounds = cur is not None and len(cur["ids"]) >= max_rounds * 2
        if cur is not None and same_session and within_gap and not over_rounds:
            cur["ids"].append(r["id"])
            cur["last_ts"] = r["ts"]
            cur["turns"].append({"role": r["role"], "content": r["content"]})
        else:
            cur = {"ids": [r["id"]], "session": r["session"],
                   "last_ts": r["ts"],
                   "turns": [{"role": r["role"], "content": r["content"]}]}
            blocks.append(cur)
    # 归一化：摘要轨 user+assistant 成对才算一轮
    for b in blocks:
        pairs = sum(1 for t in b["turns"] if t["role"] == "user")
        b["rounds"] = pairs
    return blocks


def mark_digested(db_path: str, ids: list[int]) -> None:
    if not ids:
        return
    with _conn(db_path) as c:
        ph = ",".join("?" * len(ids))
        c.execute(f"UPDATE dialog_log SET digested=1 WHERE id IN ({ph})", ids)


def add_digest(db_path: str, summary: str, span: int) -> None:
    with _conn(db_path) as c:
        c.execute("INSERT INTO memory_digests (summary, span, created_at) "
                  "VALUES (?,?,?)",
                  (summary.strip(), span, int(time.time() * 1000)))


def _norm(s: str) -> str:
    """画像归一化：去首尾空白/引号/句末标点，小写（去重比对用）。"""
    s = re.sub(r"^[\s\"'「」『』]+|[\s\"'「」『』。！？.!?]+$", "", s or "")
    return s.lower()


def upsert_profile(db_path: str, content: str,
                   source: str = "") -> tuple[str, int]:
    """画像入库（规则去重，非 LLM——避免合并带来的级联丢失）。

    去重规则：与 active 画像归一化后比对——
      - 完全相同 → hit_count+1 / last_seen 刷新
      - 新文本是旧文本超集 → 保留新、忽略旧
      - 否则插入新条目
    返回 (new|merged|dup, id)。升级位：画像量 >100 后可换 LLM 语义合并。
    """
    content = (content or "").strip()
    if not content:
        return "dup", 0
    now = int(time.time() * 1000)
    nc = _norm(content)
    with _conn(db_path) as c:
        rows = c.execute(
            "SELECT id, content FROM profile_entries WHERE active=1 "
            "ORDER BY last_seen DESC LIMIT 50").fetchall()
        for r in rows:
            old = _norm(r["content"])
            if nc == old:
                c.execute("UPDATE profile_entries SET hit_count=hit_count+1, "
                          "last_seen=?, source=? WHERE id=?",
                          (now, source or r["source"], r["id"]))
                return "merged", r["id"]
            if nc and old and len(nc) > len(old) and old in nc:
                return "dup", r["id"]  # 更泛的旧画像已覆盖，无需新条
        cur = c.execute(
            "INSERT INTO profile_entries (content, source, hit_count, "
            "first_seen, last_seen) VALUES (?,?,1,?,?)",
            (content, source or "", now, now))
        return "new", cur.lastrowid


def profiles(db_path: str, n: int = 50) -> list[dict]:
    with _conn(db_path) as c:
        rows = c.execute(
            "SELECT * FROM profile_entries WHERE active=1 "
            "ORDER BY hit_count DESC, last_seen DESC LIMIT ?", (n,)).fetchall()
    return [dict(r) for r in rows]


def digests(db_path: str, n: int = 3) -> list[dict]:
    with _conn(db_path) as c:
        rows = c.execute(
            "SELECT * FROM memory_digests ORDER BY created_at DESC "
            "LIMIT ?", (n,)).fetchall()
    return [dict(r) for r in rows]


def counts(db_path: str) -> dict:
    with _conn(db_path) as c:
        pending = c.execute(
            "SELECT COUNT(*) c FROM dialog_log WHERE digested=0").fetchone()["c"]
        nd = c.execute("SELECT COUNT(*) c FROM memory_digests").fetchone()["c"]
        np_ = c.execute(
            "SELECT COUNT(*) c FROM profile_entries WHERE active=1").fetchone()["c"]
    return {"pending_rounds": pending, "digests": nd, "profiles": np_}


def clear(db_path: str) -> dict:
    """清空全部记忆（个人数据可控）。返回删除条数。"""
    with _conn(db_path) as c:
        out = {}
        for t in ("dialog_log", "memory_digests", "profile_entries"):
            out[t] = c.execute(f"DELETE FROM {t}").rowcount
    logger.info(f"记忆已清空: {out}")
    return out


def recent_context(db_path: str, n_digest: int = 2,
                   n_profile: int = 6) -> str:
    """拼「背景注记」注入文本：最近摘要 + 高频画像。

    文本里显式声明这是背景（个性化参考）不是检索片段，防止模型拿
    过往结论当本轮新事实（幻觉护栏）。
    """
    parts: list[str] = []
    ds = digests(db_path, n_digest)
    if ds:
        parts.append("与用户聊过的过往（背景，仅辅助理解上下文）：")
        parts.extend(f"- {d['summary']}" for d in ds)
    ps = profiles(db_path, n_profile)
    if ps:
        parts.append("已知用户画像（背景，仅辅助措辞个性化）：")
        parts.extend(f"- {p['content']}" for p in ps)
    if not parts:
        return ""
    head = ("【背景注记】以下是你与这位用户的历史信息，仅供理解上下文与"
            "个性化表达；它们不是本轮检索片段，回答的事实依据仍只能是"
            "片段中的内容，禁止把背景注记当新事实引用。")
    return head + "\n" + "\n".join(parts)


# ── digest 执行 ─────────────────────────────────────────────

def _llm_chat(answerer, messages: list[dict]) -> str:
    """普通 chat 请求（复用 Answerer 的连接信息，不回源本地模型）。"""
    body = {
        "model": answerer.model,
        "messages": messages,
        "stream": False,
        **{k: v for k, v in _DIGEST_LLM.items()},
    }
    if answerer.provider == "deepseek":
        body["thinking"] = {"type": "disabled"}
    resp = requests.post(
        answerer.base_url.rstrip("/") + "/chat/completions",
        json=body, headers=answerer._headers(),
        timeout=(10, answerer.timeout))
    resp.raise_for_status()
    return (resp.json()["choices"][0]["message"].get("content") or "").strip()


def _extract_json(text: str) -> dict | None:
    """三级降级解析：json.loads → 正则抓 {...} → ast.literal_eval。

    先做双花括号修正({{ → {)：glm-4-flash 实测会照抄提示词里转义后的
    `{{"key"` 导致 json.loads 失败(RAGAS judge 同款弱模型坑)。
    """
    text = (text or "").strip()
    if not text:
        return None
    text = text.replace("{{", "{").replace("}}", "}")
    try:
        v = json.loads(text)
        return v if isinstance(v, dict) else None
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            v = json.loads(m.group(0))
            return v if isinstance(v, dict) else None
        except Exception:
            pass
    try:
        import ast
        v = ast.literal_eval(m.group(0)) if m else None
        return v if isinstance(v, dict) else None
    except Exception:
        return None


def _parse_profile_lines(raw: str) -> list[str]:
    """画像按行解析：过滤编号/引号/「无」等占位与空行。

    glm-4-flash 对 JSON 结构输出不稳，画像轨改用行文本输出(每条
    画像本就是一句话，按行比 JSON 稳一个数量级)。
    """
    out: list[str] = []
    for line in (raw or "").splitlines():
        s = line.strip().lstrip("-•·*#0123456789.、)）:：").strip()
        s = s.strip("\"'「」『』")
        if len(s) < 4:  # 太短不成画像
            continue
        if s in ("无", "暂无", "没有", "无新画像") or "没有新画像" in s:
            continue
        if s not in out:
            out.append(s)
    return out


def _digest_one_block(db_path: str, answerer, block: dict,
                      source_hint: str, verbose: bool) -> dict:
    """对一块对话做 摘要轨 + 画像轨 两趟 LLM。返回本块统计。"""
    out = {"rounds": block["rounds"], "summary": 0, "profiles": 0,
           "skipped": False}

    def _fmt(turns, cap=400) -> str:
        lines = []
        for t in turns:
            who = "用户" if t["role"] == "user" else "助手"
            lines.append(f"{who}: {(t['content'] or '')[:cap]}")
        return "\n".join(lines)

    # ① 摘要轨
    try:
        raw = _llm_chat(answerer, [
            {"role": "system", "content": _SUMMARY_PROMPT},
            {"role": "user", "content": _fmt(block["turns"])},
        ])
        obj = _extract_json(raw)
        summary = (obj or {}).get("summary", "").strip() if obj else ""
        if not summary:  # 模型没包 JSON，整段文本可能就是摘要
            summary = raw.strip().strip("\"'").splitlines()[0].strip()[:200]
        if summary:
            add_digest(db_path, summary, block["rounds"])
            out["summary"] = 1
            if verbose:
                print(f"  [digest] 摘要({block['rounds']}轮): {summary[:60]}…")
    except Exception as e:
        logger.warning(f"摘要生成失败(跳过): {e}")

    # ② 画像轨（只取 user 消息，按行解析，弱模型不稳走 JSON）
    users = [t for t in block["turns"] if t["role"] == "user"]
    if users:
        try:
            existing = "\n".join(f"- {p['content']}" for p in
                                 profiles(db_path, 30)) or "（无）"
            quotes = _fmt(users, cap=300)
            raw = _llm_chat(answerer, [
                {"role": "system", "content": _PROFILE_PROMPT
                 .replace("{quotes}", quotes)
                 .replace("{existing}", existing)},
                {"role": "user", "content": "请抽取。"},
            ])
            cands = _parse_profile_lines(raw)
            if not cands:  # 兜底：模型仍给了 JSON → 解析 profiles 数组
                obj = _extract_json(raw)
                vals = (obj or {}).get("profiles", []) if obj else []
                cands = [c for c in vals
                         if isinstance(c, str) and len(c.strip()) >= 4]
            for cand in cands:
                kind, _pid = upsert_profile(db_path, cand, source_hint)
                out["profiles"] += 1
                if verbose and kind != "dup":
                    print(f"  [digest] 画像[{kind}]: {cand[:50]}")
        except Exception as e:
            logger.warning(f"画像抽取失败(跳过): {e}")
    return out


def run_digest(db_path: str, answerer, max_blocks: int = 6,
               verbose: bool = True) -> dict:
    """离线加工一次：摘要 + 画像。LLM 不可用/全失败时安全返回。

    返回 {processed_blocks, rounds, summaries, profiles, skipped, errors}。
    """
    if answerer is None:
        return {"error": "LLM 未启用，digest 需要 Answerer"}
    try:
        ok, why = answerer.available()
        if not ok:
            return {"error": why}
    except Exception as e:
        return {"error": str(e)}
    blocks = _pending_blocks(db_path)[:max_blocks]
    if not blocks:
        if verbose:
            print("  [digest] 没有待处理的对话")
        return {"processed_blocks": 0, "rounds": 0, "summaries": 0,
                "profiles": 0, "errors": 0}

    total = {"rounds": 0, "summaries": 0, "profiles": 0, "errors": 0}
    for i, b in enumerate(blocks, 1):
        if verbose:
            print(f"  [digest] 块 {i}/{len(blocks)}: "
                  f"{b['rounds']}轮 session={b['session'][:8] or '(无)'}")
        try:
            r = _digest_one_block(db_path, answerer, b,
                                  source_hint=b["session"][:8] or "(无会话)",
                                  verbose=verbose)
            total["rounds"] += r["rounds"]
            total["summaries"] += r["summary"]
            total["profiles"] += r["profiles"]
            mark_digested(db_path, b["ids"])
        except Exception as e:
            logger.warning(f"块 {i} digest 失败: {e}")
            total["errors"] += 1  # 不 mark，下次重试
    return {"processed_blocks": len(blocks), **total}
