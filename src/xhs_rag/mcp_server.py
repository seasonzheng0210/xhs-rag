"""MCP server —— 把收藏 RAG 的检索 / 问答 / 统计暴露给 MCP 客户端（stdio）。

让 WorkBuddy 等 AI 客户端能直接问用户的收藏库，无需开浏览器。

接入方式（WorkBuddy）—— 在 ~/.workbuddy/mcp.json 的 mcpServers 加一条：
    "xhs-rag": {
        "command": "<装好依赖的 python 绝对路径>",   # 如 venv 的 python.exe
        "args": ["-m", "xhs_rag.cli", "mcp"],
        "cwd": "<克隆本仓库的绝对路径>"
    }
然后在本模块目录用 `python -m xhs_rag.cli mcp` 也能直接以 stdio 跑。

工具：
  - search(query, k=5)  语义检索收藏夹，返回带原帖链接的结果列表
  - ask(query)          检索 + LLM 生成带引用的回答（LLM 不可用时退化为 search）
  - stats()             收藏库统计（笔记/图片/视频/ASR 字符/chunks）

模型在 mcp.run() 前的主线程预热（约 10-20s，与 web serve 同策略），工具调用即时返回。
⚠️ 不要在 MCP 工具线程内懒加载模型 —— Windows 上会死等挂起，预热必须发生在主线程。
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from loguru import logger

from .core.config import Config, load_config

_ctx: dict | None = None
_build_failed: str | None = None


def _build_ctx(cfg: Config) -> dict:
    """组装 retriever + answerer（含模型预热）。任何异常都不该让 server 崩。"""
    from .index.retriever import Retriever
    from .store.db import DB

    db = DB(cfg.path("paths.db"))
    retriever = Retriever(cfg, db)
    t0 = time.time()
    logger.info("预热模型(embedding + rerank)...")
    retriever.warmup()
    logger.info(f"模型预热完成, 耗时 {time.time()-t0:.0f}s")

    answerer = None
    try:
        from .qa.answer import Answerer

        ans = Answerer(cfg)
        ok, why = ans.available()
        if ok:
            logger.info(f"LLM 问答已启用: {ans.provider} / {ans.model}")
        else:
            logger.warning(f"LLM 问答不可用，只提供检索：{why}")
            ans = None
        answerer = ans
    except Exception as e:
        logger.warning(f"LLM 模块加载失败，只提供检索：{e}")

    return {"cfg": cfg, "retriever": retriever, "answerer": answerer,
            "db_path": str(cfg.path("paths.db"))}


def _get_ctx() -> dict:
    global _ctx
    if _ctx is None:
        if _build_failed:
            raise RuntimeError(f"收藏库模型初始化失败: {_build_failed}")
        _ctx = _build_ctx(load_config())
    return _ctx


def _enrich(results: list[dict]) -> list[dict]:
    """补 url / note_type（独立 sqlite 连接，避免跨线程）。"""
    if not results:
        return results
    ctx = _get_ctx()
    try:
        conn = sqlite3.connect(ctx["db_path"])
        conn.row_factory = sqlite3.Row
        try:
            for r in results:
                note = conn.execute(
                    "SELECT url, note_type FROM notes WHERE note_id=?",
                    (r["note_id"],)).fetchone()
                r["url"] = note["url"] if note else ""
                r["note_type"] = note["note_type"] if note else "note"
        finally:
            conn.close()
    except Exception as e:
        logger.warning(f"补 url 失败: {e}")
        for r in results:
            r.setdefault("url", "")
            r.setdefault("note_type", "note")
    return results


def _trim(r: dict, n: int = 400) -> dict:
    """结果瘦身：text 截断，避免 MCP 返回体过大。"""
    out = dict(r)
    text = out.get("text", "")
    out["text"] = text[:n] + ("…" if len(text) > n else "")
    return out


def _tool_search(query: str, k: int = 5) -> str:
    if not query.strip():
        return json.dumps({"error": "query 不能为空"}, ensure_ascii=False)
    ctx = _get_ctx()
    t0 = time.time()
    results = _enrich(ctx["retriever"].search(query.strip(), k=k))
    return json.dumps({
        "query": query.strip(),
        "secs": round(time.time() - t0, 1),
        "count": len(results),
        "results": [_trim(r) for r in results],
    }, ensure_ascii=False, indent=2)


def _tool_ask(query: str, history: list[dict] | None = None) -> str:
    if not query.strip():
        return json.dumps({"error": "query 不能为空"}, ensure_ascii=False)
    ctx = _get_ctx()
    q = query.strip()
    # 多轮: 追问式 query 先结合历史改写成独立检索词(history 由客户端传入)
    history = [h for h in (history or [])
               if isinstance(h, dict) and h.get("role") in ("user", "assistant")]
    search_q = q
    answerer0 = ctx["answerer"]
    if history and answerer0 is not None and answerer0.needs_rewrite(q, history):
        try:
            search_q = answerer0.rewrite_query(q, history)
        except Exception:
            search_q = q
    t0 = time.time()
    results = _enrich(ctx["retriever"].search(search_q))
    out: dict = {"query": q, "search_secs": round(time.time() - t0, 1),
                 "answer": "", "model": "", "results": [_trim(r, 200) for r in results]}
    if search_q != q:
        out["rewritten_query"] = search_q
    if not results:
        out["answer"] = "收藏夹里没有检索到相关内容，换个问法试试。"
        return json.dumps(out, ensure_ascii=False, indent=2)
    answerer = ctx["answerer"]
    if answerer is None:
        out["answer"] = "（LLM 问答未启用）检索到以下内容，请自行查阅。"
        return json.dumps(out, ensure_ascii=False, indent=2)
    try:
        t1 = time.time()
        out["answer"] = answerer.answer(q, results, history or None)
        out["model"] = answerer.model
        out["llm_secs"] = round(time.time() - t1, 1)
    except Exception as e:
        logger.warning(f"AI 回答失败: {e}")
        out["answer"] = f"（AI 回答失败：{e}）检索到以下内容，请自行查阅。"
    return json.dumps(out, ensure_ascii=False, indent=2)


def _tool_stats() -> str:
    ctx = _get_ctx()
    try:
        import lancedb

        tbl = lancedb.connect(str(Path(ctx["db_path"]).parent
                                  / "lancedb")).open_table(
            ctx["retriever"].table_name)
        chunks = tbl.count_rows()
    except Exception:
        chunks = -1
    conn = sqlite3.connect(ctx["db_path"])
    conn.row_factory = sqlite3.Row
    try:
        notes = conn.execute("SELECT COUNT(*) c FROM notes").fetchone()["c"]
        images = conn.execute(
            "SELECT COUNT(*) c FROM images WHERE ocr_done=1").fetchone()["c"]
        vids = conn.execute("SELECT COUNT(*) c FROM videos").fetchone()["c"]
        asr_chars = conn.execute(
            "SELECT COALESCE(SUM(LENGTH(asr_text)),0) c FROM videos").fetchone()["c"]
    finally:
        conn.close()
    return json.dumps({"notes": notes, "images": images, "videos": vids,
                       "asr_chars": asr_chars, "chunks": chunks},
                      ensure_ascii=False, indent=2)


def main() -> int:
    """stdio MCP server 入口。"""
    # MCP stdio 要求 stdout 只承载 JSON-RPC 协议消息。
    # CLI 的 setup_logging 会把 loguru 打到 stdout（带 ANSI 颜色），
    # 任何一行日志混入都会让客户端解析失败，这里强制全部改道 stderr。
    import sys

    from loguru import logger as _logger

    _logger.remove()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    _logger.add(sys.stderr, level="INFO", colorize=False,
                format="{time:HH:mm:ss} | {level: <7} | {message}")

    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:  # 未装 mcp 包时给出可操作提示
        logger.error(f"缺少 mcp 包: {e}\n  请先: pip install \"mcp<2\"（v1 API）")
        return 1

    mcp = FastMCP("xhs-rag", instructions=(
        "用户的小红书收藏夹知识库。search 检索收藏内容(带原帖链接); "
        "ask 基于检索做 LLM 问答(答案带 [n] 引用, 对应 results 下标); "
        "stats 查库统计。回答请优先依据 ask 返回的 results, 不要编造。"))

    @mcp.tool(description="语义检索小红书收藏夹，返回带原帖链接的结果列表")
    def search(query: str, k: int = 5) -> str:
        try:
            return _tool_search(query, k)
        except Exception as e:
            return json.dumps({"error": f"search 失败: {e}"}, ensure_ascii=False)

    @mcp.tool(description="基于收藏夹做 LLM 问答：检索 + 生成带引用的回答。"
                          "多轮对话时传 history=[{role,content},...]（本轮之前的对话），"
                          "追问会被自动改写成独立检索词")
    def ask(query: str, history: list[dict] | None = None) -> str:
        try:
            return _tool_ask(query, history)
        except Exception as e:
            return json.dumps({"error": f"ask 失败: {e}"}, ensure_ascii=False)

    @mcp.tool(description="收藏库统计：笔记数/图片数/视频数/ASR 字符/chunks")
    def stats() -> str:
        try:
            return _tool_stats()
        except Exception as e:
            return json.dumps({"error": f"stats 失败: {e}"}, ensure_ascii=False)

    # 预热必须在 mcp.run() 之前、主线程里做：MCP 工具是在 anyio worker
    # 线程执行的，实测在 worker 线程内首次 import torch / 加载 bge-m3 会
    # 死等挂起（0 CPU、内存停在 ~150MB）；主线程预热仅需 8-12s。
    # 即使预热失败也继续启动 server，让工具返回可读错误而不是裸崩。
    global _ctx, _build_failed
    try:
        _ctx = _build_ctx(load_config())
        _logger.info("模型已预热, MCP server 启动")
    except Exception as e:
        _build_failed = str(e)
        _logger.error(f"预热失败(工具将返回错误): {e}")

    mcp.run()  # stdio transport
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
