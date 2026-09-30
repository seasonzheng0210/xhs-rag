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
  - search(query, k=5, image="")  语义检索，返回带原帖链接的结果列表
  - ask(query, image="")          检索 + LLM 单跳生成带引用的回答（LLM 不可用时
                                  退化为 search；自动注入 M11 长期记忆背景注记，
                                  并把本轮问答落盘供离线 digest）
  - agent(query, image="")        M10 多步循环（最多 8 步；更准但慢 3-5 倍）
  - stats()                       收藏库统计（笔记/图片/视频/ASR 字符/chunks）

  image 传图片的本地绝对路径，会先 OCR 图中文字再并入检索词
  （与 CLI 的 `--image`、Web 的传图入口共用同一套 OcrEngine）。

模型在 mcp.run() 前的主线程预热（约 10-20s，与 web serve 同策略），工具调用即时返回。
⚠️ 不要在 MCP 工具线程内懒加载模型 —— Windows 上会死等挂起，预热必须发生在主线程。
   OCR 引擎同理，见 main() 里的 OcrEngine.warmup()。
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


def _apply_image(cfg: Config, image: str, query: str) -> tuple[str, str | None]:
    """image 非空时，把它 OCR 出的文字并入 query。返回 (新 query, 错误信息)。

    语义与 CLI 的 `--image` 保持一致：**合并**而非替换 —— 截图里的正文
    往往比用户手打的关键词更完整，两者一起检索召回更好。
    """
    from .process.ocr import ocr_image_to_query

    info = ocr_image_to_query(cfg, image)
    if not info:
        return query, "图片没有识别出可用文字（或文件不存在/读取失败）"
    q = query.strip()
    return (f"{q} {info['text']}".strip() if q else info["text"])[:300], None


def _tool_search(query: str, k: int = 5, image: str = "") -> str:
    if not query.strip() and not image.strip():
        return json.dumps({"error": "query 与 image 至少要给一个"},
                          ensure_ascii=False)
    ctx = _get_ctx()
    from_image = False
    if image.strip():
        query, err = _apply_image(ctx["cfg"], image.strip(), query)
        if err:
            return json.dumps({"error": err}, ensure_ascii=False)
        from_image = True
    t0 = time.time()
    results = _enrich(ctx["retriever"].search(query.strip(), k=k))
    out = {
        "query": query.strip(),
        "secs": round(time.time() - t0, 1),
        "count": len(results),
        "results": [_trim(r) for r in results],
    }
    if from_image:
        out["from_image"] = True
    return json.dumps(out, ensure_ascii=False, indent=2)


def _tool_ask(query: str, history: list[dict] | None = None,
              record: bool = True, image: str = "") -> str:
    """record=False 时不把本轮对话写进长期记忆。

    image 非空时先 OCR 成文字并并入 query（查询端多模态，同 CLI `--image`）。

    调用方语义：MCP 工具入口（真人问）用默认 True；Agent 的 ask 工具
    按父会话决定 —— 非用户会话（脚本/评测）必须传 False，否则
    test_agent_cases 之类的回归会把测试问答当用户对话写进 dialog_log，
    下一个 digest 就把它摘要成"用户偏好"（2026-09-16 实测踩到过）。
    """
    if not query.strip() and not image.strip():
        return json.dumps({"error": "query 与 image 至少要给一个"},
                          ensure_ascii=False)
    ctx = _get_ctx()
    if image.strip():
        query, err = _apply_image(ctx["cfg"], image.strip(), query)
        if err:
            return json.dumps({"error": err}, ensure_ascii=False)
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
    rewrite_fn = None
    if ctx["answerer"] is not None:
        rewrite_fn = lambda qq: ctx["answerer"].rewrite_query(qq)  # noqa: E731
    # ── M11: 记忆背景注记（摘要+画像）。与 serve /api/answer 同入口。──
    memory_note = ""
    log_fn = None
    try:
        from .memory import log_dialog, recent_context

        log_fn = log_dialog if record else None
        memory_note = recent_context(ctx["db_path"])
    except Exception as e:
        logger.warning(f"记忆背景注记读取失败(忽略): {e}")
    results = _enrich(ctx["retriever"].search(search_q, rewrite_fn=rewrite_fn))
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
        # M11: 对话原文落盘 + 背景注记注入同一处，保证「问了 → 记住」闭环。
        if log_fn:
            log_fn(ctx["db_path"], "user", q, "mcp")
        out["answer"] = answerer.answer(q, results, history or None, memory_note)
        out["model"] = answerer.model
        out["llm_secs"] = round(time.time() - t1, 1)
        if log_fn:
            log_fn(ctx["db_path"], "assistant", out["answer"], "mcp")
    except Exception as e:
        logger.warning(f"AI 回答失败: {e}")
        out["answer"] = f"（AI 回答失败：{e}）检索到以下内容，请自行查阅。"
    return json.dumps(out, ensure_ascii=False, indent=2)


def _tool_agent(query: str, image: str = "") -> str:
    """跑一次 M10 LangGraph Agent 多步循环，返回答案 + 执行轨迹。

    与 ask 的区别：ask 是单跳（检索 → 生成）；agent 会自己决定检索几次、
    要不要精读某篇笔记、要不要补充检索，最多 8 步，步数耗尽有 finalize
    兜底。适合「跨多篇笔记汇总」「对比几篇」「先在库里找再回答」这类问题。

    ⚠️ 代价：每次多 2-4 次 LLM 调用，耗时通常是 ask 的 3-5 倍（30-90s）。
    客户端应把工具超时放宽到 ≥120s。
    """
    if not query.strip() and not image.strip():
        return json.dumps({"error": "query 与 image 至少要给一个"},
                          ensure_ascii=False)
    ctx = _get_ctx()
    if image.strip():
        query, err = _apply_image(ctx["cfg"], image.strip(), query)
        if err:
            return json.dumps({"error": err}, ensure_ascii=False)
    from .agent import RAGAgent

    # session="mcp" → 本轮问答落盘供离线 digest（与 _tool_ask 同口径）；
    # 注意不能留空，否则 Agent 会认为"非用户会话"而不写记忆。
    agent = RAGAgent(ctx["cfg"], verbose=False, session="mcp")
    out = agent.run(query.strip())
    return json.dumps({
        "query": query.strip(),
        "answer": out.get("answer", ""),
        "steps": out.get("steps", 0),
        "hit_limit": bool(out.get("hit_limit")),
        "tools": out.get("tool_calls", []),
        "secs": out.get("secs", 0),
    }, ensure_ascii=False, indent=2)


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
        "ask 基于检索做单跳 LLM 问答(答案带 [n] 引用, 对应 results 下标); "
        "agent 走多步循环，适合跨多篇笔记汇总/对比类问题(更准但慢 3-5 倍); "
        "stats 查库统计。search/ask/agent 都支持 image 参数传图片本地路径"
        "(先 OCR 图中文字再检索)。回答请优先依据返回的 results, 不要编造。"))

    @mcp.tool(description="语义检索小红书收藏夹，返回带原帖链接的结果列表。"
                          "image 可传图片的本地绝对路径，会先 OCR 图中文字"
                          "再与 query 合并检索（query 与 image 至少给一个）")
    def search(query: str = "", k: int = 5, image: str = "") -> str:
        try:
            return _tool_search(query, k, image)
        except Exception as e:
            return json.dumps({"error": f"search 失败: {e}"}, ensure_ascii=False)

    @mcp.tool(description="基于收藏夹做 LLM 问答：检索 + 生成带引用的回答。"
                          "多轮对话时传 history=[{role,content},...]（本轮之前的对话），"
                          "追问会被自动改写成独立检索词。"
                          "image 可传图片本地绝对路径，先 OCR 图中文字再一并提问")
    def ask(query: str = "", history: list[dict] | None = None,
            image: str = "") -> str:
        try:
            return _tool_ask(query, history, True, image)
        except Exception as e:
            return json.dumps({"error": f"ask 失败: {e}"}, ensure_ascii=False)

    @mcp.tool(description="Agent 多步问答（M10）：会自己决定检索几次、精读哪篇笔记，"
                          "最多 8 步。适合「跨多篇笔记汇总/对比」类问题，比 ask 更准"
                          "但慢 3-5 倍（30-90 秒，请把工具超时放宽到 ≥120 秒）。"
                          "image 可传图片本地绝对路径，先 OCR 图中文字再提问")
    def agent(query: str = "", image: str = "") -> str:
        try:
            return _tool_agent(query, image)
        except Exception as e:
            return json.dumps({"error": f"agent 失败: {e}"}, ensure_ascii=False)

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
    # 传图检索用的 OCR 引擎同样必须在主线程预热：MCP 工具跑在 anyio worker
    # 线程里，在那里首次加载 RapidOCR / onnxruntime 会死等挂起（0 CPU）。
    try:
        from .process.ocr import shared_ocr_engine

        _t0 = time.time()
        # 走共享实例：工具里的 ocr_image_to_query 复用同一引擎，
        # 避免每次传图都重新加载一次 RapidOCR（3–15s）。
        _ocr_ok = shared_ocr_engine(load_config()).warmup()
        _logger.info(f"OCR 引擎预热{'完成' if _ocr_ok else '失败(降级为云端 API)'}"
                     f", 耗时 {time.time() - _t0:.0f}s")
    except Exception as e:
        _logger.warning(f"OCR 预热失败(传图检索将不可用): {e}")

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
