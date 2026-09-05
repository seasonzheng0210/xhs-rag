"""Agent 层 —— LLM 决策循环（Agentic RAG，M10）。

把 LLM 从固定管道（query → 检索 → 生成）里的"执行工"升格为"决策者"：
模型拿到工具箱后自己规划 步骤——先搜什么、看结果够不够、不够换词再搜、
够了才写最终回答。简单问题委托 ask 一步到位，复杂/对比/清单类问题
自己多步 search + read_note 组装。

工具设计原则（面试点）：最小权限。只暴露**只读**工具（检索/问答/读原文/
统计），全量同步、删索引这类写操作一律不进工具箱——Agent 的决策循环
不可控，工具权限必须可控。

循环控制（防失控三板斧）：
1. max_steps 硬上限（默认 8），超限强制带已有信息作答
2. system prompt 里写清停机条件（"够了就停，不要为用工具而用工具"）
3. 工具调用失败返回错误 JSON 而非抛异常，循环可继续

实现：OpenAI 兼容 function calling（requests 直发，不引入 SDK），
复用 mcp_server 的 ctx 构建（Retriever/Answerer 预热逻辑不重写）。

用法：
    python -m xhs_rag.cli agent "收藏里有哪些适合宝宝的做法？分辅食和护理各推荐一个"
"""
from __future__ import annotations

import json
import re
import sqlite3
import time

import requests
from loguru import logger

AGENT_SYSTEM = """你是「收藏夹 RAG」的智能体，回答用户关于他收藏的小红书笔记的问题。可用工具：
- search(query, k)：语义检索收藏片段，返回标题+文本片段（找"有哪些相关内容"用它）
- ask(query)：单跳问答，一次检索+带引用的成品回答（简单问题用它，一步到位）
- read_note(note_id)：读某篇笔记的全部原文块（检索只给片段，需要完整步骤/细节时用）
- stats()：收藏库统计（笔记数/视频数/chunks）

决策规则：
1. 简单问题直接 ask；对比、清单、多主题汇总类问题自己拆解成多次 search。
2. 每步看结果决定下一步：信息不够就换关键词或换子主题再搜；某篇笔记是关键来源就读原文。
3. 已有足够信息就停止调工具，给最终回答——不要为用工具而用工具。
4. 最终回答用中文，先结论后展开，依据工具返回内容并标注来源笔记标题，不编造。
5. 收藏夹确实没有的内容，如实说明，不要脑补。
6. 调用工具必须走 function calling 工具调用接口，禁止把工具调用写成文字内容。"""

# OpenAI function calling 工具声明（与下方 _TOOL_IMPL 一一对应）
TOOLS_SPEC = [
    {"type": "function", "function": {
        "name": "search",
        "description": "语义检索小红书收藏夹，返回相关笔记片段（标题+正文片段）",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "检索词"},
            "k": {"type": "integer", "description": "返回条数，默认 5"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "ask",
        "description": "基于收藏夹的单跳问答：自动检索并生成带 [n] 引用的回答（简单问题一步到位）",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "问题"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "read_note",
        "description": "读取某篇笔记的全部原文块（note_id 来自 search 结果）",
        "parameters": {"type": "object", "properties": {
            "note_id": {"type": "string", "description": "笔记 ID"}},
            "required": ["note_id"]}}},
    {"type": "function", "function": {
        "name": "stats",
        "description": "收藏库统计：笔记数/图片数/视频数/ASR 字符/chunks",
        "parameters": {"type": "object", "properties": {}}}},
]


class RAGAgent:
    # 弱模型自愈:把 text 里的伪工具调用 search("词", 1) 解析回结构化调用
    # (glm-4-flash 实测:单工具时走 function calling,4 个工具时偶发退化为纯文本)
    _PSEUDO_RE = re.compile(r"^\s*([a-zA-Z_]\w*)\((.*)\)\s*;?\s*$", re.S)
    _POSITIONAL = {"search": ["query", "k"], "ask": ["query"],
                   "read_note": ["note_id"], "stats": []}

    def __init__(self, cfg, verbose: bool = True, max_steps: int = 8):
        self.cfg = cfg
        self.verbose = verbose
        self.max_steps = max_steps
        self._ctx: dict | None = None
        from .qa.answer import Answerer
        self._ans = Answerer(cfg)  # 只借 base_url/model/headers/_payload

    # ── ctx：复用 mcp_server 的构建与预热（主线程调用，无 Windows 死锁问题）──
    def _get_ctx(self) -> dict:
        if self._ctx is None:
            from .mcp_server import _build_ctx
            self._ctx = _build_ctx(self.cfg)
        return self._ctx

    # ── LLM 单次调用（非流式，带 tools）────────────────────────
    def _chat(self, messages: list[dict]) -> dict:
        body = self._ans._payload(messages, stream=False)
        body["tools"] = TOOLS_SPEC
        body["max_tokens"] = 2000
        resp = requests.post(
            self._ans.base_url.rstrip("/") + "/chat/completions",
            json=body, headers=self._ans._headers(),
            timeout=(10, self._ans.timeout))
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]

    # ── 工具实现 ───────────────────────────────────────────────
    def _tool_search(self, query: str, k: int = 5) -> str:
        ctx = self._get_ctx()
        results = ctx["retriever"].search(query.strip(), k=int(k))
        out = [{"note_id": r["note_id"], "title": r.get("title", ""),
                "text": (r.get("text") or "")[:300]} for r in results]
        return json.dumps({"count": len(out), "results": out},
                          ensure_ascii=False)

    def _tool_ask(self, query: str) -> str:
        from .mcp_server import _tool_ask
        return _tool_ask(query)

    def _tool_read_note(self, note_id: str) -> str:
        ctx = self._get_ctx()
        conn = sqlite3.connect(ctx["db_path"])
        conn.row_factory = sqlite3.Row
        try:
            note = conn.execute(
                "SELECT title, url FROM notes WHERE note_id=?",
                (note_id,)).fetchone()
        finally:
            conn.close()
        if not note:
            return json.dumps({"error": f"note_id {note_id} 不存在"},
                              ensure_ascii=False)
        chunks = [r.get("text") or ""
                  for r in ctx["retriever"]._get_table().to_arrow().to_pylist()
                  if r.get("note_id") == note_id and (r.get("text") or "").strip()]
        return json.dumps(
            {"note_id": note_id, "title": note["title"], "url": note["url"],
             "chunks": [c[:800] for c in chunks]}, ensure_ascii=False)

    def _tool_stats(self) -> str:
        from .mcp_server import _tool_stats
        return _tool_stats()

    def _exec_tool(self, name: str, args: dict) -> str:
        try:
            if name == "search":
                return self._tool_search(args.get("query", ""), args.get("k", 5))
            if name == "ask":
                return self._tool_ask(args.get("query", ""))
            if name == "read_note":
                return self._tool_read_note(args.get("note_id", ""))
            if name == "stats":
                return self._tool_stats()
            return json.dumps({"error": f"未知工具 {name}"}, ensure_ascii=False)
        except Exception as e:  # 工具失败不炸循环
            logger.warning(f"工具 {name} 失败: {e}")
            return json.dumps({"error": f"{name} 失败: {e}"}, ensure_ascii=False)

    def _parse_pseudo_call(self, text: str) -> dict | None:
        """把 content 里的伪工具调用文本解析成 {name, args}。不是伪调用返回 None。"""
        m = self._PSEUDO_RE.match(text.strip())
        if not m or m.group(1) not in self._POSITIONAL:
            return None
        fn, argstr = m.group(1), m.group(2).strip()
        if not argstr:
            return {"name": fn, "args": {}}
        try:
            import ast
            vals = ast.literal_eval(f"[{argstr}]")
        except Exception:
            return None
        if not isinstance(vals, list) or not all(
                isinstance(v, (str, int, float, bool)) for v in vals):
            return None
        names = self._POSITIONAL[fn]
        if len(vals) > len(names):
            return None
        return {"name": fn, "args": dict(zip(names, vals))}

    # ── 主循环 ────────────────────────────────────────────────
    def run(self, query: str) -> dict:
        """跑一个决策循环，返回 {answer, steps, tool_calls, secs}。"""
        t0 = time.time()
        messages = [{"role": "system", "content": AGENT_SYSTEM},
                    {"role": "user", "content": query}]
        trace: list[dict] = []
        for step in range(1, self.max_steps + 1):
            msg = self._chat(messages)
            calls = msg.get("tool_calls") or []
            content = (msg.get("content") or "").strip()
            if not calls:
                # 自愈:弱模型偶发把工具调用写成纯文本,解析回来继续循环
                pseudo = self._parse_pseudo_call(content)
                if pseudo is None:  # 真正的最终回答
                    return {"answer": content, "steps": step - 1,
                            "tool_calls": trace,
                            "secs": round(time.time() - t0, 1)}
                if self.verbose:
                    print(f"  [step {step}] (伪工具调用自愈) {content[:80]}")
                calls = [{"id": f"pseudo_{step}", "type": "function",
                          "function": {"name": pseudo["name"],
                                       "arguments": json.dumps(
                                           pseudo["args"], ensure_ascii=False)}}]
                content = ""
            # 执行全部工具调用并回填
            messages.append({"role": "assistant",
                             "content": content,
                             "tool_calls": calls})
            for c in calls:
                fn = c["function"]["name"]
                try:
                    args = json.loads(c["function"].get("arguments") or "{}")
                except Exception:
                    args = {}
                out = self._exec_tool(fn, args)
                trace.append({"step": step, "tool": fn, "args": args,
                              "out_chars": len(out)})
                if self.verbose:
                    args_s = json.dumps(args, ensure_ascii=False)
                    print(f"  [step {step}] {fn}({args_s}) → {len(out)} 字符")
                messages.append({"role": "tool", "tool_call_id": c["id"],
                                 "content": out[:4000]})
        # 超限：强制收尾
        messages.append({"role": "user", "content":
                         "已达工具调用步数上限，请立即基于已获取的信息给出最终回答。"})
        msg = self._chat(messages)
        return {"answer": (msg.get("content") or "").strip(),
                "steps": self.max_steps, "tool_calls": trace,
                "secs": round(time.time() - t0, 1), "hit_limit": True}
