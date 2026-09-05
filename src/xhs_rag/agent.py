"""Agent 层 —— LangGraph StateGraph 决策循环（Agentic RAG，M10）。

v2（2026-09-05）：从手写 while 循环迁移到 LangGraph StateGraph。动机：
1. 显式状态机：llm(决策) → tools(执行) → 条件路由(继续/收尾/超限) 画在图上，
   比藏在 while 里可审计
2. 统一的失败重试：LLM 调用与工具执行各有独立重试策略（此前工具只有
   "失败回错误 JSON" 的容错，没有重试；LLM 调用失败会直接炸掉整个 run）
3. 检查点扩展位：StateGraph + checkpointer 可平滑升级中断恢复/人审，
   个人收藏库暂无该场景，未启用（零成本预留）

状态机（节点与路由）：
    START → llm ─┬─(无工具调用)→ END(回答就绪)
                 ├─(有工具调用 且 steps<max)→ tools → llm
                 └─(steps≥max)→ finalize(强制收尾) → END
弱模型自愈保留：content 里的伪工具调用文本（search("词",1)）在 llm 节点
解析回结构化 tool_call。

工具设计原则（面试点）：最小权限。只暴露**只读**工具（检索/问答/读原文/
统计），全量同步、删索引这类写操作一律不进工具箱——Agent 的决策循环
不可控，工具权限必须可控。

实现：LangGraph 只做编排；LLM 仍是 OpenAI 兼容 function calling
（requests 直发），复用 mcp_server 的 ctx 构建（Retriever/Answerer 预热
逻辑不重写）。

用法：
    python -m xhs_rag.cli agent "收藏里有哪些适合宝宝的做法？分辅食和护理各推荐一个"
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from typing import TypedDict

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

# OpenAI function calling 工具声明（与 RAGAgent._exec_tool 分发一一对应）
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


class AgentState(TypedDict):
    """LangGraph 全局状态（节点间唯一通信媒介）。"""
    messages: list[dict]   # OpenAI 格式对话（含 tool 角色）
    pending: list[dict]    # 待执行的结构化 tool_calls
    trace: list[dict]      # 工具调用轨迹（步骤/工具/参数/返回长度）
    steps: int             # 已发生的 LLM 决策轮数
    answer: str            # 最终回答
    hit_limit: bool        # 是否触发步数上限强制收尾


class RAGAgent:
    # 弱模型自愈:把 text 里的伪工具调用 search("词", 1) 解析回结构化调用
    # (glm-4-flash 实测:单工具时走 function calling,4 个工具时偶发退化为纯文本)
    _PSEUDO_RE = re.compile(r"^\s*([a-zA-Z_]\w*)\((.*)\)\s*;?\s*$", re.S)
    _POSITIONAL = {"search": ["query", "k"], "ask": ["query"],
                   "read_note": ["note_id"], "stats": []}

    LLM_RETRIES = 2      # LLM 调用重试次数(退避 1s)
    TOOL_RETRIES = 1     # 工具失败重试次数

    def __init__(self, cfg, verbose: bool = True, max_steps: int = 8):
        self.cfg = cfg
        self.verbose = verbose
        self.max_steps = max_steps
        self._ctx: dict | None = None
        self._graph = None
        from .qa.answer import Answerer
        self._ans = Answerer(cfg)  # 只借 base_url/model/headers/_payload

    # ── ctx：复用 mcp_server 的构建与预热（主线程调用，无 Windows 死锁问题）──
    def _get_ctx(self) -> dict:
        if self._ctx is None:
            from .mcp_server import _build_ctx
            self._ctx = _build_ctx(self.cfg)
        return self._ctx

    # ── LLM 调用（带退避重试——v1 缺陷修复：此前失败会炸整个 run）────────
    def _chat(self, messages: list[dict]) -> dict:
        last_err: Exception | None = None
        for attempt in range(1, self.LLM_RETRIES + 1):
            try:
                body = self._ans._payload(messages, stream=False)
                body["tools"] = TOOLS_SPEC
                body["max_tokens"] = 2000
                resp = requests.post(
                    self._ans.base_url.rstrip("/") + "/chat/completions",
                    json=body, headers=self._ans._headers(),
                    timeout=(10, self._ans.timeout))
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]
            except Exception as e:
                last_err = e
                if self.verbose:
                    print(f"  [llm] 调用失败({attempt}/{self.LLM_RETRIES}): {e}")
                if attempt < self.LLM_RETRIES:
                    time.sleep(attempt)  # 1s 退避
        raise last_err  # 重试耗尽，由 finalize 兜底

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
        """工具执行 + 失败重试；重试耗尽返回错误 JSON（不抛异常，循环可继续）。"""
        dispatch = {
            "search": lambda: self._tool_search(args.get("query", ""),
                                                args.get("k", 5)),
            "ask": lambda: self._tool_ask(args.get("query", "")),
            "read_note": lambda: self._tool_read_note(args.get("note_id", "")),
            "stats": lambda: self._tool_stats(),
        }
        if name not in dispatch:
            return json.dumps({"error": f"未知工具 {name}"}, ensure_ascii=False)
        last_err: Exception | None = None
        for attempt in range(1, self.TOOL_RETRIES + 2):
            try:
                return dispatch[name]()
            except Exception as e:
                last_err = e
                if self.verbose:
                    print(f"  [tool] {name} 失败"
                          f"({attempt}/{self.TOOL_RETRIES + 1}): {e}")
                if attempt <= self.TOOL_RETRIES:
                    time.sleep(0.5)
        logger.warning(f"工具 {name} 重试耗尽: {last_err}")
        return json.dumps(
            {"error": f"{name} 失败(重试{self.TOOL_RETRIES + 1}次): {last_err}"},
            ensure_ascii=False)

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

    # ── LangGraph 节点 ────────────────────────────────────────
    def _node_llm(self, state: AgentState) -> dict:
        """决策节点：调 LLM，产出工具调用（或最终回答）。"""
        msg = self._chat(state["messages"])
        calls = msg.get("tool_calls") or []
        content = (msg.get("content") or "").strip()
        pending: list[dict] = []
        if calls:
            pending = calls
        else:
            # 弱模型自愈：伪工具调用文本 → 结构化 tool_call
            pseudo = self._parse_pseudo_call(content)
            if pseudo is not None:
                if self.verbose:
                    print(f"  [llm] (伪工具调用自愈) {content[:80]}")
                pending = [{"id": f"pseudo_{state['steps'] + 1}",
                            "type": "function",
                            "function": {"name": pseudo["name"],
                                         "arguments": json.dumps(
                                             pseudo["args"],
                                             ensure_ascii=False)}}]
                content = ""
        messages = list(state["messages"])
        if pending:
            messages.append({"role": "assistant", "content": content,
                             "tool_calls": pending})
        else:
            messages.append({"role": "assistant", "content": content})
        return {"messages": messages, "pending": pending,
                "steps": state["steps"] + 1}

    def _route_after_llm(self, state: AgentState) -> str:
        if state["pending"]:
            if state["steps"] < self.max_steps:
                return "tools"
            return "finalize"  # 步数耗尽还想调工具 → 强制收尾
        return "end"

    def _node_tools(self, state: AgentState) -> dict:
        """执行节点：跑完所有待执行工具，结果以 tool 消息回填。"""
        messages = list(state["messages"])
        trace = list(state["trace"])
        for c in state["pending"]:
            fn = c["function"]["name"]
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
            except Exception:
                args = {}
            out = self._exec_tool(fn, args)
            trace.append({"step": state["steps"], "tool": fn,
                          "args": args, "out_chars": len(out)})
            if self.verbose:
                args_s = json.dumps(args, ensure_ascii=False)
                print(f"  [step {state['steps']}] {fn}({args_s}) → "
                      f"{len(out)} 字符")
            messages.append({"role": "tool", "tool_call_id": c["id"],
                             "content": out[:4000]})
        return {"messages": messages, "pending": [], "trace": trace}

    def _node_finalize(self, state: AgentState) -> dict:
        """收尾节点：步数耗尽/LLM 重试失败，强制基于已有信息作答。"""
        messages = list(state["messages"])
        messages.append({"role": "user", "content":
                         "已达工具调用步数上限，请立即基于已获取的信息给出最终回答。"})
        try:
            msg = self._chat(messages)
            answer = (msg.get("content") or "").strip()
        except Exception as e:  # LLM 重试也耗尽：留可用信息而非裸崩
            answer = (f"（Agent 收尾失败: {e}；已执行 "
                      f"{len(state['trace'])} 步工具调用）")
        return {"answer": answer, "hit_limit": True, "pending": []}

    def _graph_build(self):
        if self._graph is None:
            from langgraph.graph import END, START, StateGraph
            g = StateGraph(AgentState)
            g.add_node("llm", self._node_llm)
            g.add_node("tools", self._node_tools)
            g.add_node("finalize", self._node_finalize)
            g.add_edge(START, "llm")
            g.add_conditional_edges(
                "llm", self._route_after_llm,
                {"tools": "tools", "finalize": "finalize", "end": END})
            g.add_edge("tools", "llm")
            g.add_edge("finalize", END)
            self._graph = g.compile()
        return self._graph

    # ── 对外入口（签名与 v1 一致）────────────────────────────────
    def run(self, query: str) -> dict:
        """跑一个决策循环，返回 {answer, steps, tool_calls, secs[, hit_limit]}。"""
        t0 = time.time()
        init: AgentState = {
            "messages": [{"role": "system", "content": AGENT_SYSTEM},
                         {"role": "user", "content": query}],
            "pending": [], "trace": [], "steps": 0,
            "answer": "", "hit_limit": False,
        }
        # 预热 ctx（主线程，避免 Windows worker 线程加载模型死锁）
        self._get_ctx()
        final = self._graph_build().invoke(init)
        answer = final.get("answer") or ""
        if not answer:  # 自然结束但 answer 为空时兜底取最后 assistant 文本
            for m in reversed(final["messages"]):
                if m.get("role") == "assistant" and (m.get("content") or "").strip():
                    answer = m["content"].strip()
                    break
        out = {"answer": answer, "steps": final["steps"],
               "tool_calls": final["trace"],
               "secs": round(time.time() - t0, 1)}
        if final.get("hit_limit"):
            out["hit_limit"] = True
        return out
