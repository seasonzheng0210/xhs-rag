"""MCP stdio 冒烟测试：握手 -> tools/list -> tools/call(stats, search, ask)。

运行: python scripts/mcp_smoke.py   （用项目 venv 的 python）
通过标准: 6 步全绿, 末尾 "[6] smoke DONE"。
"""
from __future__ import annotations

import anyio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PY = sys.executable
ROOT = Path(__file__).resolve().parents[1]  # scripts/ 的上一级 = 项目根


async def main() -> None:
    params = StdioServerParameters(
        command=PY,
        args=["-m", "xhs_rag.cli", "mcp"],
        cwd=str(ROOT),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("[1] initialize OK")

            tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            print(f"[2] tools/list -> {names}")

            # stats
            r = await session.call_tool("stats", {})
            txt = r.content[0].text if r.content else "{}"
            print(f"[3] stats -> {txt[:200]}")

            # search
            r = await session.call_tool("search", {"query": "旅行 攻略", "k": 2})
            txt = r.content[0].text if r.content else "{}"
            try:
                d = json.loads(txt)
                print(f"[4] search -> count={d.get('count')} "
                      f"top_note={d.get('results', [{}])[0].get('note_id') if d.get('results') else None} "
                      f"url={d.get('results', [{}])[0].get('url', '')[:50] if d.get('results') else None}")
            except Exception as e:
                print(f"[4] search raw: {txt[:200]}  ({e})")

            # ask (LLM 可用则走问答；不可用退化 search)
            try:
                r = await session.call_tool("ask", {"query": "有哪些旅行攻略"})
                txt = r.content[0].text if r.content else "{}"
                d = json.loads(txt)
                print(f"[5] ask -> model={d.get('model')} answer_len={len(d.get('answer', ''))} "
                      f"answer_preview={d.get('answer', '')[:80]!r}")
            except Exception as e:
                print(f"[5] ask error: {e}")

            print("[6] smoke DONE")


if __name__ == "__main__":
    anyio.run(main)
