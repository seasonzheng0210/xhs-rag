"""Agent 功能测试套件 —— 覆盖手测缺口,可回归。

背景: Agent 层此前只系统性跑过 1 个多跳案例(辅食+护理),单跳/负例/超步数/
stats 等路径没走过。本套件把这些路径固化成可复现用例,断言带容差
(glm-4-flash 行为有方差,不写死精确步数/精确措辞)。

用例设计(每条 = 一个行为断言,不是跑通就行):
  C1 单跳简单:   简单做法题 → 应 1-3 步内收尾,回答含关键信息
  C2 多跳综合:   跨子主题题 → 应多步(>=3),用 search+read_note,回答覆盖两主题
  C3 无结果负例: 库里没有的内容 → 应据实说"没有",不许编
  C4 超步数收尾: 单测直接调 finalize 节点(确定性),不赌模型触发
  C5 stats 路径: 问库规模 → 应调 stats(或回答含规模数字)

C4 用单测而非真跑模型触发,是因为超限分支是否触发取决于模型即时行为,
单测能稳定覆盖这条路由代码。

用法(项目 venv):
    python scripts/test_agent_cases.py                # 全跑(约 1-3 分钟,含模型预热)
    python scripts/test_agent_cases.py --cases c1,c2  # 只跑指定用例
    python scripts/test_agent_cases.py --verbose      # 显示回答摘要
退出码: 全过 0, 有失败 1。失败信息打 trace 摘要,便于定位。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xhs_rag.agent import RAGAgent  # noqa: E402
from xhs_rag.core.config import load_config  # noqa: E402

CASES = [
    {"id": "c1", "name": "单跳简单题",
     "query": "山药牛肉汤怎么做",
     "desc": "期望 1-3 步收尾,回答含关键做法信息",
     "min_steps": 1, "max_steps": 3},
    {"id": "c2", "name": "多跳跨主题",
     "query": "收藏里关于宝宝的内容，辅食和日常护理各挑一个推荐，说明理由",
     "desc": "期望 >=3 步,用 search+read_note,回答覆盖辅食与护理",
     "min_steps": 3, "max_steps": 10},
    {"id": "c3", "name": "无结果负例",
     "query": "老式电话机怎么拆开修理",
     "desc": "库中无此内容,期望据实答'没有'而非编造",
     "min_steps": 1, "max_steps": 4},
    {"id": "c4", "name": "超步数收尾(finalize 单测)",
     "query": "(单测,不走模型)",
     "desc": "构造已调 3 步工具的状态,直接调 finalize,期望给出收尾回答",
     "min_steps": 0, "max_steps": 0},
    {"id": "c5", "name": "stats 库规模",
     "query": "我收藏夹里一共有多少篇笔记",
     "desc": "期望调用 stats 工具(或回答含规模数字)",
     "min_steps": 1, "max_steps": 4},
]

TOOLS_ALLOWED = {"search", "ask", "read_note", "stats"}


def _tools(result: dict) -> set[str]:
    return {t["tool"] for t in result.get("tool_calls", [])}


def _check_c1(result: dict) -> tuple[bool, str]:
    steps = result["steps"]
    if not (1 <= steps <= 3):
        return False, f"步数 {steps} 超出 1-3"
    bad = _tools(result) - TOOLS_ALLOWED
    if bad:
        return False, f"调用了未知工具 {bad}"
    a = result["answer"]
    if len(a) < 30:
        return False, "回答过短"
    import re
    if re.match(r"^\s*(search|ask|read_note|stats)\b", a):
        return False, "回答仍是工具调用文本(自愈失败): " + a[:60]
    if "山药" not in a and "牛肉" not in a:
        return False, "回答未含山药/牛肉等关键信息"
    return True, ""


def _check_c2(result: dict) -> tuple[bool, str]:
    steps = result["steps"]
    tools = _tools(result)
    if steps < 3:
        return False, f"步数 {steps} < 3,多跳未拆开"
    if "search" not in tools or "read_note" not in tools:
        return False, f"未同时使用 search+read_note,实际 {sorted(tools)}"
    a = result["answer"]
    if len(a) < 80:
        return False, "回答过短,疑似未综合"
    if not (("辅食" in a or "汤" in a or "补铁" in a)
            and ("护理" in a or "触觉" in a or "前庭" in a)):
        return False, "回答未覆盖两个主题(辅食+护理)的任一方面"
    return True, ""


def _check_c3(result: dict) -> tuple[bool, str]:
    a = result["answer"]
    # 编造判定: 给了虚构网址,或给出超越回显的"修理操作内容"
    if "http" in a:
        return False, "编造了外部网址: " + a[:120]
    if "电话机" in a and any(k in a for k in
                              ("螺丝", "电路", "拆下", "更换", "步骤如下")):
        return False, "疑似编造了修理方法: " + a[:120]
    # 诚实判定: 明确说收藏夹里没有(措辞多样,含"无法找到")
    if not any(k in a for k in ("没有", "未找到", "没找到", "无法找到",
                                "没有检索到", "找不到", "抱歉")):
        return False, f"未表明'收藏里没有': {a[:80]}"
    return True, ""


def _check_c4(result: dict) -> tuple[bool, str]:
    # 单测: 直接调 finalize 节点,确定性覆盖超限收尾路由
    if result.get("unit_error"):
        return False, f"finalize 单测异常: {result['unit_error']}"
    if not result.get("answer"):
        return False, "finalize 未产出收尾回答"
    if not result.get("hit_limit"):
        return False, "finalize 未置 hit_limit"
    return True, ""


def _check_c5(result: dict) -> tuple[bool, str]:
    import re
    tools = _tools(result)
    a = result["answer"]
    has_digit_note = bool(re.search(r"\d+\s*篇", a)) or ("40" in a or "篇笔记" in a)
    if "stats" not in tools and not has_digit_note:
        return False, f"未调 stats 且回答无规模数字: {a[:80]}"
    return True, ""


_CHECKS = {"c1": _check_c1, "c2": _check_c2, "c3": _check_c3,
           "c4": _check_c4, "c5": _check_c5}


def run_finalize_unit(agent: RAGAgent) -> dict:
    """C4: 确定性触发 finalize 分支(不依赖模型即时行为)。"""
    state = {
        "messages": [{"role": "system", "content": "sys"},
                     {"role": "user", "content": "q"},
                     {"role": "assistant", "content": "",
                      "tool_calls": [{"id": "c4_1", "type": "function",
                                      "function": {"name": "search",
                                                   "arguments": "{}"}}]},
                     {"role": "tool", "tool_call_id": "c4_1",
                      "content": "{\"error\": \"注入故障\"}"}],
        "pending": [], "trace": [{"step": 1, "tool": "search",
                                  "args": {"query": "x"}, "out_chars": 30},
                                 {"step": 2, "tool": "search",
                                  "args": {"query": "y"}, "out_chars": 30},
                                 {"step": 3, "tool": "search",
                                  "args": {"query": "z"}, "out_chars": 30}],
        "steps": 3, "answer": "", "hit_limit": False,
    }
    try:
        out = agent._node_finalize(state)
        return {"answer": out.get("answer", ""),
                "hit_limit": bool(out.get("hit_limit")), "unit_error": ""}
    except Exception as e:
        return {"answer": "", "hit_limit": False, "unit_error": str(e)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=",".join(c["id"] for c in CASES),
                    help="逗号分隔的 case id,默认全跑")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    selected = [c for c in CASES if c["id"] in
                [s.strip() for s in args.cases.split(",")]]
    print(f"Agent 测试套件: {len(selected)} 条 "
          f"({', '.join(c['name'] for c in selected)})")

    cfg = load_config()
    agent = RAGAgent(cfg, verbose=False, max_steps=args.max_steps)
    t0 = time.time()
    results = []
    for c in selected:
        r = {"id": c["id"], "name": c["name"]}
        print(f"\n[{c['id']}] {c['name']}: {c['desc']}")
        if c["id"] == "c4":
            out = run_finalize_unit(agent)
            ok, why = _check_c4(out)
            r["steps"], r["answer"] = out.get("steps", 0), out["answer"]
            r["hit_limit"] = out.get("hit_limit")
        else:
            t1 = time.time()
            try:
                out = agent.run(c["query"])
            except Exception as e:
                ok, why = False, f"run 异常: {e}"
                out = {"answer": "", "steps": -1, "tool_calls": []}
            r["steps"], r["answer"] = out["steps"], out["answer"]
            r["tool_calls"] = out["tool_calls"]
            r["secs"] = round(time.time() - t1, 1)
            step_ok = c["min_steps"] <= out["steps"] <= c["max_steps"]
            ok, why = _CHECKS[c["id"]](out)
            if step_ok is False:
                ok, why = False, f"步数 {out['steps']} 超界 "
                f"[{c['min_steps']},{c['max_steps']}]"
            if ok:
                print(f"  ✓ {out['steps']} 步 / "
                      f"{r.get('secs', '-')}s / 工具 {sorted(_tools(out))}")
            else:
                print(f"  ✗ {why}")
        results.append({**r, "ok": ok})

    print(f"\n===== Agent 测试结果({len(selected)} 条, "
          f"总耗时 {round(time.time()-t0, 1)}s) =====")
    passed = sum(1 for r in results if r["ok"])
    for r in results:
        print(f"  {'✓' if r['ok'] else '✗'} {r['id']} {r['name']} "
              f"({r['steps']} 步)")
    if args.verbose:
        print("\n-- 回答摘要 --")
        for r in results:
            if r.get("answer"):
                print(f"  [{r['id']}] {r['answer'][:150].replace(chr(10), ' ')}")
    print(f"\n{passed}/{len(selected)} 通过")
    return 0 if passed == len(selected) else 1


if __name__ == "__main__":
    raise SystemExit(main())
