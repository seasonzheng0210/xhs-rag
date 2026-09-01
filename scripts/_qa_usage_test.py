"""配方用量保留回归测试：检索 + LLM 问答，断言回答里保留关键用量数字。

背景（2026-09-01 修的两个坑）:
  1. GLM-4-Flash 生成摘要时会省略具体数字 → SYSTEM_PROMPT 加规则 6/7 强制保留
  2. 配方类笔记的用量在图 OCR chunk 里, rerank 只用前 200 字符打分,
     OCR 噪声开头把 chunk 挤出 top5 → retriever 加同笔记补全(_complete_note_chunks)

用法:
  PYTHONPATH=src python scripts/_qa_usage_test.py [case名...]
  不带参数跑全部 case。每次约 1 分钟(检索) + 几秒(LLM), 全跑约 5 分钟。
  case 名: qiukui / shacha / shanyao

注: LLM 有随机性, 单次 FAIL 建议重跑该 case 确认。
"""
from __future__ import annotations

import sys
import time

from xhs_rag.core.config import load_config
from xhs_rag.index.retriever import Retriever
from xhs_rag.qa.answer import Answerer

# (名称, 查询, 必须包含的关键字, 说明)
CASES = [
    ("qiukui", "白灼秋葵的灵魂料汁怎么调",
     ["2勺", "1勺"],
     "秋葵正文 chunk 里的用量, 验证 prompt 规则 6/7"),
    ("shacha", "沙茶酱煎小猪扒的腌料用量",
     ["一勺", "沙茶酱", "蚝油"],
     "用量在图 OCR chunk, 验证同笔记补全"),
    ("shanyao", "山药栗子猪肚汤的材料和用量",
     ["50克", "20颗", "1个"],
     "正文材料清单, 验证问对侧面(材料和用量)时全量给全"),
]


def run_case(name: str, query: str, expects: list[str],
             ret: Retriever, ans: Answerer) -> tuple[bool, list[str], str]:
    t0 = time.time()
    results = ret.search(query, k=5)
    search_secs = round(time.time() - t0, 1)
    t1 = time.time()
    text = ans.answer(query, results)
    llm_secs = round(time.time() - t1, 1)
    missing = [e for e in expects if e not in text]
    return not missing, missing, (
        f"检索 {search_secs}s + LLM {llm_secs}s, 共 {len(results)} 条(含补全), "
        f"top1={results[0]['title'][:20] if results else '无'}\n"
        f"回答: {text[:200].replace(chr(10), ' | ')}"
    )


def main() -> int:
    args = sys.argv[1:]
    cases = [c for c in CASES if not args or c[0] in args]
    if not cases:
        print(f"未知 case: {args}, 可用: {[c[0] for c in CASES]}")
        return 2

    cfg = load_config()
    ret = Retriever(cfg)
    ans = Answerer(cfg)
    ok, why = ans.available()
    if not ok:
        print(f"!! LLM 不可用: {why}, 测试无意义, 退出")
        return 2

    failed = 0
    for name, query, expects, note in cases:
        print(f"\n=== {name} | {note} ===")
        passed, missing, detail = run_case(name, query, expects, ret, ans)
        print(detail)
        if passed:
            print(f"✅ PASS 包含: {expects}")
        else:
            failed += 1
            print(f"❌ FAIL 缺失: {missing}")

    print(f"\n汇总: {len(cases) - failed}/{len(cases)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
