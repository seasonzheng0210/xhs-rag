"""配方用量保留 + 步骤合并回归测试：检索 + LLM 问答，断言回答符合要求。

背景（2026-09-01 陆续修的三个坑）:
  1. GLM-4-Flash 生成摘要时会省略具体数字 → SYSTEM_PROMPT 加规则 6/7 强制保留
  2. 配方类笔记的用量在图 OCR chunk 里, rerank 只用前 200 字符打分,
     OCR 噪声开头把 chunk 挤出 top5 → retriever 加同笔记补全(_complete_note_chunks)
  3. 步骤类问题 LLM 会把「碗中放蒜末+小米辣+淋热油」和「加生抽+蚝油搅匀」拆成
     两步, 第 2 步悬空没说加到哪个容器 → prompt 提「同容器合并」硬约束 + 正反例

用法:
  PYTHONPATH=src python scripts/_qa_usage_test.py [case名...]
  不带参数跑全部 case。每次约 15s(检索+LLM), 全跑约 1 分钟。
  case 名: qiukui / shacha / shanyao / qiukui_sauce

注: LLM 有随机性, 单次 FAIL 建议重跑该 case 确认。
"""
from __future__ import annotations

import re
import sys
import time

from xhs_rag.core.config import load_config
from xhs_rag.index.retriever import Retriever
from xhs_rag.qa.answer import Answerer


def _check_sauce_merge():
    """返回 (expects, custom_checker) — 检测「碗中放蒜末+小米辣+加生抽+蚝油」
    都在同一行/同一列表项中（即"合并为一步"），不拆成 5、6 两步。

    自定义断言比纯包含检查更严：'碗中' 和 '2勺生抽' 都在文本里 ≠ 合并，
    还可能拆成两行。需要在同一段/同一列表项内才算合并。
    """
    expects = ["碗中", "蒜末", "小米辣", "2勺生抽", "蚝油"]

    def check(text: str) -> tuple[bool, str]:
        # 切成列表项(以 - / 1. / ① 等开头), 逐项找包含"碗中"且同项含"2勺生抽"
        items = re.split(r"\n\s*[-•·\d]+[\.、)]?", text)
        merged_items = [it for it in items if "碗中" in it and "2勺生抽" in it]
        if merged_items:
            return True, f"合并: {merged_items[0][:80]!r}..."
        # fallback: 找"碗中"后面 80 字符内是否出现"2勺生抽"
        pos = text.find("碗中")
        if pos >= 0 and "2勺生抽" in text[pos:pos + 80]:
            return True, f"合并(滑动窗口): 碗中后 80 字内含 2勺生抽"
        return False, "拆开: '碗中' 与 '2勺生抽' 不在同一列表项/窗口内"

    return expects, check


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
    ("qiukui_sauce", "白灼秋葵的做法和要点",
     _check_sauce_merge(),
     "料汁必须合并为一步: 碗中+蒜末+小米辣+生抽+蚝油 同段出现, 不拆 5/6 两步"),
]


def run_case(name: str, query: str, expects,
             ret: Retriever, ans: Answerer) -> tuple[bool, list[str], str]:
    t0 = time.time()
    results = ret.search(query, k=5)
    search_secs = round(time.time() - t0, 1)
    t1 = time.time()
    text = ans.answer(query, results)
    llm_secs = round(time.time() - t1, 1)

    # expects 可能是 list(纯包含检查) 或 (list, custom_checker)
    missing = []
    if isinstance(expects, tuple):
        words, checker = expects
        missing = [w for w in words if w not in text]
        if not missing:
            ok, why = checker(text)
            if not ok:
                return False, [why], (
                    f"检索 {search_secs}s + LLM {llm_secs}s, "
                    f"共 {len(results)} 条(含补全)\n"
                    f"回答: {text[:200].replace(chr(10), ' | ')}"
                )
    else:
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
