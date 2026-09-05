"""生成质量抽查 —— LLM-as-judge 评估 RAG 回答的忠实度/切题度/引用有效性。

链路: 评测集正例 query → 生产检索(hybrid+rerank) → Answerer 生成回答
      → 同款 LLM 当裁判,按三维打分:
        faithfulness  忠实度 1-5: 回答是否只依据检索片段,有无编造
        relevance     切题度 1-5: 回答是否回应了问题本身
        citation      引用有效性 0/1: [n] 角标是否都指向真实存在的编号

裁判与被评是同一个模型(glm-4-flash)有自评偏差,结论用于回归对比
(改动前后回答质量是否退化),不用于跨系统横向比较 —— 报告里如实标注。

用法:
    python scripts/eval_answers.py --n 8          # 抽前 8 条正例
    python scripts/eval_answers.py --verbose      # 逐条明细
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xhs_rag.core.config import load_config  # noqa: E402
from xhs_rag.index.retriever import Retriever  # noqa: E402
from xhs_rag.qa.answer import Answerer  # noqa: E402

JUDGE_PROMPT = """你是 RAG 问答质量评估裁判。给出【问题】【检索片段】【AI 回答】,按三个维度打分:

1. faithfulness(忠实度, 1-5): 回答中的事实是否都能在检索片段中找到依据。
   5=完全有依据; 3=大部分有依据但有个别推断; 1=明显编造。
2. relevance(切题度, 1-5): 回答是否回应了问题。5=正面完整回应; 1=答非所问。
3. citation(引用有效性, 0 或 1): 回答里的 [n] 角标是否都指向片段中真实存在的编号,
   且被引用的片段内容与该处陈述匹配。回答没有任何事实性陈述(纯闲聊)时给 1。

只输出 JSON(不要 markdown 代码块):
{"faithfulness": <1-5>, "relevance": <1-5>, "citation": <0|1>, "issue": "<一句话指出最主要的问题, 没有则写'无'>"}"""


def run_judge(answerer: Answerer, query: str, contexts: str,
              answer: str) -> dict:
    """用同款 LLM 当裁判。失败返回 error 占位(不中断整体)。"""
    user = (f"【问题】{query}\n\n【检索片段】\n{contexts}\n\n"
            f"【AI 回答】\n{answer}")
    try:
        import requests

        msgs = [{"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": user}]
        resp = requests.post(
            answerer.base_url.rstrip("/") + "/chat/completions",
            json=answerer._payload(msgs, stream=False) | {"max_tokens": 200},
            headers=answerer._headers(), timeout=(10, 60))
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        m = re.search(r"\{.*\}", raw, re.S)
        return json.loads(m.group(0) if m else raw)
    except Exception as e:
        return {"faithfulness": 0, "relevance": 0, "citation": 0,
                "issue": f"(judge 失败: {e})"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", type=Path,
                    default=ROOT / "data/eval/eval_set.jsonl")
    ap.add_argument("--n", type=int, default=8, help="抽前 N 条正例")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    items = [json.loads(l) for l in
             args.eval_set.read_text(encoding="utf-8").splitlines() if l.strip()]
    positives = [e for e in items if e.get("type") != "negative"][: args.n]

    cfg = load_config()
    ret = Retriever(cfg)
    ret.warmup()
    answerer = Answerer(cfg)
    ok, why = answerer.available()
    if not ok:
        print(f"LLM 不可用: {why}")
        return

    rows = []
    for e in positives:
        q = e["query"]
        t0 = time.time()
        results = ret.search(q)
        if not results:
            rows.append({"q": q, "err": "检索无结果"})
            continue
        # 组喂给裁判的片段(与 build_messages 相同的编号格式)
        ctx = "\n\n".join(
            f"[{i}] 《{r.get('title', '')}》{(r.get('text') or '')[:300]}"
            for i, r in enumerate(results, 1))
        try:
            answer = answerer.answer(q, results)
        except Exception as ex:
            rows.append({"q": q, "err": f"生成失败: {ex}"})
            continue
        verdict = run_judge(answerer, q, ctx, answer)
        rows.append({"q": q, "secs": round(time.time() - t0, 1),
                     "answer_chars": len(answer), **verdict})
        print(f"[{len(rows)}/{len(positives)}] {q!r} "
              f"F={verdict.get('faithfulness')} R={verdict.get('relevance')} "
              f"C={verdict.get('citation')} ({rows[-1].get('secs')}s)")

    done = [r for r in rows if "err" not in r]
    print(f"\n===== 生成质量抽查({len(done)}/{len(positives)} 条完成) =====")
    if done:
        f = sum(r["faithfulness"] for r in done) / len(done)
        r_ = sum(r["relevance"] for r in done) / len(done)
        c = sum(r["citation"] for r in done) / len(done)
        print(f"faithfulness={f:.2f}/5  relevance={r_:.2f}/5  "
              f"citation={c:.0%}")
        weak = [r for r in done
                if r["faithfulness"] < 4 or r["relevance"] < 4
                or r["citation"] == 0]
        print(f"低分 case: {len(weak)} 条")
    if args.verbose:
        print("\n-- 逐条 --")
        for r in rows:
            print(json.dumps(r, ensure_ascii=False)[:300])


if __name__ == "__main__":
    main()
