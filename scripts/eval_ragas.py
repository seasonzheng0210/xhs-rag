"""RAGAS 风格四指标评估 —— 自实现,零重依赖(不装官方 ragas 库)。

背景: 官方 ragas 0.4.3 需拉 35 个依赖(langchain/langgraph/openai 全家桶),
本项目走轻依赖路线,按 RAGAS 论文口径自实现四个核心指标,复用生产
Retriever(hybrid+rerank) + Answerer + 同款 LLM 当裁判:

    faithfulness        忠实度  = 有依据的原子陈述数 / 回答拆出的总陈述数
    answer_relevancy    答案相关度 = 从回答反向生成 3 个问题,与原 query 的
                        embedding 余弦相似度均值(RAGAS 反向问题生成法)
    context_precision   上下文精确率 = 按检索排名对每个片段判「能否支撑回答
                        该问题」(0/1),计算 Average Precision
                        score = Σ_i (v_i · precision@i) / Σ v_i
    context_recall      上下文召回率 = 把 ground_truth 拆句,逐句判「检索片段
                        能否归因该句」,可归因句数 / 总句数

与前 4 个指标配套,eval_set_ragas.jsonl 每条多了 ground_truth 字段
(人工依据笔记原文标注,放 data/eval/ gitignore)。

已知局限(与 eval_answers.py 相同):
    - 裁判与被评是同一个 LLM,有自评偏差,结论用于回归对比,不用于跨系统横评
    - 反向问题生成只用单温度采样,官方 ragas 还会做 LTR 重排,这里从简

用法(项目 venv):
    python scripts/eval_ragas.py --n 6 --verbose
    python scripts/eval_ragas.py --out data/eval/ragas_results.json
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

# ── 裁判 prompts(每步一次调用,批量判,控制 token 开销) ──────────

CLAIM_SPLIT_PROMPT = """你是 RAG 评估助手。把下面的 AI 回答拆解成独立的原子事实陈述:
- 每条陈述只含一个可验证的事实(含具体数字/用量/步骤动作)
- 保留原文表述,不改写不合并
- 问候语、过渡句(如"下面介绍")不算事实陈述,跳过
只输出 JSON 字符串数组(不要 markdown 代码块):
["陈述1", "陈述2", ...]"""

CLAIM_VERIFY_PROMPT = """你是 RAG 评估助手。给出【检索片段】和一组原子事实陈述,
逐条判断该陈述能否从检索片段中直接找到依据或严格推断出来(0=无依据/编造, 1=有依据)。
片段中没有的信息即使是常识也不能算有依据。
只输出 JSON 数组(不要 markdown 代码块):
[{"i": 1, "supported": 1}, {"i": 2, "supported": 0}, ...]"""

GEN_QUESTIONS_PROMPT = """你是 RAG 评估助手。给定一段 AI 回答,生成 3 个用户可能会问、
且这段回答能够回应的问题。问题要具体、多样,不要照抄回答原句。
只输出 JSON 字符串数组(不要 markdown 代码块):
["问题1", "问题2", "问题3"]"""

CONTEXT_GRADE_PROMPT = """你是 RAG 评估助手。给出【问题】和若干按检索排名排序的编号片段,
逐个判断:该片段是否有助于完整回答这个问题?(0=无关, 1=有用)
只输出 JSON 数组(不要 markdown 代码块):
[{"i": 1, "useful": 1}, {"i": 2, "useful": 0}, ...]"""

RECALL_PROMPT = """你是 RAG 评估助手。给出【标准答案】拆成的若干句子和若干编号检索片段,
逐句判断:该句的内容能否在某个检索片段中找到依据?(0=片段完全没提, 1=能找到)
只输出 JSON 数组(不要 markdown 代码块):
[{"i": 1, "attributable": 1}, {"i": 2, "attributable": 0}, ...]"""


def call_llm(answerer: Answerer, system: str, user: str,
             max_tokens: int = 600) -> str:
    import requests
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": user}]
    resp = requests.post(
        answerer.base_url.rstrip("/") + "/chat/completions",
        json=answerer._payload(msgs, stream=False) | {"max_tokens": max_tokens},
        headers=answerer._headers(), timeout=(10, 90))
    resp.raise_for_status()
    return (resp.json()["choices"][0]["message"]["content"] or "").strip()


def parse_json(text: str):
    """从 LLM 输出里抠 JSON,容忍 markdown 围栏/前后废话/伪 JSON。

    实测 deepseek-v4-flash 偶尔输出 ["k": "v", ...] 这种键值对混进数组的
    伪 JSON(合法 json.loads 必挂),这里加两级降级:
    1. 标准解析失败后,尝试把 ["a": "b", ...] 的键值对抠成值列表
    2. 都失败返回 None(调用方重试一次)
    """
    m = re.search(r"(\[.*\]|\{.*\})", text, re.S)
    if not m:
        return None
    raw = m.group(1)
    try:
        return json.loads(raw)
    except Exception:
        pass
    try:
        import ast
        out = ast.literal_eval(raw)   # 单引号 JSON / Python 字面量
        # 裸 dict 序列 "{...},{...}" 会被 ast 解析成 tuple,归一化成 list
        return list(out) if isinstance(out, tuple) else out
    except Exception:
        pass
    # 伪 JSON 降级: ["key": "value", ...] → 取出全部 value
    pairs = re.findall(
        r'["\']([^"\'\[\]{}]+?)["\']\s*:\s*["\']([^"\'\[\]{}]+?)["\']', raw)
    if pairs:
        return [v for _, v in pairs]
    return None


def ask_json(answerer: Answerer, system: str, user: str,
             max_tokens: int = 600):
    """call_llm + 解析,失败自动重试一次(裁判偶发格式坏输出/HTTP 抖动)。"""
    for attempt in (1, 2):
        try:
            out = parse_json(call_llm(answerer, system, user, max_tokens))
            if out is not None:
                return out
        except Exception:
            if attempt == 2:
                return None
    return None


def split_gt_sentences(gt: str) -> list[str]:
    """标准答案拆句(按中文句读切,过滤空句)。"""
    parts = re.split(r"[。！？；;\n]", gt)
    return [p.strip() for p in parts if len(p.strip()) >= 4]


def cosine(a, b) -> float:
    import numpy as np
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denom) if denom else 0.0


# ── 四个指标 ──────────────────────────────────────────────

def faithfulness(answerer, contexts: str, answer: str) -> tuple[float, dict]:
    claims = ask_json(answerer, CLAIM_SPLIT_PROMPT, answer)
    if not claims or not isinstance(claims, list):
        return 0.0, {"err": "claims 拆解失败"}
    claims = [str(c) for c in claims if str(c).strip()][:20]
    if not claims:
        return 1.0, {"claims": 0}
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, 1))
    user = f"【检索片段】\n{contexts}\n\n【事实陈述】\n{numbered}"
    verdicts = ask_json(answerer, CLAIM_VERIFY_PROMPT, user)
    if not isinstance(verdicts, list) or not verdicts:
        return 0.0, {"err": "claim 校验失败", "claims": len(claims)}
    supported = sum(1 for v in verdicts
                    if isinstance(v, dict) and v.get("supported") == 1)
    n = len(claims)
    return supported / n, {"claims": n, "supported": supported}


def answer_relevancy(answerer, embedder, query: str, answer: str,
                     n_questions: int = 3) -> tuple[float, dict]:
    qs = ask_json(answerer, GEN_QUESTIONS_PROMPT, answer)
    if not isinstance(qs, list) or not qs:
        return 0.0, {"err": "反向问题生成失败"}
    qs = [str(q) for q in qs if str(q).strip()][:n_questions]
    if not qs:
        return 0.0, {"err": "反向问题为空"}
    vecs = embedder.encode([query] + qs)
    sims = [cosine(vecs[0], v) for v in vecs[1:]]
    return sum(sims) / len(sims), {"gen_qs": qs, "sims": [round(s, 3) for s in sims]}


def context_precision(answerer, query: str, results: list[dict]) -> tuple[float, dict]:
    if not results:
        return 0.0, {"err": "无检索结果"}
    numbered = "\n\n".join(
        f"[{i}] 《{r.get('title', '')}》{(r.get('text') or '')[:300]}"
        for i, r in enumerate(results, 1))
    user = f"【问题】{query}\n\n【片段】\n{numbered}"
    verdicts = ask_json(answerer, CONTEXT_GRADE_PROMPT, user)
    if not isinstance(verdicts, list) or not verdicts:
        return 0.0, {"err": "片段打分失败"}
    flags = {int(v.get("i", 0)): (1 if v.get("useful") == 1 else 0)
             for v in verdicts if isinstance(v, dict)}
    hits, weighted = 0, 0.0
    for i in range(1, len(results) + 1):
        v = flags.get(i, 0)
        if v:
            hits += 1
            weighted += hits / i          # precision@i
    total = sum(flags.get(i, 0) for i in range(1, len(results) + 1))
    return (weighted / total) if total else 0.0, {
        "flags": [flags.get(i, 0) for i in range(1, len(results) + 1)]}


def context_recall(answerer, ground_truth: str, results: list[dict]) -> tuple[float, dict]:
    sents = split_gt_sentences(ground_truth)
    if not sents or not results:
        return 0.0, {"err": "无标准答案句或无检索结果"}
    numbered = "\n\n".join(
        f"[{i}] 《{r.get('title', '')}》{(r.get('text') or '')[:300]}"
        for i, r in enumerate(results, 1))
    gt = "\n".join(f"{i}. {s}" for i, s in enumerate(sents, 1))
    user = f"【标准答案句子】\n{gt}\n\n【片段】\n{numbered}"

    def _salvage(text: str):
        """裁判发散/输出被 max_tokens 截断时的兜底:正则逐条抢救判决。"""
        out = []
        for m in re.finditer(
                r'"i"\s*:\s*(\d+)[^{}]*?"attributable"\s*:\s*([01])', text):
            out.append({"i": int(m.group(1)), "attributable": int(m.group(2))})
        return out or None

    verdicts = None
    for attempt in (1, 2):
        try:
            raw = call_llm(answerer, RECALL_PROMPT, user, max_tokens=800)
        except Exception:
            continue
        verdicts = parse_json(raw) or _salvage(raw)
        if verdicts:
            break
    if not isinstance(verdicts, list) or not verdicts:
        return 0.0, {"err": "召回校验失败"}
    flags = {int(v.get("i", 0)): (1 if v.get("attributable") == 1 else 0)
             for v in verdicts if isinstance(v, dict)}
    attributed = sum(flags.get(i, 0) for i in range(1, len(sents) + 1))
    return attributed / len(sents), {
        "sents": len(sents),
        "flags": [flags.get(i, 0) for i in range(1, len(sents) + 1)]}


# ── 主流程 ────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", type=Path,
                    default=ROOT / "data/eval/eval_set_ragas.jsonl")
    ap.add_argument("--n", type=int, default=6, help="抽前 N 条")
    ap.add_argument("--out", type=Path, help="逐条明细存 JSON")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    items = [json.loads(l) for l in
             args.eval_set.read_text(encoding="utf-8").splitlines() if l.strip()]
    positives = [e for e in items if e.get("type") != "negative"][: args.n]
    print(f"RAGAS 评估集: {len(positives)} 条, "
          f"每条 ≈ 5 次 LLM 调用(拆claim/验证/反问/片段打分/召回)")

    cfg = load_config()
    ret = Retriever(cfg)
    ret.embedder.encode(["预热"])
    answerer = Answerer(cfg)
    ok, why = answerer.available()
    if not ok:
        print(f"LLM 不可用: {why}")
        return

    rows = []
    for idx, e in enumerate(positives, 1):
        q, gt = e["query"], e.get("ground_truth", "")
        t0 = time.time()
        try:
            results = ret.search(q)
            answer = answerer.answer(q, results) if results else ""
        except Exception as ex:
            rows.append({"q": q, "err": f"检索/生成失败: {ex}"})
            print(f"[{idx}/{len(positives)}] {q!r} 失败: {ex}")
            continue
        contexts = "\n\n".join(
            f"[{i}] 《{r.get('title', '')}》{(r.get('text') or '')[:300]}"
            for i, r in enumerate(results, 1))
        row: dict = {"q": q, "answer_chars": len(answer)}
        f, f_d = faithfulness(answerer, contexts, answer)
        ar, ar_d = answer_relevancy(answerer, ret.embedder, q, answer)
        cp, cp_d = context_precision(answerer, q, results)
        cr, cr_d = (context_recall(answerer, gt, results) if gt
                    else (0.0, {"err": "缺 ground_truth"}))
        row.update({"faithfulness": round(f, 3),
                    "answer_relevancy": round(ar, 3),
                    "context_precision": round(cp, 3),
                    "context_recall": round(cr, 3),
                    "secs": round(time.time() - t0, 1)})
        if args.verbose:
            row["detail"] = {"faith": f_d, "rel": ar_d,
                             "prec": cp_d, "recall": cr_d}
        rows.append(row)
        print(f"[{idx}/{len(positives)}] {q!r}  F={f:.2f} AR={ar:.2f} "
              f"CP={cp:.2f} CR={cr:.2f} ({row['secs']}s)")

    done = [r for r in rows if "err" not in r]
    print(f"\n===== RAGAS 四指标汇总({len(done)}/{len(positives)} 条完成) =====")
    if done:
        for k in ("faithfulness", "answer_relevancy",
                  "context_precision", "context_recall"):
            vals = [r[k] for r in done if k in r]
            if vals:
                print(f"{k:<18} {sum(vals)/len(vals):.3f}  (min={min(vals):.2f})")
        weak = [r["q"] for r in done
                if r["faithfulness"] < 0.8 or r["answer_relevancy"] < 0.5
                or r["context_recall"] < 0.7]
        print(f"低分 case: {len(weak)} 条 {weak}")
    if args.out and rows:
        args.out.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"明细已存 {args.out}")


if __name__ == "__main__":
    main()
