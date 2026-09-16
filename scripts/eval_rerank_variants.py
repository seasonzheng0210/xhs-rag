"""rerank 变体评测：模型 × 打分窗口 的正交对比（P0-1）。

背景：2026-09-13 只做了「判负增益 → 关闭」，未做原定做法（换 bge-reranker-v2-m3 +
打分窗口 160→400）。本脚本用**同一套评测集**对以下组合出对比表，用数据决定 rerank 最终去留：

    hybrid（无 rerank，当前生产）  ← 对照基准
    + bge-reranker-base  @160     ← 复现 09-13 旧配置
    + bge-reranker-base  @400
    + bge-reranker-v2-m3 @160
    + bge-reranker-v2-m3 @400     ← P0 原定做法

用法（每次跑一个组合，避免两模型同时常驻内存）：
    python scripts/eval_rerank_variants.py --rerank-model bge-reranker-v2-m3 --window 400
    python scripts/eval_rerank_variants.py --rerank-model base            --window 160
    python scripts/eval_rerank_variants.py --norank                        # 对照基准

输出：单行 JSON（便于脚本汇总）+ 人类可读汇总表。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_retrieval import (  # noqa: E402
    dedup_by_note, load_eval_set, mrr, ndcg_at_k, recall_at_k, retrieve_hybrid,
)
from xhs_rag.core.config import load_config  # noqa: E402
from xhs_rag.index.retriever import Retriever  # noqa: E402

MODEL_DIRS = {
    "base": "bge-reranker-base",
    "v2-m3": "bge-reranker-v2-m3",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank-model", choices=sorted(MODEL_DIRS), default="v2-m3")
    ap.add_argument("--window", type=int, default=160, help="打分窗口字符数")
    ap.add_argument("--pool", type=int, default=20)
    ap.add_argument("--norank", action="store_true", help="只跑 hybrid 对照（不加载 reranker）")
    ap.add_argument("--limit", type=int, default=0, help="只跑评测集前 N 条（前缀子集，便于横向对比）")
    ap.add_argument("--partial", type=Path, default=None,
                    help="每处理完一条就写一次部分结果的 JSON 路径（防段错误丢结果）")
    ap.add_argument("--eval-set", type=Path, default=ROOT / "data/eval/eval_set.jsonl")
    args = ap.parse_args()

    cfg = load_config()
    # 关键：构造 Retriever 后强制关闭其内部 rerank，候选由本脚本自己重排，
    # 这样「模型/窗口」两个变量完全可控，且不动生产 config。
    ret = Retriever(cfg)
    ret.rerank_enabled = False
    ret.warmup()

    eval_set = load_eval_set(args.eval_set)
    if args.limit:
        eval_set = eval_set[: args.limit]
    positives = [e for e in eval_set if e["relevant"]]
    negatives = [e for e in eval_set if not e["relevant"]]
    print(f"评测集: {len(eval_set)} 条 (正例 {len(positives)} / 负例 {len(negatives)}), "
          f"池={args.pool}, k=5", file=sys.stderr)

    ranker = None
    label = "hybrid(no-rr)"
    if not args.norank:
        from FlagEmbedding import FlagReranker
        model_dir = cfg.path("paths.data_dir") / "models" / MODEL_DIRS[args.rerank_model]
        if not (model_dir / "pytorch_model.bin").exists() and not (
                model_dir / "model.safetensors").exists():
            print(f"[FAIL] 模型不存在: {model_dir}", file=sys.stderr)
            raise SystemExit(1)
        t_load = time.time()
        ranker = FlagReranker(str(model_dir), use_fp16=False, device="cpu")
        print(f"reranker 加载完成({args.rerank_model}, {time.time()-t_load:.0f}s)",
              file=sys.stderr)
        label = f"{args.rerank_model}@{args.window}"

    r5 = r10 = n5 = 0.0
    t_all = 0.0
    n_pos_done = 0
    neg_top1 = []
    for qi, e in enumerate(eval_set, 1):
        q = e["query"]
        t0 = time.time()
        hits = retrieve_hybrid(ret, q, args.pool)
        if ranker is not None:
            cands = [h["text"][: args.window] for h in hits]
            pairs = [[q, c] for c in cands]
            scores = ranker.compute_score(pairs, normalize=True)
            if not isinstance(scores, list):
                scores = [scores]
            order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
            hits = [hits[i] for i in order]
            top1 = scores[order[0]] if order else None
        else:
            top1 = None
        elapsed = time.time() - t0
        t_all += elapsed

        ranked = dedup_by_note(hits)
        if not e["relevant"]:
            if top1 is None and ranked:
                top1 = hits[0].get("_rrf_score")
            neg_top1.append((q, top1))
        else:
            rel = set(e["relevant"])
            r5 += recall_at_k(ranked, rel, 5)
            r10 += mrr(ranked, rel, 10)
            n5 += ndcg_at_k(ranked, rel, 5)
            n_pos_done += 1

        # 逐条进度 + 部分结果落盘：本机实测 bge-reranker-v2-m3 批量重排会
        # 间歇段错误（SIGSEGV，见 docs/2026-09-16-全量推进总结.md），
        # 靠这个能在崩溃后仍拿到「已完成前缀」的可用指标。
        print(f"  [{qi}/{len(eval_set)}] {elapsed:5.1f}s "
              f"R@5={r5 / n_pos_done if n_pos_done else 0:.3f} "
              f"MRR={r10 / n_pos_done if n_pos_done else 0:.3f} | {q[:28]}",
              file=sys.stderr, flush=True)
        if args.partial:
            args.partial.write_text(json.dumps({
                "label": label, "done": qi, "total": len(eval_set),
                "n_pos_done": n_pos_done,
                "Recall@5": round(r5 / n_pos_done, 3) if n_pos_done else 0.0,
                "MRR@10": round(r10 / n_pos_done, 3) if n_pos_done else 0.0,
                "nDCG@5": round(n5 / n_pos_done, 3) if n_pos_done else 0.0,
            }, ensure_ascii=False), encoding="utf-8")

    n = len(positives) or 1
    out = {
        "label": label,
        "pool": args.pool,
        "window": None if args.norank else args.window,
        "n_pos": len(positives),
        "Recall@5": round(r5 / n, 3),
        "MRR@10": round(r10 / n, 3),
        "nDCG@5": round(n5 / n, 3),
        "latency_s_per_query": round(t_all / max(len(eval_set), 1), 2),
    }
    print("\n===== 结果 =====")
    print(f"{'配置':<22}{'Recall@5':>10}{'MRR@10':>10}{'nDCG@5':>10}{'时延/query':>12}")
    print(f"{out['label']:<22}{out['Recall@5']:>10.3f}{out['MRR@10']:>10.3f}"
          f"{out['nDCG@5']:>10.3f}{out['latency_s_per_query']:>11.2f}s")
    print("\n[RERANK_VARIANT_JSON] " + json.dumps(out, ensure_ascii=False))
    if neg_top1:
        print("\n负例 top1（人工核查，越低越好）:", file=sys.stderr)
        for q, s in neg_top1:
            print(f"  {s if s is None else round(float(s), 4)}  {q[:40]}", file=sys.stderr)


if __name__ == "__main__":
    main()
