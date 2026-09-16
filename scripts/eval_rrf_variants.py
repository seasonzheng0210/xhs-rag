"""P2-2 评测：RRF 融合口径 × 权重 正交对比（不改生产 config）。

动机：retriever._rrf_merge 原实现把 dense+sparse 拼成一个列表后按**全局**
rank 累加 1/(k+rank+1)，等价于给 dense 固定顺位优势（池=20 时 dense 占
rank 0-19、sparse 占 20-39），与 docstring 声称的标准 RRF（两路各自排位求和）
不符。本脚本用同一评测集量化：
    concat              现役行为（全局 rank）
    per_list w=(1,1)    标准 RRF，两路等权
    per_list w=(1,2)    偏 BM25（长尾精确命中）
    per_list w=(2,1)    偏向量（语义/近义表达）
不跑 rerank（生产已关闭），只比融合后的排序质量与时延。

用法：
    python scripts/eval_rrf_variants.py                # 全部变体, 池=20
    python scripts/eval_rrf_variants.py --pool 20 --verbose
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from xhs_rag.core.config import load_config            # noqa: E402
from xhs_rag.index.retriever import Retriever          # noqa: E402
from eval_retrieval import (dedup_by_note, load_eval_set,  # noqa: E402
                            mrr, ndcg_at_k, recall_at_k, retrieve_dense)

VARIANTS = [
    ("concat(现役)", "concat", (1.0, 1.0)),
    ("per_list 1:1", "per_list", (1.0, 1.0)),
    ("per_list 1:1.5", "per_list", (1.0, 1.5)),
    ("per_list 1:2", "per_list", (1.0, 2.0)),
    ("per_list 1.5:1", "per_list", (1.5, 1.0)),
]


def retrieve(ret: Retriever, query: str, pool: int,
             mode: str, weights: tuple[float, float]) -> list[dict]:
    dense = retrieve_dense(ret, query, pool)
    try:
        sparse = ret._bm25_search(query, pool)
    except Exception as e:
        print(f"  [warn] BM25 失败退纯向量: {e}")
        sparse = []
    if sparse:
        return Retriever._rrf_merge(dense, sparse, pool, k=ret.rrf_k,
                                    mode=mode, weights=weights)
    return dense


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", type=Path,
                    default=ROOT / "data/eval/eval_set.jsonl")
    ap.add_argument("--pool", type=int, default=20)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    eval_set = load_eval_set(args.eval_set)
    positives = [e for e in eval_set if e.get("type") != "negative"]
    print(f"评测集: {len(eval_set)} 条 (正例 {len(positives)}), 池={args.pool}, "
          f"k=5, rerank=关闭")

    cfg = load_config()
    t0 = time.time()
    ret = Retriever(cfg)
    ret.embedder.encode(["预热"])
    print(f"embedder 就绪 {time.time() - t0:.0f}s | rrf_k={ret.rrf_k}\n")

    summary: dict[str, dict] = {}
    baseline_top5: dict[str, list[str]] = {}

    for label, mode, weights in VARIANTS:
        recals, mrrs, ndcgs, lat = [], [], [], []
        changed = 0
        for e in eval_set:
            q, rel = e["query"], set(e["relevant"])
            t1 = time.time()
            hits = retrieve(ret, q, args.pool, mode, weights)
            lat.append(time.time() - t1)
            ranked = dedup_by_note(hits)
            if label.startswith("concat"):
                baseline_top5[q] = ranked[:5]
            elif ranked[:5] != baseline_top5.get(q, []):
                changed += 1
            if e.get("type") == "negative" or not rel:
                continue
            recals.append(recall_at_k(ranked, rel, 5))
            mrrs.append(mrr(ranked, rel))
            ndcgs.append(ndcg_at_k(ranked, rel, 5))
        summary[label] = {
            "mode": mode, "weights": list(weights),
            "Recall@5": sum(recals) / len(recals),
            "MRR@10": sum(mrrs) / len(mrrs),
            "nDCG@5": sum(ndcgs) / len(ndcgs),
            "latency": sum(lat) / len(lat),
            "top5_changed_vs_concat": changed if not label.startswith("concat") else 0,
        }

    print("===== 结果 =====")
    print(f"{'配置':<18}{'Recall@5':>10}{'MRR@10':>10}{'nDCG@5':>10}"
          f"{'时延/query':>12}{'top5变化':>10}")
    for label, m in summary.items():
        print(f"{label:<18}{m['Recall@5']:>10.3f}{m['MRR@10']:>10.3f}"
              f"{m['nDCG@5']:>10.3f}{m['latency']:>11.2f}s"
              f"{m['top5_changed_vs_concat']:>10}")
    print()
    for label, m in summary.items():
        print("[RRF_VARIANT_JSON] " + json.dumps(
            {"label": label, "pool": args.pool, "n_pos": len(positives), **m},
            ensure_ascii=False))


if __name__ == "__main__":
    main()
