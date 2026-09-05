"""检索离线评测 —— Recall@k / MRR@10 / nDCG@5,三模式对比。

目的: 把「检索质量靠人工抽检」变成可复现的硬数据,量化各环节真实增益:
    dense       纯向量(bge-m3 cosine)
    hybrid      向量 + BM25 RRF 融合(生产默认链路,无 rerank)
    hybrid+rr   hybrid + bge-reranker 精排(生产完整链路)

评测口径:
    - 笔记级标注(relevant = note_id 集合),结果按 note_id 去重保序
    - Recall@5 / MRR@10 / nDCG@5(二元相关度)
    - 负例 query(relevant=[])不计入指标,单独列 top1 供人工核查
      (负例的自动拒答评估见 CRAG-lite,不在此脚本)

用法(项目 venv):
    python scripts/eval_retrieval.py                       # 全模式,池=20
    python scripts/eval_retrieval.py --pool 6              # 生产默认候选池对照
    python scripts/eval_retrieval.py --mode hybrid         # 只跑单模式
    python scripts/eval_retrieval.py --verbose             # 输出逐 query 明细

评测集格式(JSONL, 每行一条):
    {"query": "...", "relevant": ["note_id", ...], "type": "keyword|semantic|ambiguous|negative", "note": "标注理由"}
评测集含个人收藏 note_id,放 data/eval/(gitignore),不入库。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xhs_rag.core.config import load_config  # noqa: E402
from xhs_rag.index.retriever import Retriever  # noqa: E402


def load_eval_set(path: Path) -> list[dict]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("//"):
            items.append(json.loads(line))
    return items


def dedup_by_note(rankeds: list[dict]) -> list[str]:
    """候选按 note_id 去重保序(笔记级评测口径)。"""
    seen: set[str] = set()
    out = []
    for h in rankeds:
        nid = h["note_id"]
        if nid not in seen:
            seen.add(nid)
            out.append(nid)
    return out


def recall_at_k(ranked_notes: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked_notes[:k]) & relevant) / len(relevant)


def mrr(ranked_notes: list[str], relevant: set[str], k: int = 10) -> float:
    for i, nid in enumerate(ranked_notes[:k], start=1):
        if nid in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_notes: list[str], relevant: set[str], k: int = 5) -> float:
    """二元增益 nDCG。"""
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, nid in enumerate(ranked_notes[:k])
        if nid in relevant
    )
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def retrieve_dense(ret: Retriever, query: str, pool: int) -> list[dict]:
    qvec = ret.embedder.encode([query])[0]
    return ret._get_table().search(qvec).limit(pool).to_list()


def retrieve_hybrid(ret: Retriever, query: str, pool: int) -> list[dict]:
    dense = retrieve_dense(ret, query, pool)
    try:
        sparse = ret._bm25_search(query, pool)
    except Exception as e:
        print(f"  [warn] BM25 失败退纯向量: {e}")
        sparse = []
    if sparse:
        return Retriever._rrf_merge(dense, sparse, pool)
    return dense


def retrieve_hybrid_rr(ret: Retriever, query: str, pool: int) -> tuple[list[dict], list[float]]:
    hits = retrieve_hybrid(ret, query, pool)
    scores = ret._rerank(query, [h["text"][:160] for h in hits])
    paired = [(h, s if s is not None else -1.0) for h, s in zip(hits, scores)]
    paired.sort(key=lambda x: x[1], reverse=True)
    return [h for h, _ in paired], [s for _, s in paired]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", type=Path,
                    default=ROOT / "data/eval/eval_set.jsonl")
    ap.add_argument("--pool", type=int, default=20,
                    help="候选池大小(生产默认 rerank.top_k_in, 当前 6)")
    ap.add_argument("--mode", choices=["dense", "hybrid", "hybrid+rr", "all"],
                    default="all")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    eval_set = load_eval_set(args.eval_set)
    positives = [e for e in eval_set if e.get("type") != "negative"]
    negatives = [e for e in eval_set if e.get("type") == "negative"]
    print(f"评测集: {len(eval_set)} 条 "
          f"(正例 {len(positives)} / 负例 {len(negatives)}), "
          f"候选池={args.pool}, k=5")

    cfg = load_config()
    t0 = time.time()
    ret = Retriever(cfg)
    ret.embedder.encode(["预热"])  # 触发模型加载
    print(f"embedder 就绪 {time.time()-t0:.0f}s")

    modes = (["dense", "hybrid", "hybrid+rr"] if args.mode == "all"
             else [args.mode])
    summary: dict[str, dict[str, float]] = {}

    for mode in modes:
        recals, mrrs, ndcgs, latencies = [], [], [], []
        detail_lines = []
        for e in eval_set:
            q, rel_set = e["query"], set(e["relevant"])
            t1 = time.time()
            if mode == "dense":
                hits = retrieve_dense(ret, q, args.pool)
                top_scores = None
            elif mode == "hybrid":
                hits = retrieve_hybrid(ret, q, args.pool)
                top_scores = None
            else:
                hits, top_scores = retrieve_hybrid_rr(ret, q, args.pool)
            latencies.append(time.time() - t1)

            ranked_notes = dedup_by_note(hits)
            if e.get("type") == "negative":
                top1 = hits[0] if hits else None
                detail_lines.append(
                    f"  [负例] {q!r} top1: "
                    f"{(top1 or {}).get('title', '(空)')!r}")
                continue

            r5 = recall_at_k(ranked_notes, rel_set, 5)
            m = mrr(ranked_notes, rel_set)
            n5 = ndcg_at_k(ranked_notes, rel_set, 5)
            recals.append(r5)
            mrrs.append(m)
            ndcgs.append(n5)
            hit_mark = "✓" if ranked_notes and ranked_notes[0] in rel_set else (
                "△" if set(ranked_notes[:5]) & rel_set else "✗")
            detail_lines.append(
                f"  [{hit_mark}] R@5={r5:.2f} MRR={m:.2f} nDCG={n5:.2f} "
                f"| {q!r} -> {ranked_notes[:3]}")

        summary[mode] = {
            "Recall@5": sum(recals) / len(recals) if recals else 0.0,
            "MRR@10": sum(mrrs) / len(mrrs) if mrrs else 0.0,
            "nDCG@5": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
            "avg_latency": sum(latencies) / len(latencies),
        }
        if args.verbose:
            print(f"\n== {mode} 逐条明细 ==")
            print("\n".join(detail_lines))

    print(f"\n===== 汇总(池={args.pool}, 正例 {len(positives)} 条) =====")
    header = f"{'mode':<12}{'Recall@5':>10}{'MRR@10':>10}{'nDCG@5':>10}{'时延/query':>12}"
    print(header)
    for mode, m in summary.items():
        print(f"{mode:<12}{m['Recall@5']:>10.3f}{m['MRR@10']:>10.3f}"
              f"{m['nDCG@5']:>10.3f}{m['avg_latency']:>10.2f}s")


if __name__ == "__main__":
    main()
