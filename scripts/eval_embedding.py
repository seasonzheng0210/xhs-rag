"""Embedding 对比实验 —— bge-m3 vs 轻量替代在同一评测集上的检索质量。

动机(面试常问): "为什么选 bge-m3? 换个小模型行不行?"
用 bge-small-zh-v1.5(~100MB, 512 维, CPU 快 5-10 倍)在同一 26 条评测集上
跑同样的 dense 检索, 拿 Recall@5/MRR/nDCG 数据回答, 不拍脑袋。

注意:
  - bge-small-zh-v1.5 查询侧要加检索指令前缀(bge 系列约定), 文档侧不加
  - bge-m3 的分数字段取自 LanceDB _distance(余弦), 与 small 的归一化点积
    语义一致(都是越大越相关), 指标只看排序不受分值尺度影响
  - 首次运行会从 modelscope 下载 ~100MB 模型到 data/models/

用法:
    python scripts/eval_embedding.py [--pool 20]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xhs_rag.core.config import load_config  # noqa: E402
from xhs_rag.index.retriever import Retriever  # noqa: E402

QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
SMALL_MODEL_ID = "BAAI/bge-small-zh-v1.5"


def ensure_small_model(cfg) -> Path:
    model_dir = cfg.path("paths.data_dir") / "models" / SMALL_MODEL_ID.split("/")[-1]
    if not (model_dir / "model.safetensors").exists() and not (
            model_dir / "pytorch_model.bin").exists():
        import modelscope

        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"首次使用, 下载 {SMALL_MODEL_ID} (~100MB)...")
        modelscope.snapshot_download(SMALL_MODEL_ID, local_dir=str(model_dir))
    return model_dir


def load_corpus(ret: Retriever) -> list[dict]:
    tbl = ret._get_table()
    df = tbl.to_pandas()
    return df[["note_id", "seq", "title", "section", "text"]].to_dict("records")


def dense_search(corpus_vecs: np.ndarray, corpus: list[dict],
                 qvec: np.ndarray, pool: int) -> list[dict]:
    """归一化点积 = 余弦, 全表 flat 扫描(与生产小表路径一致)。"""
    cv = corpus_vecs / (np.linalg.norm(corpus_vecs, axis=1, keepdims=True) + 1e-12)
    qv = qvec / (np.linalg.norm(qvec) + 1e-12)
    sims = cv @ qv
    top = np.argsort(-sims)[:pool]
    return [corpus[i] for i in top]


def eval_mode(rows: list[dict], mode_name: str, encode_query, encode_docs,
              positives: list[dict], pool: int) -> dict:
    t0 = time.time()
    doc_vecs = np.array(encode_docs([r["text"] for r in rows]))
    enc_secs = time.time() - t0
    recals, mrrs, ndcgs = [], [], []

    def metrics(ranked_notes: list[str], relevant: set[str]) -> tuple:
        r5 = (len(set(ranked_notes[:5]) & relevant) / len(relevant)
              if relevant else 0.0)
        m = next((1.0 / i for i, n in enumerate(ranked_notes[:10], 1)
                  if n in relevant), 0.0)
        dcg = sum(1.0 / math.log2(i + 2) for i, n in
                  enumerate(ranked_notes[:5]) if n in relevant)
        ideal = sum(1.0 / math.log2(i + 2)
                    for i in range(min(len(relevant), 5)))
        return r5, m, (dcg / ideal if ideal else 0.0)

    for e in positives:
        relevant = set(e["relevant"])
        qvec = np.array(encode_query([e["query"]])[0])
        hits = dense_search(doc_vecs, rows, qvec, pool)
        seen: set[str] = set()
        ranked = []
        for h in hits:
            if h["note_id"] not in seen:
                seen.add(h["note_id"])
                ranked.append(h["note_id"])
        r5, m, n5 = metrics(ranked, relevant)
        recals.append(r5)
        mrrs.append(m)
        ndcgs.append(n5)
    n = len(positives)
    return {"mode": mode_name,
            "Recall@5": sum(recals) / n, "MRR@10": sum(mrrs) / n,
            "nDCG@5": sum(ndcgs) / n, "enc_secs": enc_secs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", type=Path,
                    default=ROOT / "data/eval/eval_set.jsonl")
    ap.add_argument("--pool", type=int, default=20)
    args = ap.parse_args()

    items = [json.loads(l) for l in
             args.eval_set.read_text(encoding="utf-8").splitlines() if l.strip()]
    positives = [e for e in items if e.get("type") != "negative"]

    cfg = load_config()
    ret = Retriever(cfg)
    ret.embedder.encode(["预热"])
    rows = load_corpus(ret)
    print(f"语料 {len(rows)} chunks, 评测 {len(positives)} 条, 池={args.pool}")

    results = []

    # ① bge-m3(生产): 复用 LocalEmbedder(含自身缓存)
    m3 = ret.embedder
    results.append(eval_mode(
        rows, "bge-m3(1024d,生产)",
        lambda ts: m3.encode(ts), lambda ts: m3.encode(ts),
        positives, args.pool))

    # ② bge-small-zh-v1.5(轻量对照)
    from FlagEmbedding import FlagModel

    small_dir = ensure_small_model(cfg)
    small = FlagModel(str(small_dir), use_fp16=False,
                      query_instruction_for_retrieval=QUERY_INSTRUCTION)
    results.append(eval_mode(
        rows, "bge-small-zh(512d)",
        lambda ts: small.encode_queries(ts), lambda ts: small.encode(ts),
        positives, args.pool))

    print(f"\n===== embedding 对比(池={args.pool}, 正例 {len(positives)}) =====")
    print(f"{'model':<22}{'Recall@5':>10}{'MRR@10':>10}{'nDCG@5':>10}"
          f"{'全语料编码':>12}")
    for r in results:
        print(f"{r['mode']:<22}{r['Recall@5']:>10.3f}{r['MRR@10']:>10.3f}"
              f"{r['nDCG@5']:>10.3f}{r['enc_secs']:>10.1f}s")


if __name__ == "__main__":
    main()
