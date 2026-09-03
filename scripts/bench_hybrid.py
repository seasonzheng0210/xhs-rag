"""候选池级探针: RRF 融合到底往 rerank 池里加了什么。

对比 dense top_k_in 与 hybrid(=dense∪bm25 经 RRF 截断)两个候选池:
  - 池大小、池覆盖的笔记数
  - bm25 单独召回的 chunk 里,有多少 note_id 根本不在 dense 池(纯关键词命中、向量漏检)
跑法: python scripts/bench_hybrid.py --pool
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, "src")

from xhs_rag.core.config import load_config  # noqa: E402
from xhs_rag.index.retriever import Retriever  # noqa: E402
from xhs_rag.store.db import DB  # noqa: E402

QUERIES = [
    "月子里怎么吃",
    "沙茶酱 做法",
    "白灼秋葵",
    "燕麦 早餐",
    "带娃出行 注意事项",
    "酥皮 蛋挞",
    "蒸蛋 辅食",       # 育儿+食谱 交集词
    "高铁 米粉",       # 具体商品词
]

cfg = load_config()
db = DB(cfg.path("paths.db"))
r = Retriever(cfg, db)
r.warmup()  # 一次预热,避免反复加载

print(f"{'query':<18} {'dense池':>7} {'hybrid池':>8} {'bm25独有chunk':>12} {'bm25独有笔记':>12}")
for q in QUERIES:
    qvec = r.embedder.encode([q])[0]
    dense = r._get_table().search(qvec).limit(r.top_k_in).to_list()
    sparse = r._bm25_search(q, r.top_k_in)
    merged = r._rrf_merge(dense, sparse, r.top_k_in)
    dense_keys = {(h["note_id"], h.get("seq")) for h in dense}
    sparse_keys = {(h["note_id"], h.get("seq")) for h in sparse}
    merged_keys = {(h["note_id"], h.get("seq")) for h in merged}
    only_bm25 = sparse_keys - dense_keys
    only_bm25_notes = {k[0] for k in only_bm25}
    dense_notes = {k[0] for k in dense_keys}
    # bm25 独有且 note 完全不在 dense 池 —— 真正的"关键词救回"
    saved_notes = only_bm25_notes - dense_notes
    tag = " ⭐救回笔记" if saved_notes else ""
    print(f"{q:<18} {len(dense_keys):>7} {len(merged_keys):>8} "
          f"{len(only_bm25):>12} {len(saved_notes):>12}{tag}")
    if saved_notes:
        for nid in sorted(saved_notes):
            row = db.get_note(nid)
            print(f"    └ {nid} {row.get('title', '')[:40] if row else ''}")
