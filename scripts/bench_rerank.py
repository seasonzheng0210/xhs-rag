# -*- coding: utf-8 -*-
"""基准: 量化 bge-m3 编码 + bge-reranker-v2-m3 推理耗时, 定位 M6 检索瓶颈。

用法: PYTHONPATH=src python scripts/bench_rerank.py [cand_len] [top_k_in]
"""
import sys
import time

sys.path.insert(0, "src")

from xhs_rag.core.config import load_config
from xhs_rag.index.embedding import LocalEmbedder
from xhs_rag.index.retriever import Retriever

cand_len = int(sys.argv[1]) if len(sys.argv) > 1 else 500
top_k_in = int(sys.argv[2]) if len(sys.argv) > 2 else 20

cfg = load_config()

# 1) 测向量检索(不含 rerank)
r = Retriever(cfg)
t0 = time.time()
qvec = r.embedder.encode(["月子喂养应该注意什么"])[0]
t1 = time.time()
hits = r._get_table().search(qvec).limit(top_k_in).to_list()
t2 = time.time()
print(f"[1] embed 编码 query: {t1-t0:.1f}s")
print(f"[2] lancedb 向量检索 top{top_k_in}: {t2-t1:.2f}s")
print(f"    命中 {len(hits)} 条, 文本长度分布: "
      f"{min(len(h['text']) for h in hits)}~{max(len(h['text']) for h in hits)} 字符")

# 2) 测 rerank 推理(截断到 cand_len)
candidates = [h["text"][:cand_len] for h in hits]
model = r._ensure_reranker()
t3 = time.time()
pairs = [["月子喂养应该注意什么", c] for c in candidates]
raw = model.compute_score(pairs)
t4 = time.time()
print(f"[3] rerank {len(candidates)} 对, 截断 {cand_len} 字符: {t4-t3:.1f}s")
print(f"    平均每对 {((t4-t3)/len(candidates))*1000:.0f} ms, 分数范围 {min(raw):.3f}~{max(raw):.3f}")
print(f"\n总计(不含加载): {t4-t0:.1f}s")
