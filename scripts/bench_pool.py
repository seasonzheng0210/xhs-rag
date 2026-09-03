"""实验: rerank 候选池大小 6 vs 20 对时延与 top3 的影响(hybrid 开/关)。

跑法: python scripts/bench_pool.py
输出每个 query 在 4 种配置下的耗时与 top3(note_id#seq)。
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
    "蒸蛋 辅食",
    "婴儿 补铁 汤",
]


def make(pool: int, hybrid: bool) -> Retriever:
    r = Retriever(load_config(), DB(load_config().path("paths.db")))
    r.top_k_in = pool
    r.hybrid = hybrid
    return r


cfg = load_config()
db = DB(cfg.path("paths.db"))
# 每种配置独立实例 = 独立模型加载,只测一次预热后的查询耗时
combos = [("pool6+dense", 6, False), ("pool6+hyb", 6, True),
          ("pool20+dense", 20, False), ("pool20+hyb", 20, True)]

# 预热各配置的模型(避免把加载时间算进查询)
for name, pool, hyb in combos:
    r = make(pool, hyb)
    r.warmup()

print(f"{'query':<18}", end="")
for name, _, _ in combos:
    print(f" {name:>14}", end="")
print()
for q in QUERIES:
    row = []
    results_by = {}
    for name, pool, hyb in combos:
        r = make(pool, hyb)
        t0 = time.time()
        res = r.search(q, k=3)
        dt = time.time() - t0
        top = "|".join(f"{x['note_id'][:6]}#{x.get('seq')}" for x in res[:3])
        row.append(f"{dt:4.1f}s {top[:26]}")
        results_by[name] = [x["note_id"] for x in res[:3]]
    print(f"{q:<18} " + "  ".join(f"{x:>36}" for x in row))
    # 一致性: 20+hyb 相对 6+dense 的 top3 是否引入新笔记
    base = set(results_by["pool6+dense"])
    cur = results_by["pool20+hyb"]
    new = [n for n in cur if n not in base]
    if new:
        print(f"{'':<18}   ↑ pool20+hyb top3 新增笔记: "
              + ", ".join(n[:8] for n in new))
