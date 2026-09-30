# -*- coding: utf-8 -*-
"""逐 query 比对 GPU 臂 / CPU 臂的 rerank 分数，验证设备不变性。

若两臂对同一 query 的 (sum, min, max) 完全一致，则 09-16 采到的
v2-m3@400 指标（Recall/MRR/nDCG）与设备无关 —— 换设备只影响延迟。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "data/tmp/t7"
GPU_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else T / "progress.jsonl"
CPU_FILE = Path(sys.argv[2]) if len(sys.argv) > 2 else T / "cpu_progress.jsonl"


def load(p):
    rows = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        x = json.loads(line)
        if x.get("ev") == "ok" and x.get("cyc") == 1:
            rows[x["i"]] = x
    return rows


g = load(GPU_FILE)
c = load(CPU_FILE)
common = sorted(set(g) & set(c))
print(f"GPU 臂第1轮 {len(g)} 条 / CPU 臂第1轮 {len(c)} 条 / 可比对 {len(common)} 条\n")

same = diff = 0
worst = []
for i in common:
    a, b = g[i], c[i]
    ok = (a["sum"] == b["sum"] and a["min"] == b["min"] and a["max"] == b["max"])
    if ok:
        same += 1
    else:
        diff += 1
        worst.append((abs(a["sum"] - b["sum"]), i, a["q"], a["sum"], b["sum"]))
print(f"逐位一致: {same}/{len(common)}   有差异: {diff}")
if worst:
    worst.sort(reverse=True)
    print("\n差异最大的前 5 条:")
    for d, i, q, sa, sb in worst[:5]:
        print(f"  i={i:3d} |Δsum|={d:.6f}  gpu={sa} cpu={sb}  {q}")

if common:
    tg = [g[i]["t"] for i in common]
    tc = [c[i]["t"] for i in common]
    print(f"\n耗时对比（同 {len(common)} 条）: GPU 均 {sum(tg)/len(tg):.2f}s | "
          f"CPU 均 {sum(tc)/len(tc):.2f}s | 倍数 {sum(tc)/sum(tg):.1f}x")
