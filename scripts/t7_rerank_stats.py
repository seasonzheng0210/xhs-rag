# -*- coding: utf-8 -*-
"""读复现台 progress.jsonl 做统计（配 t7_rerank_stress.py）。

用法:
    python scripts/t7_rerank_stats.py [progress.jsonl]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "data/tmp/t7"

p = Path(sys.argv[1] if len(sys.argv) > 1 else T / "cpu_progress.jsonl")
if not p.exists():
    print("(无文件)", p)
    raise SystemExit(0)

lines = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
ev = {}
for x in lines:
    ev.setdefault(x.get("ev"), []).append(x)

meta = (ev.get("start") or [{}])[0]
print("torch =", meta.get("torch"), "| pid =", meta.get("pid"),
      "| threads_unset =", meta.get("threads_unset"))
if "reranker_loaded" in ev:
    r = ev["reranker_loaded"][0]
    print("reranker: target_devices =", r.get("target_devices"),
          "| batch_size =", r.get("batch_size"),
          "| max_length =", r.get("max_length"),
          "| load_s =", r.get("load_s"))
if "warmup_done" in ev:
    print("warmup 后 threads =", ev["warmup_done"][0].get("threads"),
          "| embedding device cfg =", ev["warmup_done"][0].get("embedding_device_cfg"))

oks = ev.get("ok", [])
errs = ev.get("error", [])
errs = [x for x in errs if x.get("reason") != "no_hits"] if errs else []
print(f"\n完成 compute_score 次数 = {len(oks)}  |  Python 级异常 = {len(errs)}"
      f"  |  skip(无命中) = {len(ev.get('skip', []))}")
if "finish" in ev:
    print("finish:", ev["finish"][0])
else:
    print("finish: 未写入 → 进程非正常退出（崩溃或被强杀）")

if oks:
    ts = [o["t"] for o in oks]
    print(f"\n耗时: 首={ts[0]:.2f}s 中位={sorted(ts)[len(ts)//2]:.2f}s "
          f"末={ts[-1]:.2f}s 均={sum(ts)/len(ts):.2f}s")
    print(f"RSS : {oks[0]['rss_mb']:.0f} -> {oks[-1]['rss_mb']:.0f} MB "
          f"(净增 {oks[-1]['rss_mb']-oks[0]['rss_mb']:+.0f} MB)")
    print(f"句柄: {oks[0]['handles']} -> {oks[-1]['handles']} "
          f"(净增 {oks[-1]['handles']-oks[0]['handles']:+d})")
    print(f"threads: {sorted({o.get('threads') for o in oks})}")
    # 每 10 次采样一次资源曲线
    print("\n资源曲线（每 10 次）:")
    for o in oks[::10]:
        print(f"  i={o['i']:5d} cyc={o.get('cyc')} t={o['t']:6.2f}s "
              f"rss={o['rss_mb']:8.1f}MB h={o['handles']:<6d} sum={o.get('sum')}")
if errs:
    print("\nPython 级异常:")
    for e in errs[:10]:
        print("  ", e)
