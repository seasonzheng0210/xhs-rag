# -*- coding: utf-8 -*-
"""T7 · v2-m3 批量重排压力台（忠实复刻 09-16 崩溃配置 + 全量埋点）。

目的：v2-m3 批量重排在 2026-09-16 记录为**间歇性 SIGSEGV**（两次 ~23min 的跑动里崩了一次），
当时只有「崩了 / 没崩」这一个二值信息，没有任何栈或阶段信息。本脚本做三件事：

1. **忠实复刻**：同样的 window=400 / pool=20 / 评测集 50 条；并且**不去动 torch 线程数**
   （复刻当时脚本的真实状态）。⚠️ 注意 FlagReranker 的参数是 **`devices=`（复数）** ——
   传 `device="cpu"` 会被 `**kwargs` 吞掉，实际落到 `cuda:0`（本项目踩过的坑）。
2. **阶段埋点**：把 FlagEmbedding 内部的三个关键环节打上标记 ——
   `prepare_for_model_compat`（逐对构造输入）/ `pad_with_compat`（批次 padding）/
   `model.forward`（真正的 C 层计算）。崩了就能知道死在哪一段。
3. **资源曲线**：每完成一条 query 记一行 JSONL（RSS / 句柄数 / 耗时 / 分数校验和），
   用来判断是「内存/句柄增长型」还是「瞬时竞态型」。

配套：`t7_rerank_stats.py`（读 progress 出统计）、`t7_rerank_compare.py`（比对两臂分数/延迟）。

用法:
    python scripts/t7_rerank_stress.py --window 400 --pool 20 --devices cpu --max-min 45 \\
        --progress data/tmp/t7/cpu_progress.jsonl \\
        --marks    data/tmp/t7/cpu_marks.log \\
        --fault    data/tmp/t7/cpu_faulthandler.log

    # --devices auto 复刻"静默走 GPU"的那条路（对照臂）
    # 内置护栏：--devices 与实际 target_devices 不符时直接 exit(3)
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ── faulthandler 必须在 import torch 之前开，才能抓到解释器/原生崩溃 ──
_fault_file = None


def enable_faulthandler(path: Path):
    global _fault_file
    path.parent.mkdir(parents=True, exist_ok=True)
    _fault_file = open(path, "w", buffering=1, encoding="utf-8")
    faulthandler.enable(_fault_file)
    try:  # Ctrl-Break 手动转储（部分平台不支持，非致命）
        faulthandler.register(3, _fault_file, all_threads=True)
    except Exception as e:
        print(f"[warn] faulthandler.register(SIGBREAK) 不可用: {e}")
    return _fault_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank-model", default="bge-reranker-v2-m3")
    ap.add_argument("--window", type=int, default=400)
    ap.add_argument("--pool", type=int, default=20)
    ap.add_argument("--devices", default="cpu",
                    help="★ 必须用 devices=（复数）。FlagReranker 没有 device 参数，"
                         "传 device='cpu' 会被 **kwargs 吞掉，实际落到 cuda:0。"
                         "'auto' = 复刻那条静默走 GPU 的路径。")
    ap.add_argument("--max-min", type=float, default=60.0, help="墙钟上限（分钟）")
    ap.add_argument("--threads", type=int, default=-1, help="-1 = 不干预（复刻原始状态）")
    ap.add_argument("--fixed", action="store_true",
                    help="固定用第一条 query 循环（隔离数据变量，测资源/竞态）")
    ap.add_argument("--progress", type=Path, required=True)
    ap.add_argument("--marks", type=Path, required=True, help="阶段标记文件")
    ap.add_argument("--fault", type=Path, required=True)
    args = ap.parse_args()

    enable_faulthandler(args.fault)
    args.progress.parent.mkdir(parents=True, exist_ok=True)
    mark_f = open(args.marks, "w", buffering=1, encoding="utf-8")
    prog_f = open(args.progress, "w", buffering=1, encoding="utf-8")

    def emit(obj: dict) -> None:
        prog_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        prog_f.flush()
        os.fsync(prog_f.fileno())

    def mark(phase: str) -> None:
        mark_f.write(f"{time.time():.3f}\t{phase}\n")
        mark_f.flush()

    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))

    import psutil  # noqa: E402
    import torch  # noqa: E402

    proc = psutil.Process()
    emit({"ev": "start", "ts": time.time(), "pid": os.getpid(),
          "torch": torch.__version__, "cuda_avail": torch.cuda.is_available(),
          "threads_unset": torch.get_num_threads(),
          "interop": torch.get_num_interop_threads(),
          "args": vars(args) | {"progress": str(args.progress),
                                "marks": str(args.marks), "fault": str(args.fault)}})
    print(f"[env] torch={torch.__version__} threads={torch.get_num_threads()} "
          f"cuda={torch.cuda.is_available()}", flush=True)

    from xhs_rag.core.config import load_config  # noqa: E402
    from xhs_rag.index.retriever import Retriever  # noqa: E402
    from eval_retrieval import load_eval_set, dedup_by_note, retrieve_hybrid  # noqa: E402

    cfg = load_config()
    device_used = cfg.get("embedding.local.device", "auto")
    ret = Retriever(cfg)
    ret.rerank_enabled = False          # 与 eval_rerank_variants.py 完全一致
    mark("retriever.warmup.begin")
    ret.warmup()
    mark("retriever.warmup.end")
    th_after_warmup = torch.get_num_threads()
    print(f"[env] 配置 device={device_used} | warmup 后 torch threads="
          f"{th_after_warmup}", flush=True)
    emit({"ev": "warmup_done", "ts": time.time(), "threads": th_after_warmup,
          "embedding_device_cfg": device_used})

    eval_set = load_eval_set(ROOT / "data/eval/eval_set.jsonl")
    queries = [e["query"] for e in eval_set]
    if args.fixed:
        queries = [queries[0]]
    print(f"[env] 评测集 {len(eval_set)} 条；本次循环基数 {len(queries)} 条"
          f"{'（fixed 模式）' if args.fixed else ''}", flush=True)

    # ── 加载 reranker（与 eval 脚本同路径同参数）──
    from FlagEmbedding import FlagReranker  # noqa: E402
    model_dir = cfg.path("paths.data_dir") / "models" / args.rerank_model
    mark("reranker.load.begin")
    t_load = time.time()
    _dev = None if args.devices == "auto" else args.devices
    ranker = FlagReranker(str(model_dir), use_fp16=False, devices=_dev)
    mark("reranker.load.end")
    print(f"[env] reranker 加载完成 {time.time()-t_load:.0f}s | "
          f"torch threads={torch.get_num_threads()} | "
          f"batch_size={ranker.batch_size} | max_length={ranker.max_length}",
          flush=True)
    emit({"ev": "reranker_loaded", "ts": time.time(),
          "load_s": round(time.time() - t_load, 1),
          "threads": torch.get_num_threads(),
          "batch_size": ranker.batch_size, "max_length": ranker.max_length,
          "target_devices": list(ranker.target_devices)})

    # ★ 护栏：绝不允许再次悄悄跑错设备路径（这正是本轮踩到的坑）
    if args.devices != "auto" and args.devices not in list(ranker.target_devices):
        print(f"[FATAL] 设备未生效！期望 {args.devices!r}，"
              f"实际 {list(ranker.target_devices)!r} —— 参数名必须用 devices=（复数）",
              flush=True)
        sys.exit(3)

    if args.threads > 0:
        torch.set_num_threads(args.threads)
        print(f"[env] 已强制 torch threads={torch.get_num_threads()}", flush=True)

    # ── 阶段埋点：patch 掉 base.py 命名空间里的两个 compat 函数 + model.forward ──
    import FlagEmbedding.inference.reranker.encoder_only.base as febase  # noqa: E402

    _p = febase.prepare_for_model_compat
    _d = febase.pad_with_compat

    def prepare_marked(*a, **k):
        mark("prepare_for_model")
        return _p(*a, **k)

    def pad_marked(*a, **k):
        mark("pad_with_compat.begin")
        r = _d(*a, **k)
        mark("pad_with_compat.end")
        return r

    febase.prepare_for_model_compat = prepare_marked
    febase.pad_with_compat = pad_marked

    _fwd = ranker.model.forward

    def fwd_marked(*a, **k):
        mark("model.forward.begin")
        r = _fwd(*a, **k)
        mark("model.forward.end")
        return r

    ranker.model.forward = fwd_marked  # Module.__call__ 会取实例属性 → 生效
    mark("patched")

    # ── 主循环 ──
    deadline = time.time() + args.max_min * 60
    it = 0
    cyc = 0
    t_start = time.time()
    while time.time() < deadline:
        cyc += 1
        for q in queries:
            if time.time() >= deadline:
                break
            it += 1
            t0 = time.time()
            try:
                hits = retrieve_hybrid(ret, q, args.pool)
                if not hits:
                    emit({"ev": "skip", "i": it, "cyc": cyc, "q": q,
                          "reason": "no_hits", "t": round(time.time() - t0, 2)})
                    continue
                cands = [h["text"][: args.window] for h in hits]
                pairs = [[q, c] for c in cands]
                mark(f"call#{it}.begin")
                scores = ranker.compute_score(pairs, normalize=True)
                mark(f"call#{it}.end")
                if not isinstance(scores, list):
                    scores = [scores]
            except Exception as e:  # 观测用：任何 Python 级异常都要留痕
                emit({"ev": "error", "i": it, "cyc": cyc, "q": q,
                      "err": f"{type(e).__name__}: {e}",
                      "t": round(time.time() - t0, 2)})
                print(f"  [{it}] PY-ERROR {type(e).__name__}: {e}", flush=True)
                continue

            el = time.time() - t0
            try:
                rss = proc.memory_info().rss / 2**20
                handles = proc.num_handles()
            except Exception:
                rss = handles = -1
            emit({"ev": "ok", "i": it, "cyc": cyc, "q": q[:40],
                  "n": len(cands), "t": round(el, 2),
                  "rss_mb": round(rss, 1), "handles": handles,
                  "threads": torch.get_num_threads(),
                  "sum": round(float(sum(scores)), 4),
                  "min": round(float(min(scores)), 4),
                  "max": round(float(max(scores)), 4)})
            if it % 5 == 0 or it <= 3:
                print(f"  [{it}] {el:5.1f}s rss={rss:7.1f}MB h={handles} "
                      f"sum={sum(scores):8.3f} | {q[:30]}", flush=True)

    emit({"ev": "finish", "ts": time.time(), "iters": it, "cycles": cyc,
          "elapsed_s": round(time.time() - t_start, 1)})
    print(f"\n>>> 到点结束：{it} 次 compute_score / {cyc} 轮 / "
          f"{time.time()-t_start:.0f}s（未崩溃）", flush=True)


if __name__ == "__main__":
    main()
