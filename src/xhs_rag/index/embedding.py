"""M5 编码 —— 本地 bge-m3 embedding(FlagEmbedding)。

设计:
  - 懒加载 BGEM3FlagModel(首次调用才载入,约 2.2GB 模型)
  - 返回 dense 向量(1024 维),sparse 权重一并返回(bge-m3 多语种稀疏能力,留作 hybrid 用)
  - **设备按用途分离**(2026-09-30 起):
      * 建索引(purpose="index")默认 auto —— 有 CUDA 就用 GPU。91 篇全量重建
        在 CPU 上约 9.5 分钟,GPU 上降到分钟级
      * 在线查询(purpose="query")默认 cpu —— serve/mcp 是**常驻进程**,查询单条
        且命中 LRU 缓存,走 CPU 可避免长期占用显存(曾因显存争抢触发过 BSOD)
  - 显式配置 cuda 但检测不到 GPU 时**直接抛错**,不静默回退(慢 5-8 倍且难察觉)
  - batch_size 由设备决定:GPU 默认 32,CPU 默认 8
  - 断点续传:indexer 层控制,embedding 层只负责编码
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from loguru import logger

from ..core.config import Config

_DEVICES = ("auto", "cuda", "cpu")
_CUDA_HINT = (
    "pip install torch==2.14.0+cu130 "
    "--index-url https://download.pytorch.org/whl/cu130"
)


class LocalEmbedder:
    def __init__(self, cfg: Config, purpose: str = "query"):
        self.cfg = cfg
        self.purpose = purpose if purpose in ("index", "query") else "query"
        self.model_dir = cfg.path("paths.data_dir") / "models" / "bge-m3"
        self.max_length = int(cfg.get("embedding.local.max_length", 1024))
        self.threads = int(cfg.get("embedding.local.torch_threads", 4))
        self.device, self.use_fp16 = self._resolve_device()
        default_bs = 32 if self.device == "cuda" else 8
        self.batch_size = int(cfg.get("embedding.local.batch_size", default_bs))
        self._model = None
        self._cache: dict[str, list[float]] = {}  # query → dense 向量
        self._lock = threading.Lock()
        self._cache_max = 200

    def _resolve_device(self) -> tuple[str, bool]:
        """决定推理设备,返回 (device, use_fp16)。

        auto: 有 CUDA 用 CUDA,否则回退 CPU 并**打 WARNING**(不算静默回退)。
        cuda: 强制要求 GPU,检测不到直接抛错 —— 避免"以为在跑 GPU,其实在磨 CPU"。
        """
        key = (
            "embedding.local.query_device"
            if self.purpose == "query"
            else "embedding.local.device"
        )
        default = "cpu" if self.purpose == "query" else "auto"
        want = str(self.cfg.get(key, default)).strip().lower()
        if want not in _DEVICES:
            raise ValueError(f"{key} 取值非法:{want!r},只支持 {_DEVICES}")

        if want == "cpu":
            return "cpu", False

        avail = False
        try:
            import torch

            avail = bool(torch.cuda.is_available())
        except Exception as exc:  # torch 缺失/构建异常
            logger.debug(f"torch CUDA 探测失败:{exc}")

        if avail:
            return "cuda", True
        if want == "cuda":
            raise RuntimeError(
                f"{key}=cuda 但检测不到可用 CUDA 设备(当前 torch 很可能是 CPU 构建)。"
                f"拒绝静默回退到 CPU。修法:{_CUDA_HINT}"
            )
        logger.warning(f"{key}=auto 但检测不到 CUDA,已回退 CPU(索引速度约慢 5-8 倍)")
        return "cpu", False

    def _ensure_model(self):
        """懒加载模型。模型不存在时从 modelscope 下载。"""
        if self._model is not None:
            return self._model
        if not (self.model_dir / "pytorch_model.bin").exists() and not (
            self.model_dir / "model.safetensors"
        ).exists():
            logger.info("首次使用,下载 bge-m3 模型(~2.2GB,仅一次)")
            import modelscope

            self.model_dir.mkdir(parents=True, exist_ok=True)
            modelscope.snapshot_download(
                "BAAI/bge-m3", local_dir=str(self.model_dir))
        from FlagEmbedding import BGEM3FlagModel
        import torch

        t0 = time.time()
        if self.device == "cuda":
            logger.info(
                f"加载 bge-m3 模型(CUDA fp16, batch={self.batch_size}, purpose={self.purpose})..."
            )
            self._model = BGEM3FlagModel(
                str(self.model_dir),
                use_fp16=self.use_fp16,
                device="cuda:0",
            )
        else:
            logger.info(
                f"加载 bge-m3 模型(CPU {self.threads} 线程, batch={self.batch_size}, "
                f"purpose={self.purpose})..."
            )
            torch.set_num_threads(self.threads)  # CPU 下才需要压线程数
            self._model = BGEM3FlagModel(
                str(self.model_dir),
                use_fp16=False,
                device="cpu",
            )
        logger.info(
            f"bge-m3 加载完成(device={self.device}, fp16={self.use_fp16}, "
            f"{time.time() - t0:.0f}s)"
        )
        return self._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        """批量编码,返回 dense 向量列表(1024 维)。单条 query 走 LRU 缓存。"""
        if not texts:
            return []
        if len(texts) == 1:
            cached = self._cache_get(texts[0])
            if cached is not None:
                return [cached]
        model = self._ensure_model()
        out = model.encode(
            texts,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            batch_size=self.batch_size,
        )
        vecs = out["dense_vecs"].tolist()
        if len(texts) == 1:
            self._cache_put(texts[0], vecs[0])
        return vecs

    def _cache_get(self, q: str) -> list[float] | None:
        with self._lock:
            return self._cache.get(q)

    def _cache_put(self, q: str, vec: list[float]):
        with self._lock:
            if len(self._cache) >= self._cache_max:
                self._cache.clear()  # 超上限直接清空(简化 LRU,200 条够用)
            self._cache[q] = vec

    def encode_with_sparse(self, texts: list[str]) -> tuple[list[list[float]], list[dict]]:
        """编码并返回 (dense, sparse_weights)。sparse 是 {token_id: weight} 字典。"""
        if not texts:
            return [], []
        model = self._ensure_model()
        out = model.encode(
            texts,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=True,
            batch_size=self.batch_size,
        )
        dense = out["dense_vecs"].tolist()
        sparse = []
        for w in out["lexical_weights"]:
            # {token_id(int): weight} → 可 JSON 序列化的 {str: float}
            sparse.append({str(k): float(v) for k, v in w.items()})
        return dense, sparse
