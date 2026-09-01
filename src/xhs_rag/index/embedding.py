"""M5 编码 —— 本地 bge-m3 embedding(FlagEmbedding)。

设计:
  - 懒加载 BGEM3FlagModel(首次调用才载入,2.2GB 模型,加载约 30s)
  - 返回 dense 向量(1024 维),sparse 权重一并返回(bge-m3 多语种稀疏能力,留作 hybrid 用)
  - CPU 推理:约 5-15s/chunk(i5-5200U),建索引是一次性的,可接受
  - query 编码加 4 线程 + LRU 缓存(Web 检索重复搜索秒回)
  - 断点续传:indexer 层控制,embedding 层只负责编码
"""
from __future__ import annotations

import threading
from pathlib import Path

from loguru import logger

from ..core.config import Config


class LocalEmbedder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model_dir = cfg.path("paths.data_dir") / "models" / "bge-m3"
        self.max_length = int(cfg.get("embedding.local.max_length", 1024))
        self._model = None
        self._cache: dict[str, list[float]] = {}  # query → dense 向量
        self._lock = threading.Lock()
        self._cache_max = 200

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

        logger.info("加载 bge-m3 模型(CPU)...")
        import torch

        torch.set_num_threads(4)  # i5 4 线程全开,单线程编码太慢
        self._model = BGEM3FlagModel(
            str(self.model_dir),
            use_fp16=False,
            device="cpu",
        )
        logger.info("bge-m3 加载完成")
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
            batch_size=int(self.cfg.get("embedding.local.batch_size", 8)),
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
            batch_size=int(self.cfg.get("embedding.local.batch_size", 8)),
        )
        dense = out["dense_vecs"].tolist()
        sparse = []
        for w in out["lexical_weights"]:
            # {token_id(int): weight} → 可 JSON 序列化的 {str: float}
            sparse.append({str(k): float(v) for k, v in w.items()})
        return dense, sparse
