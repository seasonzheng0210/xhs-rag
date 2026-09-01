"""M5 检索 —— 语义检索 + rerank 重排。

流程:
  query → bge-m3 编码 → LanceDB cosine 搜索(top_k_in) → bge-reranker-v2-m3 重排(top_k_out)
rerank 模型本地化(bge-reranker-v2-m3,约 2.3GB,首次下载后常驻内存)。
"""
from __future__ import annotations

from loguru import logger

from ..core.config import Config
from .embedding import LocalEmbedder


class Retriever:
    def __init__(self, cfg: Config, db=None):
        self.cfg = cfg
        self.db = db  # sqlite DB(取笔记链接用),可空
        self.embedder = LocalEmbedder(cfg)
        self.lance_dir = cfg.path("paths.data_dir") / "lancedb"
        self.table_name = cfg.get("vectorstore.table_name", "xhs_notes")
        self.top_k_in = int(cfg.get("rerank.top_k_in", 50))
        self.top_k_out = int(cfg.get("rerank.top_k_out", 5))
        self._reranker = None
        self._table = None

    def _get_table(self):
        import lancedb

        if self._table is None:
            db = lancedb.connect(str(self.lance_dir))
            self._table = db.open_table(self.table_name)
        return self._table

    def _ensure_reranker(self):
        """懒加载本地 bge-reranker-base。"""
        if self._reranker is not None:
            return self._reranker
        model_id = self.cfg.get("rerank.local.model", "BAAI/bge-reranker-base")
        model_dir = self.cfg.path("paths.data_dir") / "models" / model_id.split("/")[-1]
        if not (model_dir / "pytorch_model.bin").exists() and not (
            model_dir / "model.safetensors"
        ).exists():
            logger.info(f"首次使用,下载 {model_id} 模型(仅一次)")
            import modelscope

            model_dir.mkdir(parents=True, exist_ok=True)
            modelscope.snapshot_download(
                model_id, local_dir=str(model_dir),
                ignore_file_pattern=["onnx/*"])
        from FlagEmbedding import FlagReranker

        logger.info("加载 reranker 模型(CPU)...")
        import torch

        torch.set_num_threads(4)  # i5 4 线程全开
        self._reranker = FlagReranker(str(model_dir), use_fp16=False, device="cpu")
        # FlagReranker 初始化可能重置 torch 线程池,加载后强制设回
        torch.set_num_threads(4)
        logger.info(f"reranker 加载完成, torch threads={torch.get_num_threads()}")
        return self._reranker

    def search(self, query: str, k: int | None = None) -> list[dict]:
        """检索并重排,返回 [{note_id, title, section, text, score, url}]。"""
        top_k_out = k or self.top_k_out
        top_k_in = max(top_k_out * 4, self.top_k_in)

        # 1) 编码 query
        qvec = self.embedder.encode([query])[0]
        # 2) 向量检索
        tbl = self._get_table()
        hits = tbl.search(qvec).limit(top_k_in).to_list()
        if not hits:
            return []

        # 3) rerank 重排(候选截断 200 字符,CPU 推理 O(n^2) 长文本太慢)
        candidates = [h["text"][:200] for h in hits]
        scores = self._rerank(query, candidates)
        ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
        ranked = [x for x in ranked if x[1] is not None][:top_k_out]

        results = []
        for hit, score in ranked:
            url = ""
            if self.db is not None:
                try:  # 跨线程/连接失效时 url 留空,不影响检索
                    note = self.db.get_note(hit["note_id"])
                    url = note.get("url", "") if note else ""
                except Exception:
                    url = ""
            results.append({
                "note_id": hit["note_id"],
                "title": hit["title"],
                "section": hit.get("section", ""),
                "text": hit["text"],
                "score": round(float(score), 4),
                "url": url,
            })
        return results

    def _rerank(self, query: str, candidates: list[str]) -> list[float | None]:
        try:
            model = self._ensure_reranker()
            pairs = [[query, c] for c in candidates]
            raw = model.compute_score(pairs)
            if isinstance(raw, float):  # 单条
                return [raw]
            return [float(x) for x in raw]
        except Exception as e:
            logger.warning(f"rerank 失败,退回向量检索原始顺序: {e}")
            # 降级:按向量相似度原序,给个伪分数(0-1 归一)
            return [None] * len(candidates)
