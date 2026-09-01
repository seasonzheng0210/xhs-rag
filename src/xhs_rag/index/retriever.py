"""M5 检索 —— 语义检索 + rerank 重排。

流程:
  query → bge-m3 编码 → LanceDB cosine 搜索(top_k_in) → bge-reranker-v2-m3 重排(top_k_out)
rerank 模型本地化(bge-reranker-v2-m3,约 2.3GB,首次下载后常驻内存)。
"""
from __future__ import annotations

import time

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
        # 检索结果缓存: query → (时间戳, results)。rerank CPU 推理是主要耗时,
        # 相同/相近 query 短时间重复搜索直接秒回
        self._search_cache: dict[str, tuple[float, list[dict]]] = {}
        self._cache_ttl = float(cfg.get("rerank.cache_ttl", 300))
        self._cache_max = int(cfg.get("rerank.cache_max", 20))

    def warmup(self) -> None:
        """预热: 加载 embedder + reranker,把模型加载时间从首次查询挪到启动阶段。"""
        t0 = time.time()
        self.embedder.encode(["预热"])
        self._ensure_reranker()
        logger.info(f"检索预热完成(embedder+reranker),耗时 {time.time()-t0:.0f}s")

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
        """检索并重排,返回 [{note_id, title, section, text, score, url}]。

        返回前做同笔记补全: rerank 只用片段前 200 字符打分,
        配方/做法类笔记的用量常在图 OCR chunk 里(开头是 OCR 噪声,打分极低),
        会被挤出 topN 导致 LLM 看不到关键数字。因此只要某笔记有一个
        chunk 进 topN,就把它在库里的其他 chunk 一并补进结果尾部。
        """
        top_k_out = k or self.top_k_out
        # rerank 候选数: 配置优先, 至少比输出多 2 条供淘汰
        # (旧逻辑 max(top_k_out*4, ...) 在 k=5 时强制 20 候选, rerank CPU 每对 ~1.2s,
        #  直接把检索拖到 30s —— 候选数必须由配置控制)
        top_k_in = max(self.top_k_in, top_k_out + 2)

        # 0) 检索缓存命中直接返回(浅拷贝,防调用方修改污染缓存)
        cache_key = f"{top_k_out}:{query}"
        cached = self._search_cache.get(cache_key)
        if cached and time.time() - cached[0] < self._cache_ttl:
            return [dict(r) for r in cached[1]]

        # 1) 编码 query
        qvec = self.embedder.encode([query])[0]
        # 2) 向量检索
        tbl = self._get_table()
        hits = tbl.search(qvec).limit(top_k_in).to_list()
        if not hits:
            return []

        # 3) rerank 重排(候选截断 160 字符,CPU 推理与长度近似线性,越短越快)
        candidates = [h["text"][:160] for h in hits]
        scores = self._rerank(query, candidates)
        ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
        ranked = [x for x in ranked if x[1] is not None][:top_k_out]

        results = []
        for hit, score in ranked:
            results.append({
                "note_id": hit["note_id"],
                # seq 必须带上: _complete_note_chunks 用它做去重 key,
                # 缺了会退化成 (note_id, None) 导致同笔记多条 top5 重复补全
                "seq": hit.get("seq"),
                "title": hit["title"],
                "section": hit.get("section", ""),
                "text": hit["text"],
                "score": round(float(score), 4),
                "url": self._note_url(hit["note_id"]),
            })
        results = self._complete_note_chunks(hits, results)

        # 缓存(超上限时清空最旧一半,简化策略)
        if len(self._search_cache) >= self._cache_max:
            stale = sorted(self._search_cache.items(),
                           key=lambda x: x[1][0])[: self._cache_max // 2]
            for kk, _ in stale:
                self._search_cache.pop(kk, None)
        self._search_cache[cache_key] = (time.time(), results)
        return results

    def _note_url(self, note_id: str) -> str:
        """查笔记 url,db 不可用时留空(不影响检索)。"""
        if self.db is None:
            return ""
        try:  # 跨线程/连接失效时 url 留空,不影响检索
            note = self.db.get_note(note_id)
            return note.get("url", "") if note else ""
        except Exception:
            return ""

    def _complete_note_chunks(self, candidates: list[dict],
                              results: list[dict]) -> list[dict]:
        """同笔记补全: topN 结果所属笔记的其他 chunk 追加到尾部。

        只在 top_k_in 向量候选里补(候选是"相关但没进 top5"的同笔记片段),
        不从全表拉——长视频笔记可能切几十个帧 OCR chunk, 全拉会污染上下文。
        每笔记最多补 max_per_note 条, 按候选顺序取最相关的。
        """
        if not results:
            return results
        max_per_note = 3
        # note_id -> 按候选顺序(即相关性序)的 chunk 列表
        by_note: dict[str, list[dict]] = {}
        for h in candidates:
            by_note.setdefault(h["note_id"], []).append(h)

        seen = {(r["note_id"], r.get("seq")) for r in results}
        extra: list[dict] = []
        note_added: dict[str, int] = {}  # 按笔记维度计数,避免同笔记多条 top5 重复补
        for r in results:  # 保持 rerank 顺序,逐个笔记补全
            note = r["note_id"]
            url = r["url"]
            for h in by_note.get(note, []):
                if note_added.get(note, 0) >= max_per_note:
                    break
                key = (h["note_id"], h.get("seq"))
                if key in seen:
                    continue
                seen.add(key)
                extra.append({
                    "note_id": h["note_id"],
                    "title": h.get("title") or r["title"],
                    "section": h.get("section") or "",
                    "text": h.get("text") or "",
                    "score": 0.0,
                    "url": url,
                })
                note_added[note] = note_added.get(note, 0) + 1
        return results + extra

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
