"""M5 索引 —— 向量化 vault/ 全部 Markdown,写入 LanceDB。

流程:
  vault/*.md → Chunker 切分 → LocalEmbedder 编码 → LanceDB upsert
幂等:--force 重建表;否则按 note_id 增量(已存在则跳过)。
"""
from __future__ import annotations

import json
from pathlib import Path

import lancedb
from loguru import logger

from ..core.config import Config
from .chunk import Chunker
from .embedding import LocalEmbedder


class Indexer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.vault = cfg.path("paths.vault_dir")
        self.lance_dir = cfg.path("paths.data_dir") / "lancedb"
        self.table_name = cfg.get("vectorstore.table_name", "xhs_notes")
        self.chunker = Chunker(cfg)
        self.embedder = LocalEmbedder(cfg)
        self._db = None

    def _connect(self):
        if self._db is None:
            self.lance_dir.mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(str(self.lance_dir))
        return self._db

    def count_indexed(self) -> int:
        """返回已入库索引的 note_id 数(覆盖检查用)。"""
        db = self._connect()
        try:
            tbl = db.open_table(self.table_name)
            return len(set(tbl.to_pandas()["note_id"].unique()))
        except Exception:
            return 0

    def run(self, force: bool = False, limit: int | None = None) -> dict:
        mds = sorted(self.vault.glob("*.md"))
        if limit:
            mds = mds[:limit]
        logger.info(f"M5 建索引: {len(mds)} 篇 Markdown")

        db = self._connect()
        if force and self.table_name in db.table_names():
            db.drop_table(self.table_name)

        # 已索引的 note_id(增量跳过)
        # 注意:不能用 tbl.to_batches()——lancedb 0.37 无此方法,异常被吞会导致全量重编码
        indexed: set[str] = set()
        try:
            tbl = db.open_table(self.table_name)
            indexed = set(tbl.to_pandas()["note_id"].unique())
        except Exception:
            pass  # 表不存在,全新索引

        chunks: list[dict] = []
        for md in mds:
            if md.stem in indexed and not force:
                continue
            chunks.extend(self.chunker.split_md(md))
        logger.info(f"待编码 chunk: {len(chunks)}")

        if not chunks:
            return {"md": len(mds), "chunks": 0, "skip_md": len(indexed)}

        # 分批编码(避免一次吃太多内存)
        batch_size = int(self.cfg.get("embedding.local.batch_size", 8))
        rows = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]
            dense = self.embedder.encode([c["text"] for c in batch])
            for c, vec in zip(batch, dense):
                rows.append({
                    "note_id": c["note_id"],
                    "seq": c["seq"],
                    "title": c["title"],
                    "section": c["section"],
                    "text": c["text"],
                    "vector": vec,
                    "json": json.dumps({"chunk": c}, ensure_ascii=False),
                })
            if (i // batch_size) % 5 == 0:
                logger.info(f"[{min(i+batch_size, len(chunks))}/{len(chunks)}] chunk 编码中...")

        # 写入 LanceDB(表存在→追加,不存在→创建;force→重建)
        exists = self.table_name in db.table_names()
        if force and exists:
            db.drop_table(self.table_name)
            exists = False
        if exists:
            tbl = db.open_table(self.table_name)
            tbl.add(rows)
            logger.info(f"追加 {len(rows)} rows 到已有表 {self.table_name}")
        else:
            tbl = db.create_table(self.table_name, data=rows)
        # 向量索引: 只在语料够大时建 ANN。
        # IVF_PQ 的 num_partitions/nprobes 是给大表调的;几百行的小表建了反而
        # 召回饿死(空分区多 → 实际返回远少于 limit,实测 218 行只回 ~6 条)。
        # 小表直接 flat 精确扫描,毫秒级且召回=全表。
        if len(rows) >= int(self.cfg.get("vectorstore.ann_min_rows", 20000)):
            try:
                tbl.create_index(
                    metric="cosine", index_type="IVF_PQ",
                    num_partitions=16, num_sub_vectors=64,
                )
                logger.info("语料规模达标,向量索引(IVF_PQ)创建完成")
            except Exception as e:
                logger.warning(f"向量索引创建失败(不影响检索): {e}")
        else:
            logger.info(
                f"语料 {len(rows)} 行 < ann_min_rows,保持 flat 精确扫描"
                "(小表 ANN 会饿死召回)")

        logger.success(f"M5 索引完成: {len(mds)} 篇, {len(rows)} chunks")
        return {"md": len(mds), "chunks": len(rows), "skip_md": len(indexed)}
