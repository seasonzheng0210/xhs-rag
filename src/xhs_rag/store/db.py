"""SQLite 存储层 —— 断点续传的根基。

状态机：new → listed → detailed → media_done → ocr_done → indexed
任何一步失败都停在原状态，下次同步从断点继续（schema 见 store/schema.sql）。
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

from loguru import logger

from ..core.config import ROOT

SCHEMA = ROOT / "src" / "xhs_rag" / "store" / "schema.sql"


def content_hash(parts: list[str]) -> str:
    """增量判定依据：标题/作者/类型/时间等可空字段组合的 sha1。

    注意：M1 只有列表数据，hash 只覆盖列表可见字段；M2 拉到详情后
    会重算一次（hash 变则触发重新 OCR / embedding）。
    """
    raw = "\x1f".join(p or "" for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


class DB:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._init_schema()

    def _init_schema(self) -> None:
        if SCHEMA.exists():
            self.conn.executescript(SCHEMA.read_text(encoding="utf-8"))
        self.conn.commit()

    def close(self) -> None:
        try:
            self.conn.commit()
            self.conn.close()
        except Exception:
            pass

    def __enter__(self) -> "DB":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ──────────────────────────────────────────────────────
    # notes
    # ──────────────────────────────────────────────────────
    def upsert_note(self, n: dict) -> str:
        """单条写入。已存在则只更新「列表可见字段 + 状态」，
        不动 status 的既有进度（detailed/ocr 等更后面的状态不倒退）。"""
        note_id = n["note_id"]
        old = self.get_note(note_id)

        new_status = n.get("status", "new")
        if old and old.get("status") not in ("new", "listed"):
            # 已进入更后阶段，只更新元数据，保留进度
            new_status = old["status"]

        sql = """
        INSERT INTO notes (
            note_id, title, author_id, author_name, note_type, "desc",
            published_at, collected_at, liked_count, collect_count, comment_count,
            tags, xsec_token, xsec_source, url, content_hash, status,
            error_msg, last_synced_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(note_id) DO UPDATE SET
            title=excluded.title, author_id=excluded.author_id,
            author_name=excluded.author_name, note_type=excluded.note_type,
            "desc"=excluded."desc", published_at=excluded.published_at,
            liked_count=excluded.liked_count, collect_count=excluded.collect_count,
            comment_count=excluded.comment_count, tags=excluded.tags,
            xsec_token=excluded.xsec_token, url=excluded.url,
            content_hash=excluded.content_hash, status=excluded.status,
            last_synced_at=excluded.last_synced_at
        """
        now = int(time.time() * 1000)
        self.conn.execute(sql, (
            note_id,
            n.get("title", ""),
            n.get("author_id"),
            n.get("author_name"),
            n.get("note_type"),
            n.get("desc", ""),
            n.get("published_at"),
            n.get("collected_at") or now,
            n.get("liked_count", 0),
            n.get("collect_count", 0),
            n.get("comment_count", 0),
            json.dumps(n.get("tags", []), ensure_ascii=False),
            n.get("xsec_token"),
            n.get("xsec_source", "pc_feed"),
            n.get("url"),
            n.get("content_hash"),
            new_status,
            n.get("error_msg"),
            now,
        ))
        return note_id

    def upsert_notes(self, notes: Iterable[dict]) -> tuple[int, int]:
        """批量写入，返回 (新增数, 更新数)。"""
        new_cnt = updated_cnt = 0
        for n in notes:
            old = self.get_note(n["note_id"])
            if old is None:
                new_cnt += 1
            else:
                updated_cnt += 1
            self.upsert_note(n)
        self.conn.commit()
        return new_cnt, updated_cnt

    def get_note(self, note_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM notes WHERE note_id=?", (note_id,)).fetchone()
        return dict(row) if row else None

    def notes_by_status(self, status: str, limit: int = 1000) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM notes WHERE status=? ORDER BY collected_at DESC LIMIT ?",
            (status, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def count(self, status: str | None = None) -> int:
        if status:
            row = self.conn.execute("SELECT COUNT(*) c FROM notes WHERE status=?", (status,)).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) c FROM notes").fetchone()
        return int(row["c"]) if row else 0

    def set_status(self, note_id: str, status: str,
                   error_msg: str | None = None) -> None:
        """显式推进状态机（M2：listed → detailed → media_done）。"""
        if error_msg:
            self.conn.execute(
                "UPDATE notes SET status=?, error_msg=?, retry_count=retry_count+1 "
                "WHERE note_id=?",
                (status, error_msg, note_id),
            )
        else:
            self.conn.execute(
                "UPDATE notes SET status=?, error_msg=NULL, last_synced_at=? WHERE note_id=?",
                (status, int(time.time() * 1000), note_id),
            )
        self.conn.commit()

    # ──────────────────────────────────────────────────────
    # images / videos —— M2 详情阶段落库
    # ──────────────────────────────────────────────────────
    def upsert_images(self, note_id: str, urls: list[str],
                      paths: list[str] | None = None) -> int:
        """写图片列表。已存在同 seq 的跳过（断点续传）。返回实际写入数。"""
        written = 0
        for seq, url in enumerate(urls):
            row = self.conn.execute(
                "SELECT 1 FROM images WHERE note_id=? AND seq=?",
                (note_id, seq),
            ).fetchone()
            if row:
                continue
            local_path = (paths[seq] if paths and seq < len(paths) else None)
            self.conn.execute(
                "INSERT INTO images (note_id, seq, url, local_path) VALUES (?,?,?,?)",
                (note_id, seq, url, local_path),
            )
            written += 1
        self.conn.commit()
        return written

    def upsert_video(self, note_id: str, v: dict) -> None:
        """写/更新视频元数据。v 为 None 或空时跳过。"""
        if not v:
            return
        self.conn.execute(
            """
            INSERT INTO videos (
                note_id, duration_sec, width, height, video_url,
                video_id, md5, stream_types, processed_at
            ) VALUES (?,?,?,?,?,?,?,?,?)
            ON CONFLICT(note_id) DO UPDATE SET
                duration_sec=excluded.duration_sec, width=excluded.width,
                height=excluded.height,
                video_url=CASE WHEN excluded.video_url <> '' THEN excluded.video_url
                               ELSE videos.video_url END,
                video_id=excluded.video_id, md5=excluded.md5,
                stream_types=excluded.stream_types
            """,
            (
                note_id,
                v.get("duration_sec"),
                v.get("width"),
                v.get("height"),
                v.get("video_url") or "",
                v.get("video_id"),
                v.get("md5"),
                v.get("stream_types"),
                int(time.time() * 1000),
            ),
        )
        self.conn.commit()

    def get_video(self, note_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM videos WHERE note_id=?", (note_id,)).fetchone()
        return dict(row) if row else None

    def images_of(self, note_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM images WHERE note_id=? ORDER BY seq", (note_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ──────────────────────────────────────────────────────
    # sync_runs —— 每次同步可观测 + 可断点续传
    # ──────────────────────────────────────────────────────
    def start_run(self, trigger: str = "manual") -> str:
        run_id = uuid.uuid4().hex[:12]
        self.conn.execute(
            "INSERT INTO sync_runs (run_id, started_at, trigger, status) VALUES (?,?,?,?)",
            (run_id, int(time.time() * 1000), trigger, "running"),
        )
        self.conn.commit()
        return run_id

    def finish_run(self, run_id: str, status: str, **counts: int) -> None:
        """结束一次同步。run_id 必已由 start_run 创建，直接 UPDATE。
        （早先版本用 INSERT...ON CONFLICT，漏写 NOT NULL 的 started_at 会崩。）"""
        now = int(time.time() * 1000)
        if counts:
            fields = ", ".join(f"{k}=?" for k in counts)
            sql = f"UPDATE sync_runs SET finished_at=?, status=?, {fields} WHERE run_id=?"
            self.conn.execute(sql, (now, status, *counts.values(), run_id))
        else:
            self.conn.execute(
                "UPDATE sync_runs SET finished_at=?, status=? WHERE run_id=?",
                (now, status, run_id),
            )
        self.conn.commit()

    # ──────────────────────────────────────────────────────
    # jsonl 留档 —— 原始响应落盘，崩溃可重放、字段可回溯
    # ──────────────────────────────────────────────────────
    @staticmethod
    def append_jsonl(path: Path, obj: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
