-- 小红书收藏夹 RAG —— SQLite 数据模型
-- 状态机：new → listed → detailed → media_done → ocr_done → indexed
-- 任何一步失败都停在原状态，下次同步从断点继续。

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ────────────────────────────────────────────────────────────
-- 笔记主表
-- ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS notes (
    note_id        TEXT PRIMARY KEY,
    title          TEXT NOT NULL DEFAULT '',
    author_id      TEXT,
    author_name    TEXT,
    note_type      TEXT,                  -- normal | video
    desc           TEXT,                  -- 正文原文
    published_at   INTEGER,               -- 发布时间戳（毫秒）
    collected_at   INTEGER,               -- 首次入库时间
    liked_count    INTEGER DEFAULT 0,
    collect_count  INTEGER DEFAULT 0,
    comment_count  INTEGER DEFAULT 0,
    tags           TEXT,                  -- JSON array
    xsec_token     TEXT,                  -- ★ 跳回原帖的唯一钥匙，必须存
    xsec_source    TEXT DEFAULT 'pc_feed',
    url            TEXT,                  -- 完整链接（含 xsec_token）
    content_hash   TEXT,                  -- ★ 增量判定依据
    status         TEXT NOT NULL DEFAULT 'new',
    error_msg      TEXT,
    retry_count    INTEGER DEFAULT 0,
    last_synced_at INTEGER,
    deleted_at     INTEGER                -- 软删除：原笔记被删 / 已取消收藏
);

CREATE INDEX IF NOT EXISTS idx_notes_status ON notes(status);
CREATE INDEX IF NOT EXISTS idx_notes_hash   ON notes(content_hash);
CREATE INDEX IF NOT EXISTS idx_notes_collected ON notes(collected_at DESC);

-- ────────────────────────────────────────────────────────────
-- 图片表
-- ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS images (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    note_id        TEXT NOT NULL REFERENCES notes(note_id) ON DELETE CASCADE,
    seq            INTEGER NOT NULL,      -- 图序号，从 0 开始
    url            TEXT NOT NULL,
    local_path     TEXT,
    width          INTEGER,
    height         INTEGER,
    file_hash      TEXT,                  -- 图片内容 hash，避免重复 OCR
    ocr_text       TEXT,
    ocr_confidence REAL,
    ocr_engine     TEXT,                  -- rapidocr | vlm
    ocr_done       INTEGER DEFAULT 0,
    UNIQUE(note_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_images_note ON images(note_id);
CREATE INDEX IF NOT EXISTS idx_images_ocr  ON images(ocr_done);
CREATE INDEX IF NOT EXISTS idx_images_hash ON images(file_hash);

-- ────────────────────────────────────────────────────────────
-- 视频表（M2 新增：详情页拿到的视频元数据 + 播放器嗅探到的真实 URL）
-- ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS videos (
    note_id        TEXT PRIMARY KEY REFERENCES notes(note_id) ON DELETE CASCADE,
    duration_sec   REAL,                -- 视频时长（秒，capa.duration）
    width          INTEGER,
    height         INTEGER,
    video_url      TEXT,                -- ★ 播放器嗅探到的真实 mp4 URL（带 sign，有时效）
    video_id       TEXT,                -- mediaV2 里的 video_id（可用来重新取 URL）
    md5            TEXT,
    stream_types   TEXT,                -- JSON array，如 [258, 261]
    frame_count    INTEGER DEFAULT 0,
    frame_mode     TEXT,
    asr_status     TEXT DEFAULT 'none', -- none | pending | done | failed
    asr_text       TEXT,
    processed_at   INTEGER
);

CREATE INDEX IF NOT EXISTS idx_videos_note ON videos(note_id);

-- ────────────────────────────────────────────────────────────
-- 同步批次表：可观测 + 断点续传
-- ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sync_runs (
    run_id      TEXT PRIMARY KEY,
    started_at  INTEGER NOT NULL,
    finished_at INTEGER,
    trigger     TEXT,                     -- manual | scheduled
    listed      INTEGER DEFAULT 0,
    new_notes   INTEGER DEFAULT 0,
    updated     INTEGER DEFAULT 0,
    ocr_count   INTEGER DEFAULT 0,
    indexed     INTEGER DEFAULT 0,
    status      TEXT,                     -- running | success | failed | aborted
    error_msg   TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_started ON sync_runs(started_at DESC);

-- ────────────────────────────────────────────────────────────
-- 检索评测表：用于 A5 验收（30 条问题测试集，回归对比）
-- ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS eval_cases (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    question    TEXT NOT NULL,
    expect_note TEXT NOT NULL,            -- 期望命中的 note_id
    note        TEXT                      -- 备注
);

CREATE TABLE IF NOT EXISTS eval_results (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT,
    case_id     INTEGER NOT NULL REFERENCES eval_cases(id),
    hit_top1    INTEGER NOT NULL,
    hit_top5    INTEGER NOT NULL,
    top1_note   TEXT,
    created_at  INTEGER
);
