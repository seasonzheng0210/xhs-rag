"""媒体下载 —— 图片 / 视频下载到临时目录。

★ 防盗链：小红书 CDN 需要带 Referer 才能下载，用浏览器 context.request
  （自动带 cookie），headers 补 Referer 和 UA。
★ 临时目录：data/tmp/{note_id}/ 放图片，data/tmp/video/ 放视频。
  M3 OCR / M4 抽帧处理后即删（用户决策：不保留原图/视频）。
★ 封面缩略图（data/thumbs/{note_id}.jpg）由 M3 生成，M2 只下载原图。
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from loguru import logger
from playwright.sync_api import BrowserContext

REFERER = "https://www.xiaohongshu.com/"

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36")


class MediaDownloader:
    def __init__(self, ctx: BrowserContext, cfg: Any, tmp_root: Path):
        self.ctx = ctx
        self.cfg = cfg
        self.tmp_root = Path(tmp_root)
        self.tmp_root.mkdir(parents=True, exist_ok=True)

    # ── 图片 ──────────────────────────────────────────────
    def download_images(self, note_id: str, urls: list[str],
                        dirname: str | None = None) -> list[Path]:
        """下载图片列表到 data/tmp/{note_id or dirname}/，返回本地路径。

        已存在的同名文件跳过（断点续传）。返回按 seq 排好序的路径列表。
        """
        note_dir = self.tmp_root / (dirname or note_id)
        note_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []

        for seq, url in enumerate(urls):
            ext = self._guess_ext(url)
            dest = note_dir / f"{seq:02d}{ext}"
            if dest.exists() and dest.stat().st_size > 0:
                paths.append(dest)
                continue
            ok = self._download(url, dest)
            if ok:
                paths.append(dest)
            else:
                logger.warning(f"图片下载失败 [{note_id}] #{seq}: {url[:80]}")
                paths.append(dest)  # 占位，文件不存在

        return paths

    def download_image(self, url: str, dest: Path) -> bool:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size > 0:
            return True
        return self._download(url, dest)

    # ── 视频 ──────────────────────────────────────────────
    def download_video(self, note_id: str, url: str,
                       timeout_ms: int = 120_000) -> Path | None:
        if not url:
            return None
        vdir = self.tmp_root / "video"
        vdir.mkdir(parents=True, exist_ok=True)
        dest = vdir / f"{note_id}.mp4"
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        ok = self._download(url, dest, timeout_ms=timeout_ms)
        return dest if ok else None

    # ── 内部 ──────────────────────────────────────────────
    def _download(self, url: str, dest: Path, timeout_ms: int = 120_000) -> bool:
        """带 Referer + UA 下载，失败返回 False。

        timeout_ms 可调：16 分钟长视频（200MB+）默认 120s 可能不够，
        调用方可传更大的值（如 600_000）。
        """
        try:
            resp = self.ctx.request.get(
                url,
                headers={"Referer": REFERER, "User-Agent": _UA},
                timeout=timeout_ms,
            )
            if resp.status != 200:
                logger.debug(f"下载 HTTP {resp.status}: {url[:80]}")
                return False
            body = resp.body()
            if not body:
                return False
            dest.write_bytes(body)
            return True
        except Exception as e:
            logger.debug(f"下载异常 {url[:80]}: {type(e).__name__}: {e}")
            return False

    @staticmethod
    def _guess_ext(url: str) -> str:
        """从 URL 猜扩展名（小红书图片可能是 jpg/webp/avif）。"""
        p = (url or "").split("?")[0].lower()
        for ext in (".jpg", ".jpeg", ".webp", ".avif", ".png", ".gif"):
            if p.endswith(ext):
                return ext
        return ".jpg"

    @staticmethod
    def file_md5(path: Path) -> str:
        return hashlib.md5(path.read_bytes()).hexdigest()
