"""日志：控制台 + 文件轮转。

Windows 控制台默认 GBK，中文日志不重新配置编码会直接抛 UnicodeEncodeError，
所以这里对 stdout 强制 UTF-8 —— 这是本项目在 Windows 上必须先解决的问题。
"""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logging(level: str = "INFO", file: Path | str | None = None) -> None:
    logger.remove()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:
        pass

    logger.add(
        sys.stdout,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <level>{message}</level>",
        colorize=True,
    )

    if file:
        path = Path(file)
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(path),
            level=level,
            rotation="10 MB",
            retention="30 days",
            encoding="utf-8",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {name}:{line} | {message}",
        )


__all__ = ["logger", "setup_logging"]
