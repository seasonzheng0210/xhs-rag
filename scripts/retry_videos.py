"""M2 补漏：重试 2 条视频下载失败（63592fa4 / 6a3fd030）。

原因回顾：
- 63592fa4：上次只嗅探到 video_id，URL 为空
- 6a3fd030：URL 有但文件没落盘（16 分钟长视频，签名可能已失效）
对策：重新打开详情页嗅探新 URL（带新 sign），立刻下载。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xhs_rag.core.config import load_config
from xhs_rag.crawler.browser import BrowserSession
from xhs_rag.crawler.media import MediaDownloader
from xhs_rag.crawler.note_detail import DetailCrawler
from xhs_rag.store.db import DB

TARGETS = [
    ("63592fa4000000001700e7f6", "泡椒鸡杂"),
    ("6a3fd0300000000006037cbf", "月子宝宝喂养实录"),
]

TARGETS = [t for t in TARGETS if (sys.argv[1:] and t[0] == sys.argv[1]) or not sys.argv[1:]]


def main() -> int:
    cfg = load_config()
    db = DB(cfg.path("paths.db"))
    tmp_root = cfg.path("paths.data_dir") / "tmp"
    vdir = tmp_root / "video"

    with BrowserSession(cfg, save_state=False) as ctx:
        crawler = DetailCrawler(ctx, cfg)
        dl = MediaDownloader(ctx, cfg, tmp_root)

        for nid, label in TARGETS:
            row = db.get_note(nid)
            if not row:
                print(f"[{nid}] 不存在，跳过")
                continue
            token = row.get("xsec_token") or ""

            print(f"[{nid}] 重试（{label}）", flush=True)
            t0 = time.time()
            res = crawler.fetch(nid, token, wait_video=25)
            print(f"[{nid}] fetch 耗时 {time.time()-t0:.0f}s, stopped={res.stopped_reason!r}, url={'有' if res.video_url else '无'}", flush=True)

            if res.stopped_reason:
                continue
            if not res.video_url:
                print(f"[{nid}] 仍未嗅探到视频 URL", flush=True)
                continue

            dest = dl.download_video(nid, res.video_url,
                                     timeout_ms=600_000 if nid.startswith("6a3fd030") else 120_000)
            if not dest:
                print(f"[{nid}] 下载失败", flush=True)
                continue

            size = dest.stat().st_size
            print(f"[{nid}] 视频已下载 {dest.name} ({size/1e6:.1f}MB)", flush=True)

            db.upsert_video(nid, {
                "duration_sec": res.video.get("duration_sec"),
                "width": res.video.get("width"),
                "height": res.video.get("height"),
                "video_url": res.video_url,
                "video_id": str(res.video.get("video_id") or ""),
                "md5": str(res.video.get("md5") or ""),
                "stream_types": res.video.get("stream_types"),
            })
            db.set_status(nid, "media_done")

            time.sleep(3)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
