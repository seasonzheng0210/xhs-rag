"""命令行入口。

M0 阶段只开放三个命令：
  doctor   环境自检（Python / Playwright / Chromium / ffmpeg / API key）
  login    扫码登录（首次必须可见浏览器窗口）
  check    登录态检查 —— M0 的验收标准就是「关掉重开后 check 依然通过」
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from loguru import logger

from .core.config import Config, load_config, save_user_id


def _setup(cfg: Config, level: str = "INFO") -> None:
    from .core.logging import setup_logging

    setup_logging(level, cfg.path("logging.file") if cfg.get("logging.file") else None)


# ── doctor ────────────────────────────────────────────────
def cmd_doctor(cfg: Config) -> int:
    print("\n环境自检\n" + "-" * 52)
    ok_all = True

    # Python
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 11)
    ok_all &= ok
    print(f"[{'OK ' if ok else 'FAIL'}] Python {v.major}.{v.minor}.{v.micro}  (需 >= 3.11)")

    # Playwright
    try:
        import playwright
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            path = p.chromium.executable_path
        ok = Path(path).exists()
        ok_all &= ok
        print(f"[{'OK ' if ok else 'FAIL'}] Playwright {playwright.__version__ if hasattr(playwright,'__version__') else ''}".strip())
        print(f"       Chromium: {path if ok else '未安装，请运行 playwright install chromium'}")
    except Exception as e:
        ok_all = False
        print(f"[FAIL] Playwright 不可用: {e}")

    # ffmpeg
    ffmpeg = find_ffmpeg(cfg)
    ok = ffmpeg is not None
    print(f"[{'OK ' if ok else 'WARN'}] ffmpeg: {ffmpeg or '未找到（M4 视频抽帧才用到，M0-M3 不影响）'}")
    if not ok:
        print("       安装：winget install Gyan.FFmpeg  或  下载后配置 video.ffmpeg_path")

    # API key
    key = cfg.api_key
    ok = bool(key)
    print(f"[{'OK ' if ok else 'WARN'}] 硅基流动 API key: {'已配置 (' + str(len(key)) + ' 位)' if ok else '未配置（M5 索引才需要）'}")
    if ok:
        print(f"       也可运行 python scripts/verify_keys.py 做一次连通性实测")

    # 运行环境
    print(f"[INFO] 项目根目录: {cfg.root}")
    print(f"[INFO] 配置文件  : {cfg.source}")
    print(f"[INFO] 登录态    : {cfg.path('paths.storage_state')}")
    print("-" * 52)
    print("全部就绪\n" if ok_all else "存在缺失项，见上\n")
    return 0 if ok_all else 1


def find_ffmpeg(cfg: Config) -> str | None:
    """按 配置路径 → PATH → imageio-ffmpeg 内置 的顺序找。"""
    configured = cfg.get("video.ffmpeg_path", "")
    if configured and Path(configured).exists():
        return str(configured)

    exe = shutil.which("ffmpeg")
    if exe:
        return exe

    try:  # pip install imageio-ffmpeg 会自带一份静态构建
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


# ── login ─────────────────────────────────────────────────
def cmd_login(cfg: Config, timeout: int | None = None) -> int:
    from .auth.qrcode_login import qrcode_login
    from .crawler.browser import BrowserSession

    cfg.ensure_dirs("paths.data_dir", "paths.storage_state")
    timeout = timeout or int(cfg.get("auth.qrcode_timeout", 180))
    png = cfg.path("auth.qrcode_output") if cfg.get("auth.qrcode_output") else Path("data/auth/qrcode.png")

    with BrowserSession(cfg, headless=bool(cfg.get("auth.headless", False))) as ctx:
        result = qrcode_login(ctx, timeout=timeout, png_path=png)

        if not result.ok:
            logger.error(f"登录未完成：{result.reason}")
            return 1

        # storage_state 由 BrowserSession.__exit__ 统一写，这里只补 user_id
        if result.user_id:
            save_user_id(cfg, result.user_id)
            logger.info(f"user_id 已写入 config.yaml: {result.user_id}")
        else:
            # 登录本身是成功的，缺的只是拼 profile 页 URL 用的 ID。
            # 不清不楚地静默过去，会让 M1 报错时无从查起。
            logger.warning(
                "登录成功，但没能提取到 user_id（M1 抓收藏列表要用）。\n"
                "        一般重跑一次 login 即可拿到；也可以手动填到 config.yaml 的 auth.user_id。"
            )

    state = cfg.path("paths.storage_state")
    if state.exists():
        logger.success(f"登录态已保存：{state}")
    else:
        logger.warning("登录态文件未生成")
    return 0


# ── check ─────────────────────────────────────────────────
def cmd_check(cfg: Config, online: bool = True) -> int:
    from .auth import session

    ok = session.report(cfg, online=online)
    return 0 if ok else 1


# ── collect ───────────────────────────────────────────────
def cmd_collect(cfg: Config, headless: bool | None = None, max_pages: int = 200) -> int:
    from .auth import session
    from .auth.session import CaptchaRequired
    from .crawler.browser import BrowserSession
    from .crawler.collect import CollectCrawler
    from .store.db import DB, content_hash

    user_id = str(cfg.get("auth.user_id", "") or "").strip()
    if not user_id:
        logger.error(
            "缺少 user_id —— 先在 config/config.yaml 的 auth.user_id 填上你的 24 位用户 ID。\n"
            "获取：登录后跑 python -m xhs_rag.cli check，或手动在网页端打开自己的主页从 URL 复制。"
        )
        return 2

    # 采集前先确认登录态（只读本地文件，零请求；在线验证由 M1 采集本身承担）
    ok_local, reason = session.has_stored_session(cfg)
    if not ok_local:
        logger.error(f"无有效登录态：{reason}。先跑 python -m xhs_rag.cli login")
        return 2

    cfg.ensure_dirs("paths.data_dir", "paths.jsonl_dir", "paths.db")
    db = DB(cfg.path("paths.db"))
    run_id = db.start_run("manual")
    jsonl = cfg.path("paths.jsonl_dir") / f"collect_{run_id}.jsonl"

    logger.info(f"开始同步收藏列表（run={run_id}，落盘 {jsonl}）")
    try:
        with BrowserSession(cfg, headless=headless) as ctx:
            crawler = CollectCrawler(ctx, cfg, user_id=user_id, jsonl=jsonl)

            def on_page(note: dict) -> None:
                # M1 只落列表可见字段；content_hash 用列表字段算，
                # M2 拉到详情后会重算（hash 变则触发重新 OCR / embedding）
                nid = note["note_id"]
                note["content_hash"] = content_hash([
                    note.get("title"), note.get("author_id"),
                    note.get("note_type"), str(note.get("published_at", "")),
                    str(note.get("liked_count", "")), str(note.get("collect_count", "")),
                ])
                note["status"] = "listed"
                db.upsert_note(note)

            result = crawler.sync(on_page=on_page, max_pages=max_pages)

    except CaptchaRequired as e:
        logger.warning(
            f"触发验证码风控（{e}）。本次已抓到的数据已落库，下次续跑即可。\n"
            "        处理：停止一切自动化访问，用日常浏览器正常浏览小红书一段时间冷却后再试。"
        )
        db.finish_run(run_id, "failed", listed=db.count("listed"))
        return 1
    except Exception as e:
        logger.exception(f"采集异常终止: {e}")
        db.finish_run(run_id, "failed", listed=db.count("listed"))
        return 1

    total = db.count()
    listed = db.count("listed")
    logger.success(
        f"本轮完成：抓取 {result.pages} 页，去重后 {len(result.note_ids)} 条；"
        f"库内共 {total} 条（{listed} 条已 listed）"
    )
    logger.info(f"停止原因：{result.stopped_reason}")
    db.finish_run(run_id, "success",
                  listed=listed, new_notes=len(result.note_ids))
    return 0


# ── detail ────────────────────────────────────────────────
def cmd_detail(cfg: Config, headless: bool | None = None,
               limit: int | None = None, note_id: str | None = None,
               skip_media: bool = False) -> int:
    """M2：抓详情 + 下载媒体。

    流程：取 status='listed' 的笔记 → 逐条打开详情页提取正文/图片/视频
    → 下载媒体到临时目录 → 状态推进 detailed → media_done。
    断点续传：失败留在 listed，重跑自动继续；已 detailed 的不重抓。
    """
    import time as _t

    from .auth import session
    from .crawler.browser import BrowserSession
    from .crawler.media import MediaDownloader
    from .crawler.note_detail import DetailCrawler, _jitter
    from .store.db import DB, content_hash

    ok_local, reason = session.has_stored_session(cfg)
    if not ok_local:
        logger.error(f"无有效登录态：{reason}。先跑 python -m xhs_rag.cli login")
        return 2

    cfg.ensure_dirs("paths.data_dir", "paths.db")
    db = DB(cfg.path("paths.db"))
    run_id = db.start_run("manual")

    # 取待处理笔记
    if note_id:
        todo = [n for n in [db.get_note(note_id)] if n]
    else:
        limit = limit or int(cfg.get("sync.max_notes_per_run", 500))
        todo = db.notes_by_status("listed", limit=limit)
    logger.info(f"M2 待处理 {len(todo)} 条（limit={limit}，skip_media={skip_media}）")

    done = failed = 0
    try:
        with BrowserSession(cfg, headless=headless) as ctx:
            crawler = DetailCrawler(ctx, cfg)
            media = MediaDownloader(
                ctx, cfg, cfg.path("paths.data_dir") / "tmp"
            )
            for i, note in enumerate(todo, 1):
                nid = note["note_id"]
                logger.info(f"[{i}/{len(todo)}] 抓详情 {nid} "
                            f"(标题: {(note.get('title') or '')[:24]})")
                try:
                    res = crawler.fetch(nid, note.get("xsec_token") or "")
                except Exception as e:
                    logger.exception(f"详情抓取异常 {nid}")
                    db.set_status(nid, "listed", f"detail_exc: {e}")
                    failed += 1
                    continue

                if not res.note:
                    logger.warning(f"详情失败（{res.stopped_reason}），跳过")
                    db.set_status(nid, "listed", res.stopped_reason)
                    failed += 1
                    continue

                # 更新 notes（保留原有 status 进度；详情字段补齐）
                merged = dict(note)
                for k, v in res.note.items():
                    if v not in (None, "", [], 0) or k == "note_type":
                        merged[k] = v
                # content_hash 重算：标题/作者/类型/正文/互动（M2 数据更全）
                merged["content_hash"] = content_hash([
                    merged.get("title"), merged.get("author_id"),
                    merged.get("note_type"), merged.get("desc"),
                    str(merged.get("published_at", "")),
                    str(merged.get("liked_count", "")),
                    str(merged.get("collect_count", "")),
                ])
                merged["status"] = "detailed"
                db.upsert_note(merged)

                # 图片落库
                if res.images:
                    db.upsert_images(nid, res.images)

                # 视频元数据 + URL
                if res.note.get("note_type") == "video":
                    res.video["video_url"] = res.video_url
                    db.upsert_video(nid, res.video)

                # 下载媒体
                if skip_media:
                    db.set_status(nid, "media_done")
                    done += 1
                    continue

                img_ok = 0
                paths = media.download_images(nid, res.images)
                img_ok = sum(1 for p in paths if p.exists() and p.stat().st_size > 0)
                if res.images and img_ok < len(res.images):
                    logger.warning(f"[{nid}] 图片 {img_ok}/{len(res.images)} 下载成功")

                vpath = None
                if res.note.get("note_type") == "video":
                    vpath = media.download_video(nid, res.video_url)
                    if vpath:
                        logger.info(f"[{nid}] 视频已下载 {vpath.name} "
                                    f"({vpath.stat().st_size / 1024 / 1024:.1f}MB)")
                    else:
                        logger.warning(f"[{nid}] 视频下载失败")

                db.set_status(nid, "media_done")
                done += 1

                # 节奏控制：详情间隔 1.5~3 秒
                _t.sleep(_jitter() / 1000)

        logger.success(f"M2 完成：成功 {done} 条，失败 {failed} 条")
        db.finish_run(run_id, "success" if not failed else "failed",
                      updated=done)
        return 0 if not failed else 1

    except Exception as e:
        logger.exception(f"M2 异常终止: {e}")
        db.finish_run(run_id, "failed", updated=done)
        return 1


# ── ocr ──────────────────────────────────────────────────
def cmd_ocr(cfg: Config, limit: int | None = None, note_id: str | None = None) -> int:
    """M3：图片 OCR + 生成 Markdown 落盘（vault/）。

    hybrid：本地 RapidOCR 为主，低置信度/短文本上硅基流动 PaddleOCR-VL 兜底。
    断点续传：images.ocr_done 置 1 即视为完成，重跑自动跳过。
    """
    from .process.ocr import OcrProcessor
    from .store.db import DB

    cfg.ensure_dirs("paths.data_dir", "paths.db", "paths.vault_dir")
    db = DB(cfg.path("paths.db"))

    if note_id:
        note = db.get_note(note_id)
        if not note:
            logger.error(f"笔记不存在: {note_id}")
            return 2
        processor = OcrProcessor(cfg, db)
        r = processor.process_note(note)
        try:
            md = processor.build_markdown(note)
            logger.success(f"[{note_id}] OCR {r['ok']} 图, Markdown: {md}")
            db.set_status(note_id, "ocr_done")
        except Exception as e:
            logger.exception(f"Markdown 生成失败: {e}")
            return 1
        return 0

    processor = OcrProcessor(cfg, db)
    stats = processor.run(limit=limit)
    return 0 if stats["failed"] == 0 else 1


# ── video ─────────────────────────────────────────────────
def cmd_video(cfg: Config, note_id: str | None = None) -> int:
    """M4：视频抽帧 OCR + ASR 语音转写,结果写入 vault Markdown。

    断点续传:videos.asr_status='done' 即视为完成,重跑自动跳过。
    """
    from .process.video import VideoProcessor
    from .store.db import DB

    cfg.ensure_dirs("paths.data_dir", "paths.db", "paths.vault_dir")
    db = DB(cfg.path("paths.db"))
    proc = VideoProcessor(cfg, db)

    if note_id:
        r = proc.process_note(note_id)
        logger.success(f"[{note_id}] 帧OCR {r['frame']} 张, ASR {'成功' if r['asr'] else '失败/跳过'}")
        return 0 if r["asr"] else 1

    stats = proc.run()
    return 0 if stats["failed"] == 0 else 1


# ── index / search ───────────────────────────────────────
def cmd_index(cfg: Config, force: bool = False, limit: int | None = None) -> int:
    """M5：向量化 vault/ Markdown 并写入 LanceDB。"""
    from .index.indexer import Indexer

    cfg.ensure_dirs("paths.data_dir", "paths.vault_dir")
    idx = Indexer(cfg)
    stats = idx.run(force=force, limit=limit)
    logger.success(
        f"索引完成: {stats['md']} 篇, {stats['chunks']} chunks, "
        f"跳过已索引 {stats['skip_md']} 篇"
    )
    return 0


def cmd_search(cfg: Config, query: str, k: int = 5) -> int:
    """M5：语义检索（本地 embedding + rerank）。"""
    from .index.retriever import Retriever
    from .store.db import DB

    cfg.ensure_dirs("paths.data_dir", "paths.db")
    db = DB(cfg.path("paths.db"))
    r = Retriever(cfg, db)
    results = r.search(query, k=k)
    if not results:
        logger.warning("没有检索到相关内容")
        return 1
    print(f"\n『{query}』 检索结果 (top {len(results)}):\n")
    for i, hit in enumerate(results, 1):
        print(f"  [{i}] {hit['title']}" + (f" — {hit['section']}" if hit["section"] else ""))
        print(f"      score={hit['score']}  {hit['url']}")
        text = hit["text"].replace("\n", " ")[:160]
        print(f"      {text}...\n")
    return 0


def cmd_ask(cfg: Config, query: str, k: int = 5) -> int:
    """M8：检索 + LLM 生成带引用的回答（终端流式打印，方便调 prompt）。"""
    from .index.retriever import Retriever
    from .qa.answer import Answerer, LLMUnavailable, pretty_stream
    from .store.db import DB

    cfg.ensure_dirs("paths.data_dir", "paths.db")
    db = DB(cfg.path("paths.db"))
    r = Retriever(cfg, db)
    t0 = time.time()
    results = r.search(query, k=k)
    search_secs = round(time.time() - t0, 1)
    if not results:
        logger.warning("没有检索到相关内容")
        return 1
    print(f"\n『{query}』 检索 {len(results)} 条 / {search_secs}s\n")
    for i, hit in enumerate(results, 1):
        print(f"  [{i}] {hit['title']}"
              + (f" — {hit['section']}" if hit["section"] else ""))
        print(f"      {hit['text'].replace(chr(10), ' ')[:120]}...")

    ans = Answerer(cfg)
    ok, why = ans.available()
    if not ok:
        logger.warning(f"跳过 AI 回答：{why}")
        return 0
    print(f"\n────── AI 回答（{ans.model}"
          f"{'，thinking' if ans.thinking else ''}） ──────\n")
    try:
        st = pretty_stream(query, results, ans)
    except LLMUnavailable as e:
        logger.error(f"AI 回答失败：{e}")
        return 1
    print(f"\n────── 生成 {st['secs']}s ──────")
    return 0


# ── sync ────────────────────────────────────────────────
def cmd_sync(cfg: Config, headless: bool = False,
             max_pages: int = 50, skip_media: bool = False) -> int:
    """M7：一键全量同步流水线。

    collect → detail → ocr → video → index，各环节断点续传天然增量：
    - 新增收藏才会进库，重复的 upsert 刷新互动数据
    - detail 只处理 listed，ocr 跳过已识别，video 跳过已转写，index 跳过已向量化
    适合定时任务：无人值守，每次只处理增量，风控触发时数据已落库、下次续跑。
    """
    t0 = time.time()
    results = {}
    logger.info("===== M7 sync 开始 =====")
    results["collect"] = cmd_collect(cfg, headless=headless, max_pages=max_pages)
    results["detail"] = cmd_detail(cfg, headless=headless, skip_media=skip_media)
    results["ocr"] = cmd_ocr(cfg)
    results["video"] = cmd_video(cfg)
    results["index"] = cmd_index(cfg)
    # 幂等检查：确认最新笔记已进索引
    try:
        from .store.db import DB
        from .index.indexer import Indexer
        db = DB(cfg.path("paths.db"))
        n_total, n_indexed = db.count(), Indexer(cfg).count_indexed()
        results["coverage"] = 0 if n_indexed >= n_total else 1
        logger.info(f"覆盖: {n_indexed}/{n_total} 篇已入索引")
    except Exception as e:
        logger.warning(f"覆盖检查跳过: {e}")
    bad = [k for k, v in results.items() if v]
    logger.success(f"===== M7 sync 结束, 耗时 {time.time()-t0:.0f}s, "
                   f"各环节: {results}, 异常环节: {bad or '无'} =====")
    return 1 if bad else 0


# ── serve ────────────────────────────────────────────────
def cmd_serve(cfg: Config) -> int:
    """M6：启动 Web UI(模型常驻,手机可访问)。"""
    from .serve.server import serve as run_server

    return run_server(cfg)


# ── main ──────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xhs", description="小红书收藏夹 RAG")
    parser.add_argument("--config", type=Path, help="指定配置文件路径")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="环境自检")
    p_login = sub.add_parser("login", help="扫码登录")
    p_login.add_argument("--timeout", type=int, help="等待扫码秒数")
    p_check = sub.add_parser("check", help="登录态检查")
    p_check.add_argument("--offline", action="store_true", help="只查本地文件，不启浏览器")
    p_collect = sub.add_parser("collect", help="同步收藏列表（M1）")
    p_collect.add_argument("--headless", action="store_true", help="无头模式（不推荐，风控更严）")
    p_collect.add_argument("--max-pages", type=int, default=200, help="翻页上限")
    p_detail = sub.add_parser("detail", help="抓详情+下载媒体（M2）")
    p_detail.add_argument("--headless", action="store_true", help="无头模式（不推荐）")
    p_detail.add_argument("--limit", type=int, help="最多处理条数")
    p_detail.add_argument("--note-id", help="只处理指定 note_id")
    p_detail.add_argument("--skip-media", action="store_true", help="只抓详情不下载媒体")
    p_ocr = sub.add_parser("ocr", help="图片 OCR + 生成 Markdown（M3）")
    p_ocr.add_argument("--limit", type=int, help="最多处理条数")
    p_ocr.add_argument("--note-id", help="只处理指定 note_id")
    p_video = sub.add_parser("video", help="视频抽帧+ASR 转写（M4）")
    p_video.add_argument("--note-id", help="只处理指定 note_id")
    p_index = sub.add_parser("index", help="向量化 Markdown 建索引（M5）")
    p_index.add_argument("--force", action="store_true", help="重建索引表")
    p_index.add_argument("--limit", type=int, help="只索引前 N 篇")
    p_search = sub.add_parser("search", help="语义检索（M5）")
    p_search.add_argument("query", help="检索关键词/问题")
    p_search.add_argument("-k", type=int, default=5, help="返回条数")
    p_ask = sub.add_parser("ask", help="检索 + LLM 带引用回答（M8）")
    p_ask.add_argument("query", help="问题")
    p_ask.add_argument("-k", type=int, default=5, help="喂给 LLM 的片段数")
    p_sync = sub.add_parser("sync", help="一键全量同步（M7：collect→detail→ocr→video→index）")
    p_sync.add_argument("--headless", action="store_true", help="无头模式（定时任务用，风控更严）")
    p_sync.add_argument("--max-pages", type=int, default=50, help="收藏列表翻页上限")
    p_sync.add_argument("--skip-media", action="store_true", help="跳过媒体下载，只抓详情")
    sub.add_parser("serve", help="启动 Web UI（M6，手机可访问）")

    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    _setup(cfg, "DEBUG" if args.verbose else "INFO")

    if args.cmd == "doctor":
        return cmd_doctor(cfg)
    if args.cmd == "login":
        return cmd_login(cfg, args.timeout)
    if args.cmd == "check":
        return cmd_check(cfg, online=not args.offline)
    if args.cmd == "collect":
        return cmd_collect(cfg, headless=None if not args.headless else True,
                           max_pages=args.max_pages)
    if args.cmd == "detail":
        return cmd_detail(cfg, headless=None if not args.headless else True,
                          limit=args.limit, note_id=args.note_id,
                          skip_media=args.skip_media)
    if args.cmd == "ocr":
        return cmd_ocr(cfg, limit=args.limit, note_id=args.note_id)
    if args.cmd == "video":
        return cmd_video(cfg, note_id=args.note_id)
    if args.cmd == "index":
        return cmd_index(cfg, force=args.force, limit=args.limit)
    if args.cmd == "search":
        return cmd_search(cfg, args.query, k=args.k)
    if args.cmd == "ask":
        return cmd_ask(cfg, args.query, k=args.k)
    if args.cmd == "sync":
        return cmd_sync(cfg, headless=args.headless,
                        max_pages=args.max_pages, skip_media=args.skip_media)
    if args.cmd == "serve":
        return cmd_serve(cfg)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
