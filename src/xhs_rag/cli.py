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

    session_bs = BrowserSession(cfg, headless=bool(cfg.get("auth.headless", False)))
    with session_bs as ctx:
        result = qrcode_login(ctx, timeout=timeout, png_path=png)

        if not result.ok:
            # ★ 事务式写回（2026-09-06）：登录失败/超时 = 中间态 cookie，
            #   绝不允许 __exit__ 把它写回覆盖旧登录态（曾致假登录视图）。
            #   旧 storage_state.json 保留，下次 login 还能正常注入重试。
            session_bs.allow_state_commit = False
            logger.error(f"登录未完成：{result.reason}")
            logger.info("旧登录态已保留（storage_state.json 未被覆盖），可直接重试 login")
            return 1

        # storage_state 由 BrowserSession.__exit__ 统一写（此时 allow_state_commit=True）
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
def cmd_collect(cfg: Config, headless: bool | None = None, max_pages: int = 200,
                manage_run: bool = True, trigger: str = "manual") -> int:
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
        # P1-1 失败告警：独立调用时才报（sync 内部由 cmd_sync 统一告警，避免重复）
        if manage_run and cfg.get("auth.notify_on_expire", True):
            from .notify import notify
            notify(cfg, "login_expired", "登录态失效", f"{reason}（采集未执行）")
        return 2

    cfg.ensure_dirs("paths.data_dir", "paths.jsonl_dir", "paths.db")
    db = DB(cfg.path("paths.db"))
    # 唯一 run 由 cmd_sync 持有（manage_run=False 时跳过，避免同一次同步多行 run）
    run_id = db.start_run(trigger) if manage_run else None
    # jsonl 落盘名：manage_run=False 时用时间戳占位，保证 CollectCrawler 总有落点
    run_label = run_id or f"sched-{int(time.time() * 1000)}"
    jsonl = cfg.path("paths.jsonl_dir") / f"collect_{run_label}.jsonl"

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
        if manage_run:
            db.finish_run(run_id, "failed", listed=db.count("listed"))
        else:
            db.close()  # manage_run=False 时跳过 finish_run（其内部已 commit），需自行提交释放写锁
        return 1
    except Exception as e:
        logger.exception(f"采集异常终止: {e}")
        if manage_run:
            db.finish_run(run_id, "failed", listed=db.count("listed"))
        else:
            db.close()
        return 1

    total = db.count()
    listed = db.count("listed")
    logger.success(
        f"本轮完成：抓取 {result.pages} 页，去重后 {len(result.note_ids)} 条；"
        f"库内共 {total} 条（{listed} 条已 listed）"
    )
    logger.info(f"停止原因：{result.stopped_reason}")
    if manage_run:
        db.finish_run(run_id, "success",
                      listed=listed, new_notes=len(result.note_ids))
    else:
        db.close()
    return 0


# ── detail ────────────────────────────────────────────────
def cmd_detail(cfg: Config, headless: bool | None = None,
               limit: int | None = None, note_id: str | None = None,
               skip_media: bool = False,
               manage_run: bool = True, trigger: str = "manual") -> int:
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
    # 唯一 run 由 cmd_sync 持有（manage_run=False 时跳过，避免同一次同步多行 run）
    run_id = db.start_run(trigger) if manage_run else None

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
        if manage_run:
            db.finish_run(run_id, "success" if not failed else "failed",
                          updated=done)
        else:
            db.close()
        return 0 if not failed else 1

    except Exception as e:
        logger.exception(f"M2 异常终止: {e}")
        if manage_run:
            db.finish_run(run_id, "failed", updated=done)
        else:
            db.close()
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


def cmd_agent(cfg: Config, query: str, max_steps: int = 8) -> int:
    """M10：Agent 模式 —— LLM 决策循环，自主多步调工具后作答。"""
    from .agent import RAGAgent
    from .qa.answer import Answerer

    cfg.ensure_dirs("paths.data_dir", "paths.db")
    ans = Answerer(cfg)
    ok, why = ans.available()
    if not ok:
        logger.error(f"Agent 需要 LLM：{why}")
        return 1
    print(f"\n『{query}』 Agent 决策循环（max_steps={max_steps}）:\n")
    agent = RAGAgent(cfg, verbose=True, max_steps=max_steps, session="cli")
    result = agent.run(query)
    print(f"\n{'=' * 60}")
    print(f"（{result['steps']} 步工具调用 / {result['secs']}s"
          + ("，触发步数上限" if result.get("hit_limit") else "") + "）\n")
    print(result["answer"])
    return 0


def _ocr_query(cfg: Config, image: str) -> str | None:
    """P2-3 查询端多模态：把用户给的截图 OCR 成检索词。

    复用采集侧同一套 OcrEngine（本地 RapidOCR 优先，命中 escalate 条件
    才上云 VLM 兜底），因此不需要任何新依赖。
    在 CLI 主线程里加载 OCR 模型——避开 Windows 工作线程里首次加载
    推理引擎死等的坑（同 mcp_server 顶部的说明）。
    返回压平后的检索词；无文字/失败返回 None。
    """
    from .process.ocr import OcrEngine

    p = Path(image).expanduser()
    if not p.exists():
        logger.error(f"图片不存在：{p}")
        return None
    t0 = time.time()
    try:
        res = OcrEngine(cfg).ocr_image(p)
    except Exception as e:
        logger.error(f"图片 OCR 失败：{e}")
        return None
    text = (res or {}).get("text") or ""
    # OCR 是多行的，检索词要压成单行并去掉过短的噪声行
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) >= 2]
    query = " ".join(lines).strip()[:300]
    if not query:
        logger.warning(f"图片没识别出可用文字：{p.name}")
        return None
    logger.info(f"图片检索词（{res.get('engine')} / conf="
                f"{res.get('confidence', 0):.2f} / {time.time() - t0:.1f}s）："
                f"{query[:80]}")
    return query


def cmd_search(cfg: Config, query: str, k: int = 5,
               image: str | None = None) -> int:
    """M5：语义检索（本地 embedding + rerank）。--image 时先 OCR 成检索词。"""
    from .index.retriever import Retriever
    from .store.db import DB

    if image:
        ocr_q = _ocr_query(cfg, image)
        if not ocr_q:
            return 1
        # 文字检索词 + 图片文字一起用（截图的正文常比用户手打的关键词更全）
        query = f"{query} {ocr_q}".strip() if query else ocr_q

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


def cmd_ask(cfg: Config, query: str, k: int = 5,
            image: str | None = None) -> int:
    """M8：检索 + LLM 生成带引用的回答（终端流式打印，方便调 prompt）。"""
    from .index.retriever import Retriever
    from .qa.answer import Answerer, LLMUnavailable, pretty_stream
    from .store.db import DB

    if image:
        ocr_q = _ocr_query(cfg, image)
        if not ocr_q:
            return 1
        query = f"{query} {ocr_q}".strip() if query else ocr_q
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
    适合定时任务：交互会话内运行（锁屏仍运行，重启后登录补跑），每次只处理增量，
    风控触发时数据已落库、下次续跑。

    Q1：整次同步持有唯一一条 sync_runs 记录（trigger=scheduled），
    子步骤传 manage_run=False，避免同一次同步产生多行 run。
    硬崩溃（进程被杀/OOM/异常中断）也落 failed，便于 status 区分「中途死了」。
    """
    from .store.db import DB
    from .index.indexer import Indexer
    from .schedule import write_sync_state

    db = DB(cfg.path("paths.db"))
    run_id = db.start_run("scheduled")  # 唯一 run，覆盖整次同步

    t0 = time.time()
    results: dict[str, int] = {}
    logger.info("===== M7 sync 开始 =====")

    def _finish(error_msg: str | None = None) -> int:
        bad = [k for k, v in results.items() if v]
        dur = time.time() - t0
        listed = indexed = 0
        try:
            listed = db.count("listed")
            indexed = Indexer(cfg).count_indexed()
        except Exception as e:
            logger.warning(f"计数跳过: {e}")
        if error_msg or bad:
            msg = error_msg or f"异常环节: {', '.join(bad)}"
            db.finish_run(run_id, "failed", listed=listed, indexed=indexed,
                          error_msg=msg)
            # P1-1 失败告警：定时任务无人值守，失败必须能被看见（Web 横幅/CLI/桌面）
            from .notify import notify
            if results.get("collect") == 2:  # cmd_collect 约定 2=登录态失效
                kind, title = "login_expired", "定时同步失败：登录态失效"
            else:
                kind, title = "sync_failed", "定时同步失败"
            notify(cfg, kind, title, f"{msg}（耗时 {dur:.0f}s）")
        else:
            db.finish_run(run_id, "success", listed=listed, indexed=indexed)
        # 兜底通道：保留 sync_state.json 写入（status 兼容回退）
        cov = results.get("coverage", 1)
        state_steps = {k: v for k, v in results.items() if k != "coverage"}
        write_sync_state(cfg, state_steps, cov, dur)
        # P1-1：成功但索引有缺口 → 警告级告警（不算失败，但要让人知道）
        if cov and not (error_msg or bad):
            from .notify import notify
            notify(cfg, "coverage_gap", "索引覆盖不全",
                   "部分笔记未进索引（见日志），可跑 python -m xhs_rag.cli index 补建",
                   level="warning")
        logger.success(
            f"===== M7 sync 结束, 耗时 {dur:.0f}s, "
            f"各环节: {results}, 异常环节: {bad or '无'} ====="
        )
        return 1 if (error_msg or bad) else 0

    try:
        results["collect"] = cmd_collect(
            cfg, headless=headless, max_pages=max_pages,
            manage_run=False, trigger="scheduled")
        results["detail"] = cmd_detail(
            cfg, headless=headless, skip_media=skip_media,
            manage_run=False, trigger="scheduled")
        results["ocr"] = cmd_ocr(cfg)
        results["video"] = cmd_video(cfg)
        results["index"] = cmd_index(cfg)
    except Exception as e:
        logger.exception(f"sync 异常中断: {e}")
        return _finish(error_msg=f"sync 异常中断: {e}")

    # 幂等检查：确认最新笔记已进索引
    try:
        n_total, n_indexed = db.count(), Indexer(cfg).count_indexed()
        results["coverage"] = 0 if n_indexed >= n_total else 1
        logger.info(f"覆盖: {n_indexed}/{n_total} 篇已入索引")
    except Exception as e:
        logger.warning(f"覆盖检查跳过: {e}")

    # P0（2026-09-16）：长期记忆消化挂到同步尾部。
    #   此前 digest 只在 serve 启动时后台跑一次 —— 纯 CLI / 只跑定时同步的用法下，
    #   对话永远躺在 dialog_log 里没被消化，"长期记忆"等于装了但从没跑过。
    #   放在 _finish() 之前、且不写 results（results 非零即算"异常环节"会把同步判失败）。
    _maybe_auto_digest(cfg)

    return _finish()


def _maybe_auto_digest(cfg: Config) -> None:
    """M11 记忆消化：同步尾部自动跑一次（P0, 2026-09-16）。

    三条硬约束：
    - **空跑要便宜**：没有待消化对话时立刻返回，不构造 Answerer
      （否则每次同步白付 10–20s 模型加载）。
    - **绝不能拖垮同步**：整段 try/except 吞掉，任何异常只 warning 级别记录 +
      （失败时）发 warning 告警，同步的退出码不受影响。
    - **不进 results**：`_finish()` 里 `results` 任何非零值都判定为"异常环节"→ 同步 failed，
      所以消化结果用独立变量/日志承载。
    """
    if not cfg.get("memory.auto_digest", True):
        logger.info("记忆消化: 配置关闭 (memory.auto_digest=false)，跳过")
        return
    db_path = str(cfg.path("paths.db"))
    try:
        from .memory import counts, run_digest

        pending = counts(db_path).get("pending_rounds", 0)
        if not pending:
            logger.info("记忆消化: 无待处理对话，跳过")
            return

        from .qa.answer import Answerer

        logger.info(f"记忆消化: 待处理 {pending} 轮，开始加工...")
        out = run_digest(
            db_path, Answerer(cfg), verbose=False,
            max_blocks=int(cfg.get("memory.auto_digest_blocks", 6)),
        )
        if out.get("error"):
            logger.warning(f"记忆消化跳过（LLM 不可用？）: {out['error']}")
            return
        logger.info(
            f"记忆消化完成: 块 {out.get('processed_blocks', 0)}"
            f" / 轮 {out.get('rounds', 0)}"
            f" / 摘要 {out.get('summaries', 0)}"
            f" / 画像 {out.get('profiles', 0)}"
            f" / 失败 {out.get('errors', 0)}"
        )
        if out.get("errors"):
            from .notify import notify

            notify(cfg, "memory_digest_failed", "长期记忆消化部分失败",
                   f"{out['errors']} 个对话块加工失败（未标记已消化，下次同步自动重试）",
                   level="warning")
    except Exception as e:  # 消化是附加能力，绝不因它让同步失败
        logger.warning(f"记忆消化异常（已忽略，不影响同步）: {e}")


# ── setup ────────────────────────────────────────────────
def cmd_setup(cfg: Config, login_timeout: int = 180, headless: bool = False,
              max_pages: int = 50, skip_media: bool = False,
              serve: bool = False) -> int:
    """一键建库：登录态检查(失效才扫码) → 全流水线同步 → 汇报。

    面向新机器 / 新用户：一条命令跑完「登录 → 收藏 → 详情 → OCR → ASR → 索引」。
    最后一步是可选的 Web UI 启动（--serve，会阻塞前台）。
    """
    t0 = time.time()
    logger.info("===== xhs setup 开始 =====")
    steps: list[tuple[str, int]] = []

    # 1) 登录态：先服务端核验，失效才弹扫码窗
    logger.info("[1/3] 检查登录态...")
    if cmd_check(cfg, online=True) == 0:
        logger.success("登录态有效")
    else:
        logger.warning("登录态失效或不存在，弹出浏览器扫码（{}s 超时）...", login_timeout)
        cmd_login(cfg, timeout=login_timeout)
        if cmd_check(cfg, online=True) != 0:
            logger.error("扫码后登录态仍未通过服务端核验。"
                         "请重跑 setup，或单独执行: python -m xhs_rag.cli login")
            return 1
        logger.success("扫码登录成功")
    steps.append(("登录", 0))

    # 2) 全流水线同步（各环节断点续传，重复跑只处理增量）
    logger.info("[2/3] 全流水线同步 (collect→detail→ocr→video→index)...")
    rc = cmd_sync(cfg, headless=headless, max_pages=max_pages,
                  skip_media=skip_media)
    steps.append(("同步", rc))
    if rc:
        logger.error("同步链路有环节失败，见上方日志。可重跑 setup 续传。")

    # 3) 汇报 + 可选 Web UI
    dur = time.time() - t0
    if serve:
        url = f"http://{cfg.get('serve.host', '0.0.0.0')}:{cfg.get('serve.port', 8765)}"
        logger.info(f"[3/3] 启动 Web UI: {url}（Ctrl+C 停止）")
        cmd_serve(cfg)
    else:
        logger.info("[3/3] 建库完成")
        logger.success(f"===== xhs setup 结束, 耗时 {dur:.0f}s =====")
        logger.info("下一步: python -m xhs_rag.cli serve  # 启动 Web UI（手机同局域网可访问）")
    return 1 if any(r for _, r in steps if r) else 0


# ── serve ────────────────────────────────────────────────
def cmd_serve(cfg: Config) -> int:
    """M6：启动 Web UI(模型常驻,手机可访问)。"""
    from .serve.server import serve as run_server

    return run_server(cfg)


# ── memory（M11 长期记忆）────────────────────────────────
def cmd_memory(cfg: Config, action: str, max_blocks: int = 6,
               yes: bool = False) -> int:
    """digest/show/clear 三动作。db_path 独立连接(线程安全)。"""
    db_path = str(cfg.path("paths.db"))
    if action == "digest":
        from .memory import counts, run_digest
        from .qa.answer import Answerer

        answerer = Answerer(cfg)
        out = run_digest(db_path, answerer, max_blocks=max_blocks)
        if out.get("error"):
            print(f"[FAIL] digest 无法执行: {out['error']}")
            return 1
        print(f"处理对话块: {out.get('processed_blocks', 0)}"
              f" / 轮次: {out.get('rounds', 0)}"
              f" / 新摘要: {out.get('summaries', 0)}"
              f" / 画像候选: {out.get('profiles', 0)}"
              f" / 失败块: {out.get('errors', 0)}")
        c = counts(db_path)
        print(f"当前: 待消化 {c['pending_rounds']} 轮"
              f" / 摘要 {c['digests']} 条 / 画像 {c['profiles']} 条")
        return 0
    if action == "show":
        from .memory import counts, digests, profiles

        c = counts(db_path)
        print(f"===== 记忆状态: 待消化 {c['pending_rounds']} 轮"
              f" / 摘要 {c['digests']} / 画像 {c['profiles']} =====")
        ds = digests(db_path, 10)
        if ds:
            print("\n-- 对话摘要(最近 10) --")
            for d in ds:
                print(f"[{d['created_at']}] ({d['span']}轮) {d['summary']}")
        ps = profiles(db_path, 30)
        if ps:
            print("\n-- 用户画像(hit 降序) --")
            for p in ps:
                print(f"[x{p['hit_count']}] {p['content']}"
                      f"{'  ← ' + p['source'] if p.get('source') else ''}")
        return 0
    if action == "clear":
        if not yes:
            print("清空全部记忆(摘要+画像+对话流水)不可撤销。")
            print("确认请执行: python -m xhs_rag.cli memory clear --yes")
            return 1
        from .memory import clear as mem_clear

        print(mem_clear(db_path))
        return 0
    print(f"未知 memory 动作: {action}")
    return 1


# ── schedule（M7 定时调度）────────────────────────────────
def cmd_schedule(cfg: Config, action: str, at: str = "09:00") -> int:
    """install/uninstall/status/run 四动作。设计见 docs/M7-调度设计.md。"""
    import subprocess as sp

    from . import schedule as sched

    if action == "install":
        rc, out = sched.install(cfg, at)
        print(f"注册计划任务 {sched.TASK_NAME} (每日 {at}): "
              f"{'成功' if rc == 0 else '失败'}")
        if out:
            print(out)
        if rc == 0:
            print(f"查看: taskschd.msc 搜索 {sched.TASK_NAME}，"
                  f"或 python -m xhs_rag.cli schedule status")
        return rc

    if action == "uninstall":
        rc, out = sched.uninstall()
        print(f"删除计划任务 {sched.TASK_NAME}: "
              f"{'成功' if rc == 0 else '失败(可能本就不存在)'}")
        if out:
            print(out)
        return 0 if rc == 0 else 0  # 幂等：不存在也算卸载成功

    if action == "status":
        registered = sched.exists()
        print(f"计划任务 {sched.TASK_NAME}: "
              f"{'已注册' if registered else '未注册'}")
        if registered:
            _, out = sched.status_detail()
            # 只挑关键行打印，避免 /V 长输出刷屏
            keys = ("任务名", "TaskName", "状态", "Status",
                    "下次运行时间", "Next Run Time",
                    "上次运行时间", "Last Run Time",
                    "上次结果", "Last Result", "要运行的任务", "Task To Run")
            for line in out.splitlines():
                if any(k in line for k in keys):
                    print("  " + line.strip())
        st = sched.read_sync_state(cfg)
        if st:
            src = st.get("source", "json")
            if st.get("running"):
                # Q4：被硬杀留下的孤儿行（status=running，无 finished_at）
                print(f"上次同步: ⚠ 中途未结束（running，始于 {st['started_at']}）"
                      f"—— 可能进程被强杀，建议重跑 sync")
            else:
                print(f"上次同步: {st['finished_at']} 耗时 {st['duration_s']}s"
                      f"（来源: {src}）")
                bad = [k for k, v in st.get('steps', {}).items() if v]
                print(f"  各环节: {st.get('steps', {})}")
                print(f"  异常环节: {bad or '无'} / 覆盖: "
                      f"{'正常' if st.get('coverage') == 0 else '有缺口'}")
                print(f"  状态: {st.get('status')} "
                      f"库内 listed={st.get('listed')} indexed={st.get('indexed')}")
                if st.get("error_msg"):
                    print(f"  错误: {st['error_msg']}")
        else:
            print("上次同步: 无记录（尚未跑过 sync 或状态文件缺失）")
        # P1-1：把未读告警带出来，让"失败可观测"闭环无需打开 Web
        try:
            from . import notify as _n
            un = _n.unacked_count(cfg)
            if un:
                print(f"\n⚠️  未读告警 {un} 条（最近 3 条）：")
                for a in _n.recent(cfg, 3, only_unacked=True):
                    print(f"  [{a['ts']}] {a['title']} —— {a['message']}")
                print("  查看全部: python -m xhs_rag.cli alerts list"
                      "  标记已读: python -m xhs_rag.cli alerts ack")
        except Exception:
            pass
        return 0

    if action == "run":
        # 两条路径：任务已注册 → schtasks /Run 走系统调度链路（真实验证）；
        # 未注册 → 直接前台跑 cmd_sync（验证流水线本身）。
        if sched.exists():
            rc, out = sched.trigger_once()
            print(f"已触发 {sched.TASK_NAME} 立即运行"
                  f"{'，退出码 ' + str(rc) if rc else ''}")
            if out:
                print(out)
            if rc:
                return rc
            print("后台运行中，几分钟后用 status 查看结果"
                  "（看 data/sync_state.json 的 finished_at 是否更新）")
            return 0
        print("任务未注册，前台直接跑 sync ...")
        return cmd_sync(cfg)

    print(f"未知 schedule 动作: {action}")
    return 1


# ── alerts（P1-1 失败告警）───────────────────────────────
def cmd_alerts(cfg: Config, action: str, limit: int = 20) -> int:
    """list / ack。告警来源：同步失败、登录态失效、索引覆盖缺口。"""
    from . import notify as _n

    if action == "list":
        items = _n.recent(cfg, limit)
        if not items:
            print("无告警记录（好事）")
            return 0
        un = _n.unacked_count(cfg)
        print(f"===== 告警 {len(items)} 条（全库未读 {un} 条）=====")
        for a in items:
            mark = "●" if not a.get("ack") else "○"
            lvl = "ERR " if a.get("level") == "error" else "WARN"
            print(f"{mark} [{a['ts']}] {lvl} {a['title']} —— {a['message']}")
        return 0

    if action == "ack":
        n = _n.ack_all(cfg)
        print(f"已标记 {n} 条告警为已读")
        return 0

    print(f"未知 alerts 动作: {action}（可用: list / ack）")
    return 1


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
    p_search = sub.add_parser("search", help="语义检索（M5，--image 可传截图）")
    p_search.add_argument("query", nargs="?", default="",
                          help="检索关键词/问题（用 --image 时可省略）")
    p_search.add_argument("-k", type=int, default=5, help="返回条数")
    p_search.add_argument("--image", help="截图路径：OCR 成检索词后合并检索（查询端多模态）")
    p_ask = sub.add_parser("ask", help="检索 + LLM 带引用回答（M8，--image 可传截图）")
    p_ask.add_argument("query", nargs="?", default="",
                       help="问题（用 --image 时可省略）")
    p_ask.add_argument("-k", type=int, default=5, help="喂给 LLM 的片段数")
    p_ask.add_argument("--image", help="截图路径：OCR 后与问题合并（查询端多模态）")
    p_sync = sub.add_parser("sync", help="一键全量同步（M7：collect→detail→ocr→video→index）")
    p_sync.add_argument("--headless", action="store_true", help="无头模式（定时任务用，风控更严）")
    p_sync.add_argument("--max-pages", type=int, default=50, help="收藏列表翻页上限")
    p_sync.add_argument("--skip-media", action="store_true", help="跳过媒体下载，只抓详情")
    p_setup = sub.add_parser("setup", help="一键建库：登录检查(必要时扫码) → 全流水线 → (可选)启动 Web UI")
    p_setup.add_argument("--login-timeout", type=int, default=180, help="扫码等待秒数")
    p_setup.add_argument("--headless", action="store_true", help="同步用无头模式（不推荐）")
    p_setup.add_argument("--max-pages", type=int, default=50, help="收藏列表翻页上限")
    p_setup.add_argument("--skip-media", action="store_true", help="跳过媒体下载，只抓详情")
    p_setup.add_argument("--serve", action="store_true", help="同步完直接启动 Web UI（阻塞）")
    sub.add_parser("serve", help="启动 Web UI（M6，手机可访问）")
    sub.add_parser("mcp", help="启动 MCP server（stdio，供 WorkBuddy 等 AI 客户端接入收藏夹问答）")

    p_agent = sub.add_parser("agent", help="Agent 模式（M10）：LLM 自主多步调工具后作答，适合对比/清单/多主题问题")
    p_agent.add_argument("query", help="问题")
    p_agent.add_argument("--max-steps", type=int, default=8, help="工具调用步数上限")

    p_mem = sub.add_parser("memory", help="长期记忆（M11）：digest 对话 / 查看 / 清空")
    p_mem_sub = p_mem.add_subparsers(dest="mem_action", required=True)
    pm_d = p_mem_sub.add_parser("digest", help="消化未处理对话 → 摘要+用户画像")
    pm_d.add_argument("--max-blocks", type=int, default=6, help="本次最多处理块数")
    p_mem_sub.add_parser("show", help="查看已存摘要与画像")
    pm_c = p_mem_sub.add_parser("clear", help="清空全部记忆（不可撤销）")
    pm_c.add_argument("--yes", action="store_true", help="跳过确认直接清空")

    p_sched = sub.add_parser("schedule", help="定时调度（M7）：注册/卸载/状态/立即触发 Windows 计划任务")
    p_sched_sub = p_sched.add_subparsers(dest="sched_action", required=True)
    p_sched_i = p_sched_sub.add_parser("install", help="注册每日计划任务")
    p_sched_i.add_argument("--at", default="09:00", help="每日触发时间 HH:MM（默认 09:00）")
    p_sched_sub.add_parser("uninstall", help="删除计划任务")
    p_sched_sub.add_parser("status", help="查看任务注册状态与上次同步结果")
    p_sched_sub.add_parser("run", help="立即触发一次（已注册走系统调度，未注册前台直跑）")

    p_alerts = sub.add_parser("alerts", help="失败告警（P1-1）：list 查看 / ack 全部标记已读")
    p_alerts_sub = p_alerts.add_subparsers(dest="alerts_action", required=True)
    p_alerts_l = p_alerts_sub.add_parser("list", help="列出最近告警")
    p_alerts_l.add_argument("--limit", type=int, default=20, help="最多显示条数")
    p_alerts_sub.add_parser("ack", help="全部标记已读")

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
        return cmd_search(cfg, args.query, k=args.k, image=args.image)
    if args.cmd == "ask":
        return cmd_ask(cfg, args.query, k=args.k, image=args.image)
    if args.cmd == "agent":
        return cmd_agent(cfg, args.query, max_steps=args.max_steps)
    if args.cmd == "sync":
        return cmd_sync(cfg, headless=args.headless,
                        max_pages=args.max_pages, skip_media=args.skip_media)
    if args.cmd == "setup":
        return cmd_setup(cfg, login_timeout=args.login_timeout,
                         headless=args.headless, max_pages=args.max_pages,
                         skip_media=args.skip_media, serve=args.serve)
    if args.cmd == "serve":
        return cmd_serve(cfg)
    if args.cmd == "memory":
        return cmd_memory(cfg, args.mem_action,
                          max_blocks=args.max_blocks
                          if args.mem_action == "digest" else 6,
                          yes=getattr(args, "yes", False))
    if args.cmd == "schedule":
        return cmd_schedule(cfg, args.sched_action, at=getattr(args, "at", "09:00"))
    if args.cmd == "alerts":
        return cmd_alerts(cfg, args.alerts_action,
                          limit=getattr(args, "limit", 20))
    if args.cmd == "mcp":
        from .mcp_server import main as mcp_main

        return mcp_main()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
