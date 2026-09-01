"""M3 OCR 处理 —— 图片 → 文本,落回数据库。

hybrid 策略(用户已确认):
  1. 本地 RapidOCR(onnxruntime CPU,零成本)跑第一遍
  2. 命中 escalate 条件(低置信度 / 文本过短 / 疑似表格)才调
     硅基流动 PaddleOCR-VL API 兜底(免费额度)

断点续传:images.ocr_done 置 1 即视为完成,重跑自动跳过。
"""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any

from loguru import logger

from ..core.config import Config, ROOT
from ..store.db import DB


class OcrEngine:
    """OCR 引擎封装:本地 RapidOCR + 云端 VL 兜底,按需懒加载。"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._rapid: Any | None = None
        self._started_at = time.time()
        self._api_calls = 0

    # ── 本地 RapidOCR ────────────────────────────────────
    def _load_rapid(self) -> Any | None:
        if self._rapid is not None:
            return self._rapid
        try:
            from rapidocr_onnxruntime import RapidOCR

            self._rapid = RapidOCR()
            logger.info("RapidOCR 本地引擎加载完成(CPU)")
            return self._rapid
        except Exception as e:
            logger.warning(f"RapidOCR 加载失败,降级为纯 API 模式: {e}")
            self._rapid = False  # 缓存失败状态,避免反复尝试
            return None

    def ocr_local(self, img_path: Path) -> dict | None:
        """本地识别,返回 {text, confidence} 或 None(失败/无文本)。

        RapidOCR 返回格式: ([[box, text, score], ...], elapse) 或
        新版直接返回 [[box, text, score], ...]。
        """
        engine = self._load_rapid()
        if not engine:
            return None
        try:
            result = engine(str(img_path))
            # 兼容两种返回: (result, elapse) 元组 / 直接 result
            if isinstance(result, tuple):
                result = result[0]
            if not result:
                return None
            lines, confs = [], []
            for item in result:
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    text = str(item[1]).strip()
                    if text:
                        lines.append(text)
                        try:
                            confs.append(float(item[2]))
                        except (TypeError, ValueError):
                            pass
            if not lines:
                return None
            conf = sum(confs) / len(confs) if confs else 0.0
            return {"text": "\n".join(lines), "confidence": conf,
                    "engine": "rapidocr"}
        except Exception as e:
            logger.debug(f"RapidOCR 识别异常 {img_path.name}: {e}")
            return None

    # ── 云端 VL 兜底 ─────────────────────────────────────
    def ocr_vlm(self, img_path: Path, timeout: int = 60) -> dict | None:
        """硅基流动 PaddleOCR-VL 图片转文本。失败返回 None。"""
        provider = self.cfg.get("ocr.api.provider", "siliconflow")
        model = self.cfg.get("ocr.api.model", "PaddlePaddle/PaddleOCR-VL-1.5")
        env_name = self.cfg.get(f"providers.{provider}.api_key_env",
                                "SILICONFLOW_API_KEY")
        base_url = self.cfg.get(f"providers.{provider}.base_url",
                                "https://api.siliconflow.cn/v1")
        import os

        key = os.environ.get(env_name, "").strip()
        if not key:
            logger.warning(f"缺少 {env_name},跳过 VLM 兜底")
            return None

        try:
            import requests

            b64 = base64.b64encode(img_path.read_bytes()).decode()
            ext = img_path.suffix.lstrip(".") or "jpg"
            mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
            payload = {
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:{mime};base64,{b64}"}},
                        {"type": "text", "text":
                         "请完整识别这张图片中的所有文字,按阅读顺序输出。"
                         "纯图片无文字则输出 <NO_TEXT>。只输出识别结果,不要解释。"},
                    ],
                }],
                "max_tokens": 1000,
            }
            resp = requests.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"},
                json=payload, timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            self._api_calls += 1
            text = (data["choices"][0]["message"]["content"] or "").strip()
            if not text or "<NO_TEXT>" in text:
                return None
            return {"text": text, "confidence": 1.0, "engine": f"vlm:{model.split('/')[-1]}"}
        except Exception as e:
            logger.debug(f"VLM OCR 异常 {img_path.name}: {e}")
            return None

    # ── 主流程 ───────────────────────────────────────────
    def ocr_image(self, img_path: Path) -> dict | None:
        """hybrid:本地优先,命中 escalate 条件再上云。返回 {text, confidence, engine}。"""
        cfg = self.cfg
        local = self.ocr_local(img_path)

        min_conf = float(cfg.get("ocr.min_confidence", 0.5))
        escalate_conf = float(cfg.get("ocr.api.escalate_on.low_confidence", 0.6))
        escalate_short = int(cfg.get("ocr.api.escalate_on.too_short", 10))

        if local:
            text = local["text"]
            conf = local["confidence"]
            too_short = len(text.strip()) < escalate_short
            low_conf = conf < escalate_conf
            if not (too_short or low_conf):
                return local
            # 本地结果不达标 → 云兜底
            logger.debug(f"{img_path.name}: 本地 conf={conf:.2f} len={len(text)},上云")
            vlm = self.ocr_vlm(img_path)
            if vlm:
                return vlm
            # 云也失败:本地结果可用则保留(避免丢内容)
            if conf >= min_conf or not too_short:
                return local
            return None

        # 本地完全没结果 → 上云
        return self.ocr_vlm(img_path)


class OcrProcessor:
    """M3 主流程:扫 data/tmp/{note_id}/ 的图片 → OCR → 写库 → 状态推进。"""

    def __init__(self, cfg: Config, db: DB):
        self.cfg = cfg
        self.db = db
        self.engine = OcrEngine(cfg)
        self.tmp_root = cfg.path("paths.data_dir") / "tmp"
        self.thumbs_dir = cfg.path("paths.data_dir") / "thumbs"
        self.vault_dir = cfg.path("paths.vault_dir")
        self.vault_dir.mkdir(parents=True, exist_ok=True)
        self.thumbs_dir.mkdir(parents=True, exist_ok=True)

    def _thumbnail(self, src: Path, note_id: str, seq: int) -> Path | None:
        """生成 320px 缩略图到 data/thumbs/{note_id}/{seq:02d}.jpg（用户决策：不留原图）。"""
        dest_dir = self.thumbs_dir / note_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{seq:02d}.jpg"
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        try:
            from PIL import Image

            width = int(self.cfg.get("sync.thumbnail_width", 320))
            img = Image.open(src)
            img = img.convert("RGB")
            if img.width > width:
                h = round(img.height * width / img.width)
                img = img.resize((width, h), Image.LANCZOS)
            img.save(dest, "JPEG", quality=85)
            return dest
        except Exception as e:
            logger.debug(f"缩略图生成失败 {src.name}: {e}")
            return None

    def process_note(self, note: dict) -> dict:
        """处理单条笔记的图片。返回 {imgs, ok, skip}。"""
        nid = note["note_id"]
        img_dir = self.tmp_root / nid
        if not img_dir.exists():
            return {"imgs": 0, "ok": 0, "skip": 0}

        imgs = self.db.images_of(nid)
        if not imgs:
            # 没登记 images 表但目录有文件(异常态),按文件名排序补录
            files = sorted(img_dir.glob("*"))
            if not files:
                return {"imgs": 0, "ok": 0, "skip": 0}
            imgs = [{"note_id": nid, "seq": i, "url": "", "local_path": str(f)}
                    for i, f in enumerate(files)]

        ok = skip = 0
        for im in imgs:
            if im.get("ocr_done"):
                skip += 1
                continue
            path = Path(im.get("local_path") or "") if im.get("local_path") else None
            if not path or not path.exists():
                path = img_dir / f"{im['seq']:02d}.jpg"
            if not path.exists():
                # 兼容 webp/avif 后缀
                cands = list(img_dir.glob(f"{im['seq']:02d}.*"))
                path = cands[0] if cands else None
            if not path:
                continue

            result = self.engine.ocr_image(path)
            if result:
                self.db.conn.execute(
                    """UPDATE images SET ocr_text=?, ocr_confidence=?,
                       ocr_engine=?, ocr_done=1, local_path=? WHERE note_id=? AND seq=?""",
                    (result["text"], result["confidence"], result["engine"],
                     str(path), nid, im["seq"]),
                )
                ok += 1
            else:
                # 识别不出:标记完成(空文本),避免无限重试
                self.db.conn.execute(
                    """UPDATE images SET ocr_text='', ocr_done=1,
                       ocr_engine='none', local_path=? WHERE note_id=? AND seq=?""",
                    (str(path), nid, im["seq"]),
                )
        self.db.conn.commit()
        return {"imgs": len(imgs), "ok": ok, "skip": skip}

    def build_markdown(self, note: dict) -> Path | None:
        """生成笔记的 Markdown 落盘到 vault/,返回路径。

        图片引用缩略图(data/thumbs/{note_id}/{seq:02d}.jpg),原图 OCR 后即删。
        """
        nid = note["note_id"]
        imgs = self.db.images_of(nid)
        md_path = self.vault_dir / f"{nid}.md"

        lines: list[str] = []
        lines.append(f"# {note.get('title') or '无标题'}")
        lines.append("")
        lines.append(f"> 作者: {note.get('author_name') or ''}  "
                     f"| 类型: {'视频' if note.get('note_type') == 'video' else '图文'}")
        lines.append(f"> 链接: {note.get('url') or ''}")
        lines.append("")
        desc = (note.get("desc") or "").strip()
        if desc:
            lines.append(desc)
            lines.append("")

        # 正文(desc 通常就是正文,小红书详情页 desc 即完整文案)
        has_ocr = False
        for im in imgs:
            text = (im.get("ocr_text") or "").strip()
            if not text:
                continue
            has_ocr = True
            # 缩略图优先;没有则用原图路径(未清理的旧数据)
            thumb = self.thumbs_dir / nid / f"{im['seq']:02d}.jpg"
            ref = thumb if thumb.exists() else (im.get("local_path") or "")
            lines.append(f"![图{im['seq'] + 1}]({ref})")
            lines.append("")
            lines.append(f"<!-- 图{im['seq'] + 1} OCR -->")
            lines.append(text)
            lines.append("")

        if not has_ocr:
            lines.append("<!-- 图片未识别出文字 -->")

        # 视频 ASR 占位(M4 回填)
        video = self.db.get_video(nid)
        if video:
            if video.get("asr_text"):
                lines.append("## 视频语音转写")
                lines.append("")
                lines.append(video["asr_text"])
            else:
                lines.append("<!-- 视频 ASR:待 M4 处理 -->")

        md_path.write_text("\n".join(lines), encoding="utf-8")
        return md_path

    def cleanup_originals(self, note_id: str) -> None:
        """用户决策：OCR 后删原图，只留 320px 缩略图。"""
        img_dir = self.tmp_root / note_id
        if img_dir.exists():
            import shutil

            shutil.rmtree(img_dir, ignore_errors=True)
            logger.debug(f"原图已清理: {img_dir}")

    def run(self, limit: int | None = None) -> dict:
        """处理所有 media_done 的笔记。断点续传:已 ocr_done 的跳过。"""
        notes = self.db.notes_by_status("media_done", limit=limit or 1000)
        logger.info(f"M3 OCR 待处理 {len(notes)} 条笔记")
        stats = {"notes": len(notes), "ok": 0, "skip": 0, "failed": 0, "vlm": 0}
        for i, note in enumerate(notes, 1):
            nid = note["note_id"]
            # 已生成 md 且图片全部 ocr_done → 视为完成
            md = self.vault_dir / f"{nid}.md"
            imgs = self.db.images_of(nid)
            all_done = all(im.get("ocr_done") for im in imgs) if imgs else False
            if all_done and md.exists():
                stats["skip"] += 1
                self.db.set_status(nid, "ocr_done")
                continue

            r = self.process_note(note)
            stats["ok"] += r["ok"]
            stats["skip"] += r["skip"]
            if r["imgs"] and r["ok"] == 0 and r["skip"] == 0:
                stats["failed"] += 1
            if self.engine._api_calls:
                stats["vlm"] = self.engine._api_calls

            # 生成缩略图 + 清理原图(用户决策:不留原图)
            img_dir = self.tmp_root / nid
            if img_dir.exists():
                imgs = self.db.images_of(nid)
                for im in imgs:
                    src = img_dir / f"{im['seq']:02d}.jpg"
                    if not src.exists():
                        cands = list(img_dir.glob(f"{im['seq']:02d}.*"))
                        src = cands[0] if cands else None
                    if src:
                        self._thumbnail(src, nid, im["seq"])
                self.cleanup_originals(nid)

            # 生成 Markdown
            try:
                self.build_markdown(note)
                self.db.set_status(nid, "ocr_done")
            except Exception as e:
                logger.exception(f"Markdown 生成失败 {nid}: {e}")
                self.db.set_status(nid, "media_done", f"md_exc: {e}")

            if i % 5 == 0:
                logger.info(f"[{i}/{len(notes)}] 进度 {stats['ok']} 图 OK, "
                            f"{stats['vlm']} 次 VLM")

        stats["vlm"] = self.engine._api_calls
        logger.success(
            f"M3 完成: {stats['notes']} 条笔记, {stats['ok']} 图新增 OCR, "
            f"{stats['skip']} 图跳过, VLM 兜底 {stats['vlm']} 次"
        )
        return stats
