"""M4 视频处理 —— 抽帧 OCR + ASR 语音转写。

流程(每一条视频笔记):
  1. ffmpeg 抽关键帧(按 config 的 frame_mode,默认 ikey 关键帧)
  2. 抽帧图走 RapidOCR 识别画面文字(复用 M3 的 OcrEngine)
  3. ffmpeg 抽 16k mono 音频 → 硅基流动 Qwen3-ASR-1.7B 转写
  4. 结果写回 videos 表(frame_count / asr_text / asr_status)并更新 vault Markdown

断点续传:videos.asr_status='done' 即视为完成,重跑自动跳过。
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from loguru import logger

from ..core.config import Config
from ..store.db import DB
from .ocr import OcrEngine


class VideoProcessor:
    def __init__(self, cfg: Config, db: DB):
        self.cfg = cfg
        self.db = db
        self.engine = OcrEngine(cfg)
        self.tmp_root = cfg.path("paths.data_dir") / "tmp"
        self.vault_dir = cfg.path("paths.vault_dir")
        self.frame_dir = cfg.path("paths.data_dir") / "tmp" / "frames"
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        self.ffmpeg = self._find_ffmpeg()

    def _find_ffmpeg(self) -> str:
        configured = self.cfg.get("video.ffmpeg_path", "")
        if configured and Path(configured).exists():
            return str(configured)
        try:
            import imageio_ffmpeg

            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            raise RuntimeError(
                "找不到 ffmpeg:请配置 video.ffmpeg_path 或 pip install imageio-ffmpeg"
            )

    # ── 抽帧 ─────────────────────────────────────────────
    def extract_frames(self, video_path: Path, note_id: str) -> list[Path]:
        """抽关键帧到 data/tmp/frames/{note_id}/,返回帧图路径列表。

        默认 ikey(仅关键帧):信息密度高、数量少,适合画面文字 OCR;
        m4v 内置 scene 检测在 940M 老卡上不可靠,不用。
        """
        mode = self.cfg.get("video.frame_mode", "ikey")
        out_dir = self.frame_dir / note_id
        out_dir.mkdir(parents=True, exist_ok=True)
        # 清掉上次的帧(断点重试时避免旧帧残留)
        for old in out_dir.glob("*.jpg"):
            old.unlink()

        scale_w = int(self.cfg.get("video.frame_scale_width", 640))
        quality = int(self.cfg.get("video.frame_quality", 4))
        pattern = str(out_dir / "frame_%04d.jpg")

        args = [self.ffmpeg, "-y", "-i", str(video_path)]
        if mode == "ikey":
            args += ["-vf", f"select='eq(pict_type,I)',scale={scale_w}:-2",
                     "-vsync", "vfr", "-q:v", str(quality)]
        else:  # 按固定间隔抽帧(兜底)
            interval = float(self.cfg.get("video.frame_interval", 2.0))
            args += ["-vf", f"fps=1/{interval},scale={scale_w}:-2",
                     "-q:v", str(quality)]
        args.append(pattern)

        r = subprocess.run(args, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            logger.warning(f"抽帧失败 [{note_id}]: {r.stderr[-200:]}")
            return []
        frames = sorted(out_dir.glob("*.jpg"))
        logger.info(f"[{note_id}] 抽帧 {len(frames)} 张 (mode={mode})")
        return frames

    # ── 音频 + ASR ───────────────────────────────────────
    def extract_audio(self, video_path: Path, note_id: str) -> Path | None:
        """抽 16k mono wav,返回路径。"""
        wav = self.tmp_root / "audio" / f"{note_id}.wav"
        wav.parent.mkdir(parents=True, exist_ok=True)
        if wav.exists() and wav.stat().st_size > 0:
            return wav
        r = subprocess.run(
            [self.ffmpeg, "-y", "-i", str(video_path),
             "-ar", "16000", "-ac", "1", str(wav)],
            capture_output=True, text=True, timeout=600,
        )
        if r.returncode != 0 or not wav.exists():
            logger.warning(f"抽音频失败 [{note_id}]: {r.stderr[-200:]}")
            return None
        return wav

    def transcribe(self, wav: Path, note_id: str) -> str | None:
        """ASR 转写:本地 SenseVoiceSmall 优先,失败回退云端 API。

        ★ 本地引擎懒加载(首次调用初始化,复用实例),模型缓存到
          data/models/sensevoice —— 免费无限量,不依赖 API 余额。
        """
        # 1) 本地 SenseVoiceSmall
        text = self._transcribe_local(wav)
        if text:
            return text

        # 2) 云端 API 兜底(账户有余额时可用)
        return self._transcribe_api(wav, note_id)

    def _transcribe_local(self, wav: Path) -> str | None:
        """funasr SenseVoiceSmall 本地推理。"""
        if not getattr(self, "_asr_model", None):
            try:
                from funasr import AutoModel

                model_dir = self._ensure_sensevoice_model()
                self._asr_model = AutoModel(
                    model=model_dir,
                    vad_model="fsmn-vad",
                    disable_update=True,
                    device="cpu",
                )
                logger.info("本地 SenseVoiceSmall ASR 加载完成")
            except Exception as e:
                logger.warning(f"本地 ASR 不可用: {e}")
                self._asr_model = False  # 缓存失败
                return None

        if self._asr_model is False:
            return None
        try:
            res = self._asr_model.generate(input=str(wav), language="auto",
                                           use_itn=True)
            if res and res[0].get("text"):
                return self._clean_sensevoice(res[0]["text"])
        except Exception as e:
            logger.warning(f"本地 ASR 推理失败: {e}")
        return None

    @staticmethod
    def _clean_sensevoice(text: str) -> str:
        """去掉 SenseVoice 输出的 <|lang|><|emotion|><|BGM|> 等标签。"""
        import re

        cleaned = re.sub(r"<\|[^|]+\|>", "", text)
        return cleaned.strip()

    def _ensure_sensevoice_model(self) -> str:
        """下载 SenseVoiceSmall 到 data/models/sensevoice(幂等)。"""
        target = self.cfg.path("paths.data_dir") / "models" / "sensevoice"
        if (target / "model.pt").exists() or (target / "model.onnx").exists():
            return str(target)
        import modelscope

        target.mkdir(parents=True, exist_ok=True)
        logger.info("首次使用,下载 SenseVoiceSmall 模型(~900MB,仅一次)")
        modelscope.snapshot_download(
            "iic/SenseVoiceSmall", local_dir=str(target))
        return str(target)

    def _transcribe_api(self, wav: Path, note_id: str) -> str | None:
        """硅基流动 Qwen3-ASR-1.7B 转写(兜底,需要账户余额)。"""
        provider = self.cfg.get("video.asr.provider", "siliconflow")
        model = self.cfg.get("video.asr.model", "Qwen/Qwen3-ASR-1.7B")
        env_name = self.cfg.get(f"providers.{provider}.api_key_env",
                                "SILICONFLOW_API_KEY")
        base_url = self.cfg.get(f"providers.{provider}.base_url",
                                "https://api.siliconflow.cn/v1")
        key = os.environ.get(env_name, "").strip()
        if not key:
            return None

        try:
            import requests

            with wav.open("rb") as f:
                resp = requests.post(
                    f"{base_url.rstrip('/')}/audio/transcriptions",
                    headers={"Authorization": f"Bearer {key}"},
                    files={"file": (f"{note_id}.wav", f, "audio/wav")},
                    data={"model": model, "language": "zh",
                          "response_format": "json"},
                    timeout=600,
                )
            resp.raise_for_status()
            data = resp.json()
            text = (data.get("text") or "").strip()
            return text or None
        except Exception as e:
            logger.warning(f"云端 ASR 失败 [{note_id}]: {type(e).__name__}: {e}")
            return None

    # ── 主流程 ───────────────────────────────────────────
    def process_note(self, note_id: str) -> dict:
        video = self.db.get_video(note_id)
        note = self.db.get_note(note_id)
        if not video or not note:
            return {"frame": 0, "asr": False, "skip": True}

        if video.get("asr_status") == "done" and video.get("frame_count", 0) > 0:
            return {"frame": 0, "asr": False, "skip": True}

        vpath = self.tmp_root / "video" / f"{note_id}.mp4"
        if not vpath.exists():
            logger.warning(f"视频文件不存在 [{note_id}],跳过")
            return {"frame": 0, "asr": False, "skip": True}

        result = {"frame": 0, "asr": False, "skip": False}

        # 1) 抽帧 + 帧 OCR
        frames = self.extract_frames(vpath, note_id)
        frame_ocr: list[dict] = []
        for i, fp in enumerate(frames):
            r = self.engine.ocr_image(fp)
            if r and r.get("text"):
                frame_ocr.append({"seq": i, "text": r["text"],
                                  "confidence": r["confidence"],
                                  "engine": r["engine"]})
        result["frame"] = len(frame_ocr)

        # 2) 抽音频 + ASR
        wav = self.extract_audio(vpath, note_id)
        if wav:
            text = self.transcribe(wav, note_id)
            if text:
                result["asr"] = True
                self.db.conn.execute(
                    """UPDATE videos SET frame_count=?, frame_mode=?,
                       asr_status='done', asr_text=?, processed_at=? WHERE note_id=?""",
                    (len(frames), self.cfg.get("video.frame_mode", "ikey"),
                     text, int(time.time() * 1000), note_id),
                )
                self.db.conn.commit()
                # 更新 vault Markdown(补上 ASR 文本)
                self._update_markdown(note, text, frame_ocr)
                return result
            else:
                # ASR 失败但帧 OCR 成功:帧信息先落库
                self.db.conn.execute(
                    """UPDATE videos SET frame_count=?, frame_mode=?,
                       asr_status='failed', processed_at=? WHERE note_id=?""",
                    (len(frames), self.cfg.get("video.frame_mode", "ikey"),
                     int(time.time() * 1000), note_id),
                )
                self.db.conn.commit()
                self._update_markdown(note, "", frame_ocr)
        return result

    def _update_markdown(self, note: dict, asr_text: str,
                         frame_ocr: list[dict]) -> None:
        """把 ASR 文本和帧 OCR 写入 vault Markdown。"""
        nid = note["note_id"]
        md = self.vault_dir / f"{nid}.md"
        if not md.exists():
            return
        content = md.read_text(encoding="utf-8")

        # 替换 ASR 占位或追加
        if asr_text:
            block = f"## 视频语音转写\n\n{asr_text}"
            if "## 视频语音转写" in content:
                import re

                content = re.sub(
                    r"## 视频语音转写\n\n.*?(?=\n## |\Z)",
                    block + "\n\n", content, flags=re.S)
            else:
                content += f"\n\n{block}\n"

        # 帧 OCR(画面文字)
        if frame_ocr:
            lines = ["## 视频画面文字(帧 OCR)", ""]
            for fr in frame_ocr:
                lines.append(f"- {fr['text']}")
            lines.append("")
            if "## 视频画面文字" in content:
                import re

                content = re.sub(
                    r"## 视频画面文字\(帧 OCR\)\n\n.*?(?=\n## |\Z)",
                    "\n".join(lines) + "\n", content, flags=re.S)
            else:
                content += "\n" + "\n".join(lines) + "\n"

        md.write_text(content, encoding="utf-8")

    def run(self) -> dict:
        """处理所有有视频且 ASR 未完成的笔记。"""
        vids = self.db.conn.execute(
            "SELECT note_id FROM videos WHERE asr_status != 'done' OR asr_status IS NULL"
        ).fetchall()
        stats = {"total": len(vids), "asr_ok": 0, "frame_only": 0, "failed": 0, "skip": 0}
        for i, row in enumerate(vids, 1):
            nid = row["note_id"]
            logger.info(f"[{i}/{len(vids)}] 视频处理 {nid}")
            try:
                r = self.process_note(nid)
                if r.get("skip"):
                    stats["skip"] += 1
                elif r.get("asr"):
                    stats["asr_ok"] += 1
                elif r.get("frame"):
                    stats["frame_only"] += 1
                else:
                    stats["failed"] += 1
            except Exception as e:
                logger.exception(f"视频处理异常 {nid}: {e}")
                stats["failed"] += 1

        logger.success(
            f"M4 完成: 共 {stats['total']} 条, ASR 成功 {stats['asr_ok']}, "
            f"仅帧 OCR {stats['frame_only']}, 失败 {stats['failed']}, 跳过 {stats['skip']}"
        )
        return stats
