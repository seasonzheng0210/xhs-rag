"""M5 切分 —— 把 vault/ 下的 Markdown 切成可检索的 chunk。

策略(与 config.chunk 对应):
  - 按 ## 二级标题切 section,保留标题作为上下文
  - section 字数 <= whole_threshold(800) 时整体作为一个 chunk
  - 超过的用滑窗(size=800, overlap=100)二次切分
  - 输出: [{note_id, title, section, text, seq}]
"""
from __future__ import annotations

import re
from pathlib import Path

from ..core.config import Config

SECTION_RE = re.compile(r"^##\s+(.+)$", re.M)
# 跳过代码块/图片引用等非正文内容
NOISE_RE = re.compile(r"!\[.*?\]\(.*?\)|<!--.*?-->", re.S)
# 视频帧 OCR 的定位标记,如 <|LOC_0|><|LOC_274|>,纯噪声
LOC_TAG_RE = re.compile(r"<\|LOC_\d+(?:\|LOC_\d+)*\|>")
# 纯 emoji/装饰符号行(无字母数字中文),如 🌻🌻🌻
# 注意: 不能是 raw string(否则 \u2600 变字面量); \uXXXX 只认 4 位,
# emoji 区(U+1F000+)必须用 8 位 \U0001F000 写法
EMOJI_LINE_RE = re.compile("^[\\s\\u2600-\\u27BF\\u2B00-\\u2BFF\\U0001F000-\\U0001FAFF\\uFE0F]+$")
# 无意义 ASCII 乱码行: 含字母、无中文、无数字、长度>=8(如 dgegbspb / aceae)
# 数字数据行(3.18±0.43)不含字母,不会被命中,安全保留
GARBAGE_ASCII_LINE_RE = re.compile(r"^[A-Za-z\s\-.()/+&%#@!?=\[\]{}|'\"`~^:;,<>]+$")


def clean_ocr_noise(text: str) -> str:
    """清洗 OCR 噪声(保守策略,只删无争议噪声,不误删正文)。

    1. 删除 <|LOC_n|> 帧定位标记(视频 OCR 的定位符,纯噪声)
    2. 删除纯 emoji/装饰符号行(🌻🌻🌻 之类)
    3. 删除无意义 ASCII 乱码行(纯字母无数字无中文,如 aceae)
    保留: 数字数据行(3.18±0.43)、中英混合行(可能含有效信息)
    """
    text = LOC_TAG_RE.sub("", text)
    out: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            out.append("")
        elif EMOJI_LINE_RE.match(s):
            continue  # 纯 emoji/符号装饰行
        elif (len(s) >= 8 and GARBAGE_ASCII_LINE_RE.match(s)
              and not any(c.isdigit() for c in s)):
            continue  # 纯字母乱码串(无数字无中文),如 aceae
        else:
            out.append(line)
    return "\n".join(out)


class Chunker:
    def __init__(self, cfg: Config):
        self.size = int(cfg.get("chunk.size", 800))
        self.overlap = int(cfg.get("chunk.overlap", 100))
        self.whole_threshold = int(cfg.get("chunk.whole_threshold", 800))
        self.strategy = cfg.get("chunk.strategy", "auto")

    def split_md(self, md_path: Path) -> list[dict]:
        """把一个 Markdown 文件切成 chunk 列表。"""
        content = md_path.read_text(encoding="utf-8")
        note_id = md_path.stem

        # 标题行(# 开头)作为 chunk 的 title 上下文
        title = ""
        title_m = re.match(r"^#\s+(.+)$", content, re.M)
        if title_m:
            title = title_m.group(1).strip()

        # 去掉图片引用和 HTML 注释(OCR 文本不在这里,那是图片替代文本)
        clean = NOISE_RE.sub("", content)
        # OCR 噪声清洗(LOC 标记/emoji 装饰行/纯字母乱码行),见 clean_ocr_noise
        clean = clean_ocr_noise(clean)
        # 去掉 # 标题行(标题单独存字段)和 > 引用块(作者/链接等元信息)
        lines = []
        for line in clean.splitlines():
            s = line.strip()
            if not s:
                lines.append("")
            elif s.startswith(">"):
                continue
            else:
                lines.append(line)
        clean = "\n".join(lines)

        sections = self._split_sections(clean)
        chunks: list[dict] = []
        for sec_title, sec_body in sections:
            sec_text = sec_title + "\n" + sec_body if sec_title else sec_body
            for piece in self._split_long(sec_text):
                pieces = piece.strip().split("\n")
                chunks.append({
                    "note_id": note_id,
                    "title": title,
                    "section": sec_title or "",
                    "text": piece.strip(),
                    "seq": len(chunks),
                })
        if not chunks:  # 空文件兜底
            chunks.append({
                "note_id": note_id, "title": title, "section": "",
                "text": clean.strip() or title, "seq": 0,
            })
        return chunks

    def _split_sections(self, content: str) -> list[tuple[str, str]]:
        """按 ## 标题切分,返回 [(标题, 正文)] 列表。"""
        matches = list(SECTION_RE.finditer(content))
        if not matches:
            return [("", content)]
        sections: list[tuple[str, str]] = []
        for i, m in enumerate(matches):
            body_start = m.end()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            sections.append((m.group(1).strip(), content[body_start:body_end]))
        # ## 标题前的引言(# 标题 + 描述)归入第一个 section
        head = content[: matches[0].start()].strip()
        if head and sections:
            t, b = sections[0]
            sections[0] = (t, head + "\n" + b)
        return sections

    def _split_long(self, text: str) -> list[str]:
        """超长文本滑窗切分。"""
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.whole_threshold:
            return [text]
        pieces: list[str] = []
        start = 0
        n = len(text)
        step = self.size - self.overlap
        while start < n:
            piece = text[start: start + self.size]
            if len(piece) < self.overlap * 2 and pieces:  # 尾部太短并入上一块
                pieces[-1] += "\n" + piece
                break
            pieces.append(piece)
            start += step
        return pieces
