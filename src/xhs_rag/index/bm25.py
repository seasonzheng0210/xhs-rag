"""BM25 全文检索 —— 与 bge-m3 向量检索做 RRF 融合（hybrid）。

为什么自己写、不用 LanceDB FTS：
  tantivy 默认分词器按空白/标点切词，中文一整段会被当成一个 token，检索基本失效；
  先 jieba 预分词再跑纯 Python BM25，对本项目规模（几百 chunk）完全够用，零原生依赖。

语料由调用方从 LanceDB 全表载入后构建，行数变化（增量同步）后重建即可。
"""
from __future__ import annotations

import math
import re
from collections import Counter

import jieba

# 只保留中英文/数字 token（去标点、空白），转小写统一大小写
_TOKEN_RE = re.compile(r"^[\w\u4e00-\u9fff]+$", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """jieba 搜索引擎模式分词（更侧重召回），过滤纯标点/空白 token。"""
    if not text:
        return []
    out = []
    for w in jieba.cut_for_search(text):
        w = w.strip().lower()
        if w and _TOKEN_RE.match(w):
            out.append(w)
    return out


class CorpusBM25:
    """基于内存语料的 Okapi BM25 检索器。

    语料: rows = [{note_id, seq, title, section, text}, ...]
    短文本、小语料场景足够；构建 O(N·len)，检索 O(N·|q|)。
    """

    def __init__(self, rows: list[dict], k1: float = 1.5, b: float = 0.75):
        self.docs: list[dict] = []
        self._doc_tokens: list[Counter] = []
        self._df: Counter = Counter()  # token -> 包含它的文档数
        total_len = 0
        for r in rows:
            text = "\n".join(filter(None, [r.get("title"), r.get("section"),
                                           r.get("text")]))
            toks = Counter(tokenize(text))
            if not toks:
                continue
            self.docs.append({
                "note_id": r["note_id"],
                "seq": r.get("seq"),
                "title": r.get("title", ""),
                "section": r.get("section", ""),
                "text": r.get("text", ""),
                "_len": sum(toks.values()),
            })
            self._doc_tokens.append(toks)
            for t in toks:
                self._df[t] += 1
            total_len += sum(toks.values())
        self.n = len(self.docs)
        self.avgdl = total_len / self.n if self.n else 0.0
        self.k1 = k1
        self.b = b

    def search(self, query: str, k: int = 20) -> list[dict]:
        """返回 top-k 命中行（note_id/seq/title/section/text 副本），不含 BM25 内部字段。"""
        if not self.n:
            return []
        qtoks = tokenize(query)
        if not qtoks:
            return []
        qcount = Counter(qtoks)
        scores: list[float] = []
        for i, toks in enumerate(self._doc_tokens):
            dl = self.docs[i]["_len"]
            if not dl:
                scores.append(0.0)
                continue
            s = 0.0
            for t, qf in qcount.items():
                f = toks.get(t, 0)
                if not f:
                    continue
                df_t = self._df.get(t, 0)
                idf = math.log(1 + (self.n - df_t + 0.5) / (df_t + 0.5))
                s += idf * qf * (f * (self.k1 + 1)) / (
                    f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            scores.append(s)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in ranked[:k]:
            d = self.docs[i]
            if scores[i] <= 0:
                break
            out.append({kk: d[kk] for kk in
                        ("note_id", "seq", "title", "section", "text")})
        return out


__all__ = ["tokenize", "CorpusBM25"]
