"""M8 LLM 问答 —— 基于检索结果生成带引用的回答。

接口格式：OpenAI 兼容（DeepSeek / Ollama / 任何 OpenAI 代理），
默认 deepseek-v4-flash，用 requests 发流式请求，不引入 openai SDK。

★ 2026-07-24 起 deepseek-chat / deepseek-reasoner 已废弃，
  统一用 deepseek-v4-flash（同价），thinking 模式是请求级开关：
  - thinking 默认开启且 effort=high，RAG 问答必须显式 {"thinking":{"type":"disabled"}}
  - thinking 开启时 temperature/top_p 会被忽略，且推理 token 也计费
  - 推理链走 reasoning_content 字段，与 content 同级

降级策略：没配 key / 接口报错 / 超时 → 抛 LLMUnavailable，
调用方（Web UI / CLI）只展示检索结果，不影响主流程。
"""
from __future__ import annotations

import json
import os
import time
from typing import Iterator

import requests
from loguru import logger

SYSTEM_PROMPT = """你是「收藏夹 RAG」的问答助手。用户会把自己在小红书收藏过的笔记片段交给你，每条带编号 [1] [2] [3]…

规则：
1. 只依据给定片段作答，不引入外部知识。片段里没有的信息，明确说「收藏里没有提到这一点」。
2. 每个事实性陈述的句末必须标注来源编号，格式如 [1] 或 [2][3]，可同时标注多个。
3. 中文回答，先给结论再展开，要点用短句分条，不要写「根据片段可知」这类开场白。
4. 不要编造片段里不存在的内容、数字或建议。
5. 片段之间若有冲突，如实指出分歧并分别标注来源。
6. 片段中出现的具体数字——用量、时间、温度、数量、比例等——必须原样保留，禁止省略或概括。例如片段写「加2勺生抽+1勺蚝油」，回答也必须写「2勺生抽、1勺蚝油」，不能只写「生抽、蚝油」。
7. 回答配方/做法/步骤类问题时，涉及调料的必须写出具体用量数字，宁可逐条列明细，也不要合并概括。"""


class LLMUnavailable(Exception):
    """LLM 不可用（未配置 / 接口故障），调用方应优雅降级。"""


class Answerer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.enabled = bool(cfg.get("llm.enabled", True))
        self.provider = cfg.get("llm.provider", "deepseek")
        # ollama 走独立配置段
        if self.provider == "ollama":
            self.base_url = cfg.get("llm.ollama.base_url",
                                    "http://127.0.0.1:11434/v1")
            self.model = cfg.get("llm.ollama.model", "qwen3:8b")
            self.api_key_env = ""
        else:
            self.base_url = cfg.get("llm.base_url", "https://api.deepseek.com")
            self.model = cfg.get("llm.model", "deepseek-v4-flash")
            self.api_key_env = cfg.get("llm.api_key_env", "DEEPSEEK_API_KEY")
        self.temperature = float(cfg.get("llm.temperature", 0.2))
        self.max_tokens = int(cfg.get("llm.max_tokens", 2000))
        self.thinking = bool(cfg.get("llm.thinking", False))
        self.thinking_effort = cfg.get("llm.thinking_effort", "low")
        self.timeout = int(cfg.get("llm.timeout", 60))
        # 每条片段喂给 LLM 的最大字符数（rerank 已截断过，这里做二次保险）
        self.max_ctx_chars = int(cfg.get("llm.max_context_chars", 600))

    # ── 可用性 ──────────────────────────────────────────────
    def available(self) -> tuple[bool, str]:
        """返回 (是否可用, 不可用原因)。"""
        if not self.enabled:
            return False, "配置里关闭了 LLM 问答（llm.enabled: false）"
        if self.provider == "ollama":
            return True, ""  # 本地服务，不预检
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            return False, f"未配置 {self.api_key_env}（填到项目根目录的 .env 里）"
        return True, ""

    def _payload(self, messages: list[dict], stream: bool) -> dict:
        body = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "max_tokens": self.max_tokens,
        }
        if self.provider == "deepseek":
            if self.thinking:
                body["thinking"] = {"type": "enabled"}
                body["reasoning_effort"] = self.thinking_effort
            else:
                # ★ 默认开启，必须显式关闭才走 non-thinking（快且省）
                body["thinking"] = {"type": "disabled"}
                body["temperature"] = self.temperature
        else:
            body["temperature"] = self.temperature
        return body

    # ── 构造上下文 ──────────────────────────────────────────
    def build_messages(self, query: str, results: list[dict]) -> list[dict]:
        """把检索结果组装成带编号的上下文。"""
        parts = []
        for i, r in enumerate(results, 1):
            kind = "视频" if r.get("note_type") == "video" else "图文"
            section = f"· {r['section']}" if r.get("section") else ""
            text = (r.get("text") or "")[: self.max_ctx_chars]
            parts.append(f"[{i}] 《{r.get('title', '无标题')}》{kind}{section}\n{text}")
        context = "\n\n".join(parts)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"以下是我收藏夹里的相关片段：\n\n{context}\n\n"
                        f"问题：{query}\n\n请按规则回答。"},
        ]

    # ── 请求 ────────────────────────────────────────────────
    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env, "").strip()
            if key:
                h["Authorization"] = f"Bearer {key}"
        return h

    def stream(self, query: str, results: list[dict]) -> Iterator[str]:
        """流式产出回答文本。不可用或出错抛 LLMUnavailable。"""
        ok, why = self.available()
        if not ok:
            raise LLMUnavailable(why)

        url = self.base_url.rstrip("/") + "/chat/completions"
        body = self._payload(self.build_messages(query, results), stream=True)
        try:
            resp = requests.post(url, json=body, headers=self._headers(),
                                 stream=True, timeout=(10, self.timeout))
            resp.raise_for_status()
        except Exception as e:
            raise LLMUnavailable(f"{self.provider} 请求失败: {e}") from e

        for raw in resp.iter_lines():
            if not raw:
                continue
            if not raw.startswith(b"data:"):
                continue
            data = raw[5:].strip()
            if data == b"[DONE]":
                break
            try:
                chunk = json.loads(data)
            except Exception:
                continue
            try:
                delta = chunk["choices"][0]["delta"]
                # thinking 开启时推理链在 reasoning_content，正文才是 content
                text = delta.get("content") or ""
            except (KeyError, IndexError):
                continue
            if text:
                yield text

    def answer(self, query: str, results: list[dict]) -> str:
        """一次性返回完整回答（CLI 用）。"""
        return "".join(self.stream(query, results))


def pretty_stream(query: str, results: list[dict], answerer: Answerer,
                  echo=print) -> dict:
    """CLI 辅助：流式打印并统计耗时。返回 {text, secs}。"""
    t0 = time.time()
    buf: list[str] = []
    for piece in answerer.stream(query, results):
        buf.append(piece)
        echo(piece, end="", flush=True)
    echo()
    return {"text": "".join(buf), "secs": round(time.time() - t0, 1)}
