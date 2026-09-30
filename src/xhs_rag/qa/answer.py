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
import re
import time
from typing import Iterator

import requests
from loguru import logger

SYSTEM_PROMPT = """你是「收藏夹 RAG」的问答助手。用户会把自己在小红书收藏过的笔记片段交给你，每条带编号 [1] [2] [3]…

【步骤/做法类问题，格式硬约束】
- 同一容器、同一主料上的连续操作必须合并为一步。
  ✅ 正确：「碗中放入蒜末、小米辣，淋上热油激香，加2勺生抽、1勺蚝油搅匀成料汁」（一步完成）
  ❌ 禁止：拆成两步「碗中放入蒜末、小米辣，淋上热油激香。」+「加2勺生抽、1勺蚝油，搅匀成料汁」（第二步没说加到哪个容器）
- 每一步必须明确动作对象（碗/锅/盘/烤盘等），禁止悬空动作，禁止「加X」却不说加到哪。

【作答前自检（★ 最重要的规则，先做这一步）】
片段"属于同一领域"不等于"回答了这个问题"。作答前必须逐个确认：
**问题的核心对象 / 动作 / 场景，在片段原文里真的出现过吗？**
- 问「宝宝发烧怎么物理降温」，片段只讲「带娃偏方」「出门时长」→ 核心词
  「发烧 / 降温」在片段里从未出现 → **必须拒答**。
- 问「增肌期一天吃多少蛋白质」，片段只讲「掉秤减脂」→ 目标相反 →
  **必须拒答**。
- 问「高血压吃什么药」，片段只讲「育儿护理」→ 领域完全不同 → **必须拒答**。
只要核心对象没出现，就属于"没有内容"，按规则 1 明确拒答。**宁可少答，
不可硬答**；硬答比拒答严重得多。

⚠️ 但也别过头——**同义说法不算"没出现"**：问「猪蹄猪手下酒菜」而片段写
「沙姜猪手」，问「掉秤」而片段写「减脂」，这些核心对象**是出现了的**，
必须正常作答，不要因为用词不完全一致就拒答。判据是"片段里有没有这个东西"，
不是"有没有一模一样的字面"。该答而不答，和答错一样是错误。

【通用规则】
1. 只依据给定片段作答，不引入外部知识。片段里没有的信息，明确说「收藏里没有提到这一点」。
2. 每个事实性陈述的句末必须标注来源编号，格式如 [1] 或 [2][3]，可同时标注多个。
   ⚠️ 引用编号只能挂在**片段原文确实写了**的内容上。禁止"先用自己的知识写出答案、
   再把片段编号贴上当作依据"——这属于伪造引用，比不标引用更严重。
3. 中文回答，先给结论再展开，要点用短句分条，不要写「根据片段可知」这类开场白。
4. 不要编造片段里不存在的内容、数字或建议。
5. 片段之间若有冲突，如实指出分歧并分别标注来源。
6. 片段中出现的具体数字——用量、时间、温度、数量、比例等——必须原样保留，禁止省略或概括。例如片段写「加2勺生抽+1勺蚝油」，回答也必须写「2勺生抽、1勺蚝油」，不能只写「生抽、蚝油」。
7. 涉及调料的具体用量必须写明，不可省略成「适量」。
8. 拒答时只说「收藏里没有提到这一点」，并可用一句话说明片段实际讲的是什么主题；
   不要推荐外部网站、链接、教程或搜索词，也不要凭通用常识给出步骤。"""

# 检索低置信时的追加提示（由 Retriever 的 dense 距离判据标记 results[0].
# low_confidence，见 retriever._search_once）。作用是让"作答前自检"从默认
# 的"看情况"变成"重点检查"—— 实测这批 query 正是最容易硬答的地方。
LOW_CONF_NOTE = """⚠️ 本次检索置信度偏低：下面这些片段很可能只是与问题「同领域」，
并未包含问题的答案。请严格执行「作答前自检」——逐个确认问题的核心对象/动作/
场景是否在片段原文里出现过；只要没出现，就按规则 1 明确拒答（「收藏里没有提到
这一点」）。绝对不要用自己的知识把答案补齐，也不要给片段未提及的内容挂引用编号。"""


# 多轮改写用的 system prompt(独立于问答 prompt,只做检索词改写)
REWRITE_PROMPT = """你在多轮对话里改写检索词。给出对话历史和用户的最新问题,
把它改写成一句**不依赖对话历史就能独立理解**的完整检索查询:
- 补全指代(「第二个」「那个汤」「它怎么做」→ 指明的具体对象)
- 补全省略的主语/宾语,保留用户的原始意图和关键词
- 只输出改写后的查询词本身,一行,不要解释、不要加引号
- 如果最新问题本身已经独立完整,原样输出"""


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

    # ── 多轮对话 ────────────────────────────────────────────
    # 指代/省略启发式: 命中任一即视为追问,需要结合历史改写检索词
    _FOLLOWUP_RE = re.compile(
        r"^(第[一二三四五六七八九十\d]+个?|那个?|这个|它|他们|上面的?|"
        r"还有(吗|呢|别的)?|换个?|再来|详细说说|展开|为什么|怎么做|多少钱|"
        r"用量|步骤|做法呢?|呢$|吗$)"
    )

    @classmethod
    def needs_rewrite(cls, query: str, history: list[dict] | None) -> bool:
        """判断是否需要结合历史改写检索词。

        规则: 没有 history 一律不改;query 已经够长且具体(>=12 字且不含
        指代词)也不改,省一次 LLM 调用。
        """
        if not history:
            return False
        q = query.strip()
        if len(q) >= 12 and not cls._FOLLOWUP_RE.search(q):
            return False
        return True

    def rewrite_query(self, query: str, history: list[dict] | None = None) -> str:
        """结合对话历史把追问改写成独立检索词。失败时原样返回(不影响主流程)。"""
        msgs = [{"role": "system", "content": REWRITE_PROMPT}]
        for h in (history or [])[-4:]:  # 最多带 2 轮(4 条),改写不需要更长
            msgs.append({"role": h.get("role", "user"),
                         "content": (h.get("content") or "")[:400]})
        msgs.append({"role": "user", "content": query})
        try:
            resp = requests.post(
                self.base_url.rstrip("/") + "/chat/completions",
                json=self._payload(msgs, stream=False)
                | {"max_tokens": 100},
                headers=self._headers(), timeout=(10, 20))
            resp.raise_for_status()
            out = (resp.json()["choices"][0]["message"]["content"] or "").strip()
            out = out.strip("\"'「」"" ").splitlines()[0].strip()
            return out or query
        except Exception as e:
            logger.warning(f"query 改写失败,用原始 query 检索: {e}")
            return query

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
    def build_messages(self, query: str, results: list[dict],
                       history: list[dict] | None = None,
                       memory_note: str = "") -> list[dict]:
        """把检索结果组装成带编号的上下文。

        history: 多轮对话历史 [{role: user|assistant, content: str}],
        注入在系统提示与当前问题之间(assistant 回答截断 300 字防膨胀)。
        memory_note: M11 长期记忆的背景注记文本(内存摘要+用户画像,
        由 memory.recent_context 生成)。以第二条 system 消息注入在
        history 之前；文本自带"非检索片段"声明，防止被当事实来源。
        """
        parts = []
        for i, r in enumerate(results, 1):
            kind = "视频" if r.get("note_type") == "video" else "图文"
            section = f"· {r['section']}" if r.get("section") else ""
            text = (r.get("text") or "")[: self.max_ctx_chars]
            parts.append(f"[{i}] 《{r.get('title', '无标题')}》{kind}{section}\n{text}")
        context = "\n\n".join(parts)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if memory_note:
            messages.append({"role": "system", "content": memory_note})
        # 检索低置信（dense top1 距离越界）→ 追加从严自检提示。
        # 标记来自 Retriever._search_once；关掉 rerank 后这是唯一可用的
        # "相不相关"信号，所以放在 prompt 里而不是当硬门控（阈值有重叠区）。
        if results and results[0].get("low_confidence"):
            messages.append({"role": "system", "content": LOW_CONF_NOTE})
        for h in (history or [])[-8:]:  # 最多 4 轮
            content = (h.get("content") or "").strip()
            if content:
                messages.append({"role": h.get("role", "user"),
                                 "content": content[:300]
                                 if h.get("role") == "assistant" else content})
        messages.append({"role": "user",
                         "content": f"以下是我收藏夹里的相关片段：\n\n{context}\n\n"
                                    f"问题：{query}\n\n请按规则回答。"})
        return messages

    # ── 请求 ────────────────────────────────────────────────
    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env, "").strip()
            if key:
                h["Authorization"] = f"Bearer {key}"
        return h

    def stream(self, query: str, results: list[dict],
               history: list[dict] | None = None,
               memory_note: str = "") -> Iterator[str]:
        """流式产出回答文本。不可用或出错抛 LLMUnavailable。

        history: 多轮对话历史,见 build_messages。
        memory_note: M11 长期记忆背景注记,见 build_messages。
        """
        ok, why = self.available()
        if not ok:
            raise LLMUnavailable(why)

        url = self.base_url.rstrip("/") + "/chat/completions"
        body = self._payload(
            self.build_messages(query, results, history, memory_note),
            stream=True)
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

    def answer(self, query: str, results: list[dict],
               history: list[dict] | None = None,
               memory_note: str = "") -> str:
        """一次性返回完整回答（CLI 用）。"""
        return "".join(self.stream(query, results, history, memory_note))


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
