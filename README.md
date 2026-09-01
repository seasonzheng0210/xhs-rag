# xhs-rag · 小红书收藏夹 RAG

把自己在小红书收藏 / 点赞过的笔记，变成**可搜索、可问答、能一键回溯原帖**的个人知识库。
Markdown 落盘，Obsidian 可直接打开；数据全部留在本机。

> 合规：仅采集**本人账号**的收藏内容，低频、不外传、不商用。

## 功能

| 里程碑 | 内容 | 状态 |
|---|---|---|
| **M0** | 环境（Playwright + ffmpeg）+ 扫码登录（持久化登录态） | ✅ |
| **M1** | 收藏列表采集（断点续传、限速、验证码冷却） | ✅ |
| **M2** | 笔记详情 + 图片 / 视频下载（`xsec_token` 原样保留） | ✅ |
| **M3** | 图片 OCR（RapidOCR 本地 + API 混合降级）+ Markdown 落盘 | ✅ |
| **M4** | 视频抽帧 OCR + ASR 转写 | ✅ |
| **M5** | 索引与检索（bge-m3 embedding + bge-reranker 重排，LanceDB） | ✅ |
| **M6** | Web UI（移动端优先，局域网手机可访问） | ✅ |
| **M7** | 定时增量同步（每日自动 collect → 详情 → OCR → ASR → 索引） | ✅ |
| **M8** | LLM 问答（检索结果 → 带 `[n]` 引用的流式回答，点角标跳原帖） | ✅ |

## 数据流

```
小红书收藏夹 ──▶ 扫码登录(Playwright 持久化)
     │
     ▼
收藏列表(JSONL+SQLite) ──▶ 笔记详情 ──▶ 图片 / 视频 ──▶ 本地磁盘
                                        │             │
                                        ▼             ▼
                                    OCR(图片)    抽帧 OCR + ASR(视频)
                                        │             │
                                        ▼             ▼
                                    Markdown(Vault) ◀┘
                                        │
                                        ▼
                        切分 chunk ──▶ embedding ──▶ LanceDB
                                        │
                                        ▼
                    Web UI: query → 向量检索 → rerank 重排 → 结果卡片
                                        │
                                        ▼
                            LLM 问答(SSE 流式, 带 [n] 引用可点跳转)
```

## 环境要求

- Python ≥ 3.11（本机 3.13）
- ffmpeg（视频抽帧 / ASR 用，可执行文件或 PATH 均可）
- 可选：硅基流动 API key（M3/M4 的 OCR/ASR 升级降级用，本地 RapidOCR 兜底）
- 可选：DeepSeek API key（M8 LLM 问答用，不填则只出检索结果，不影响其他功能）

检索链路（embedding + rerank）为**纯本地模型**（bge-m3 + bge-reranker-base），无外部 API 依赖，数据不出本机。

## 快速开始

```bash
# 0. 安装
pip install -e ".[ocr,index,serve,schedule]"
playwright install chromium

# 1. 配置密钥（可选，仅 API 降级链路需要）
cp .env.example .env          # 填入 SILICONFLOW_API_KEY 等
python scripts/verify_keys.py

# 2. 扫码登录（首次弹出浏览器窗口，扫码后登录态持久化）
python -m xhs_rag.cli login

# 3. 全流水线同步（收藏 → 详情 → OCR → ASR → 索引）
python -m xhs_rag.cli sync

# 4. 启动 Web UI（默认 0.0.0.0:8765，手机同局域网可访问）
python -m xhs_rag.cli serve

# 5. 命令行问答（不想开浏览器时，检索 + LLM 带引用回答）
python -m xhs_rag.cli ask "月子里怎么吃"
```

Windows 下也可双击 [`scripts/run.bat`](scripts/run.bat)。

## 命令

| 命令 | 作用 |
|---|---|
| `python -m xhs_rag.cli doctor` | 环境自检：Python / Chromium / ffmpeg / API key |
| `python -m xhs_rag.cli login` | 扫码登录，登录态写入 `data/auth/` + `data/browser_profile/` |
| `python -m xhs_rag.cli check` | 登录态检查（`--offline` 只查本地文件） |
| `python -m xhs_rag.cli collect` | 同步收藏列表到 SQLite + JSONL |
| `python -m xhs_rag.cli sync` | 全流水线：collect → detail → OCR → ASR → 增量索引 |
| `python -m xhs_rag.cli serve` | 启动 Web UI（检索 + 问答） |
| `python -m xhs_rag.cli ask "问题"` | 命令行问答：检索 + LLM 带引用回答（`-k` 控制片段数） |

## LLM 问答（M8）

Web UI 检索后会用 LLM 基于命中的片段生成回答，流式输出，每个事实性陈述带 `[n]` 引用角标，**点击角标可跳到对应笔记卡片**。

- 模型：`deepseek-v4-flash`（`$0.14 / 1M` 输入、`$0.28 / 1M` 输出，1M 上下文）。
  ⚠️ `deepseek-chat` / `deepseek-reasoner` 已于 **2026-07-24 废弃**，别再写这两个名字。
- **thinking 默认关闭**：DeepSeek V4 系列默认开启 thinking 且 effort=high，推理 token 同样计费、延迟翻倍。
  RAG 问答用不上推理链，代码里显式发了 `{"thinking": {"type": "disabled"}}`。
- 未配 `DEEPSEEK_API_KEY` 或接口异常时**优雅降级**：照常展示检索结果，只多一条提示，不报错。
- 想零成本可把 `llm.provider` 改成 `ollama` 走本地模型（2 核机器上会很慢）。

## 目录

```
config/            配置（config.yaml 不入库，由 config.example.yaml 复制）
src/xhs_rag/       源码
  auth/            扫码登录 + storage_state 持久化
  crawler/         收藏列表 / 详情 / 媒体下载
  process/         OCR（RapidOCR 本地 + API 混合）与视频 ASR
  index/           chunk 切分 / embedding / LanceDB 索引 / rerank 检索
  serve/           Web UI（标准库 http.server，零额外依赖）
  store/           SQLite + JSONL
assets/            stealth.min.js 反自动化检测脚本（自写）
data/              运行时数据（全部 gitignore，不入库）
  browser_profile/ ★ Playwright 持久化目录，设备指纹恒定，别删
vault/             输出的 Obsidian vault（gitignore，不入库）
scripts/           验证与入口脚本
```

> ⚠️ `data/browser_profile/` 是登录信任的载体。**删掉它等于换设备**，小红书大概率会弹验证码。

## 隐私

- `.env`（API key）、`data/`（含登录态、图片、视频、数据库）、`vault/`（个人收藏 Markdown）、`config/config.yaml`（含 user_id）**全部 gitignore**，不入库。
- 检索链路全本地模型，收藏内容不出本机（仅 OCR/ASR 的 API 降级链路会发送图片/音频到对应云服务，可在配置中关闭）。
