# xhs-rag · 小红书收藏夹 RAG

把自己在小红书收藏 / 点赞过的笔记，变成**可搜索、可问答、能一键回溯原帖**的个人知识库。
Markdown 落盘，Obsidian 可直接打开；数据全部留在本机。

> 合规：仅采集**本人账号**的收藏内容，低频、不外传、不商用。

## 怎么用（两种形态）

| 形态 | 入口 | 适合谁 |
|---|---|---|
| **命令行 / Web UI** | `xhs setup` 一键建库 → `xhs serve` 开 Web 问答 | 想自己检索收藏的人 |
| **AI 助手（MCP / Skill）** | 仓库根 [`SKILL.md`](SKILL.md) 教 AI 读懂本项目；`xhs mcp` 把 search/ask/stats 暴露成 MCP 工具 | 想让 WorkBuddy / Claude 等直接「问我的收藏」的人（接入见 [M9](#mcp-接入m9)） |

## 功能

| 里程碑 | 内容 | 状态 |
|---|---|---|
| **M0** | 环境（Playwright + ffmpeg）+ 扫码登录（持久化登录态） | ✅ |
| **M1** | 收藏列表采集（断点续传、限速、验证码冷却） | ✅ |
| **M2** | 笔记详情 + 图片 / 视频下载（`xsec_token` 原样保留） | ✅ |
| **M3** | 图片 OCR（RapidOCR 本地 + API 混合降级）+ Markdown 落盘 | ✅ |
| **M4** | 视频抽帧 OCR + ASR 转写 | ✅ |
| **M5** | 索引与检索（bge-m3 embedding + **BM25+RRF 混合检索** + bge-reranker 重排，LanceDB） | ✅ |
| **M6** | Web UI（移动端优先，局域网手机可访问） | ✅ |
| **M7** | 增量同步（collect → 详情 → OCR → ASR → 索引，可单次执行或由外部定时驱动） | ✅（内置 scheduler 未实现，定时依赖外部如 WorkBuddy 自动化） |
| **M8** | LLM 问答（检索结果 → 带 `[n]` 引用的流式回答，点角标跳原帖） | ✅ |
| **M9** | MCP server（stdio，把 search / ask / stats 暴露给 WorkBuddy 等 AI 客户端直问收藏夹） | ✅ |

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
pip install -e ".[ocr,index,serve]"
pip install "mcp<2"      # 可选：仅 M9 MCP 接入需要
playwright install chromium

# 0b. 检索模型（二选一，检索/问答必需）
#    A. 本地模型（推荐：免费、无限量、数据不出本机）—— 首次建索引自动下载到 data/models/
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU 版即可
pip install flagembedding modelscope
#    B. 或配云端 embedding API key（见 .env.example，本地模型缺省时自动降级）

# 1. 配置密钥（可选，仅 API 降级链路 / LLM 问答需要）
cp .env.example .env          # 填入 SILICONFLOW_API_KEY 等
python scripts/verify_keys.py

# ※ config.yaml 无需手动创建：代码自动回退 config/config.example.yaml（含默认路径/模型名）。
#   需要改参数（如 rerank.top_k_in、llm.provider、serve.debug）时再 cp config.example.yaml config.yaml 覆盖。

# 2a. 最快路径：一键建库（登录失效会自动弹码 → 全流水线 → 完事）
python -m xhs_rag.cli setup

# 2b. 或分步执行：扫码登录（首次弹出浏览器窗口，扫码后登录态持久化）
python -m xhs_rag.cli login

# 3. 全流水线同步（收藏 → 详情 → OCR → ASR → 索引）
python -m xhs_rag.cli sync

# 4. 启动 Web UI（默认 0.0.0.0:8765，手机同局域网可访问）
python -m xhs_rag.cli serve

# 5. 命令行问答（不想开浏览器时，检索 + LLM 带引用回答）
python -m xhs_rag.cli ask "月子里怎么吃"

# 6. MCP server（可选，AI 客户端如 WorkBuddy 直问收藏夹，接入见下文 M9）
python -m xhs_rag.cli mcp
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
| `python -m xhs_rag.cli setup` | **一键建库**：登录检查（失效自动弹扫码）→ 全流水线 →（`--serve` 连 Web UI） |
| `python -m xhs_rag.cli serve` | 启动 Web UI（检索 + 问答） |
| `python -m xhs_rag.cli ask "问题"` | 命令行问答：检索 + LLM 带引用回答（`-k` 控制片段数） |
| `python -m xhs_rag.cli mcp` | 启动 MCP server（stdio，供 WorkBuddy 等 AI 客户端接入收藏夹问答） |

## MCP 接入（M9）

把收藏 RAG 的 `search` / `ask` / `stats` 暴露为 MCP 工具，**AI 客户端无需开浏览器即可直问你的收藏库**。
也适合作为仓库根 `SKILL.md` 的运行时后端：AI 助手读 SKILL.md 学会本项目，再经 MCP 直查你的数据。

```bash
pip install "mcp<2"   # v1 API（FastMCP）；mcp 2.x 改名为 MCPServer，本项目代码按 v1 写
python -m xhs_rag.cli mcp    # 手动以 stdio 方式跑起来，可配合任意 MCP client 调试
```

WorkBuddy 注册 —— 在 `~/.workbuddy/mcp.json` 的 `mcpServers` 加一条，然后到连接器管理页对 `xhs-rag` 点「信任」：

```json
{
  "mcpServers": {
    "xhs-rag": {
      "command": "<你的 venv 或系统 python 绝对路径>",
      "args": ["-m", "xhs_rag.cli", "mcp"],
      "cwd": "<克隆本仓库的绝对路径>"
    }
  }
}
```

> 把两个 `<...>` 占位符换成你机器上的真实路径：`command` 指向装好依赖的 python（`pip install -e ".[ocr,index,serve]"` 后可直接用系统 python），`cwd` 指向本仓库目录。
> 首次调用会预热本地模型（embedder + reranker，约 10-20s），之后即时响应。

可用工具：

| 工具 | 参数 | 说明 |
|---|---|---|
| `search` | `query`, `k=5` | 语义检索收藏夹，返回带原帖链接的结果 |
| `ask` | `query` | 检索 + LLM 生成带 `[n]` 引用的回答（LLM 不可用退化为检索） |
| `stats` | — | 收藏库统计：笔记 / 图片 / 视频 / ASR 字符 / chunks |

实现要点（都是踩过的坑）：

- **stdout 只走协议**：MCP stdio 要求 stdout 只承载 JSON-RPC。CLI 默认把 loguru 打到 stdout（带 ANSI 颜色），
  任何一行日志混入都会让客户端解析失败 —— `mcp_server.main()` 里强制把日志全部改道 stderr。
- **预热必须在主线程**：MCP 工具在 anyio worker 线程执行，实测在 worker 线程内首次 import torch / 加载
  bge-m3 会死等挂起（0 CPU、内存停在 ~150MB）；因此模型预热放在 `mcp.run()` 之前的主线程做（约 10-20s，
  与 web serve 同策略），启动即就绪、工具调用即时返回。
- **mcp 版本钉 `<2`**：2.x 把 `FastMCP` 改名为 `MCPServer`，本项目按 v1 API 编写。

## LLM 问答（M8）

Web UI 检索后会用 LLM 基于命中的片段生成回答，流式输出，每个事实性陈述带 `[n]` 引用角标，**点击角标可跳到对应笔记卡片**。

- 默认方案（零成本）：**智谱 GLM-4-Flash**（永久免费、不限 Token、128K 上下文）。
  注册 [open.bigmodel.cn](https://open.bigmodel.cn) 实名后，把 key 填到 `.env` 的 `ZHIPU_API_KEY=`。
  配置在 `config.yaml` 的 `llm` 段，改 `provider: deepseek` + `DEEPSEEK_API_KEY` 可切回付费 DeepSeek。
- DeepSeek 备选：`deepseek-v4-flash`（`$0.14 / 1M` 输入、`$0.28 / 1M` 输出，1M 上下文）。
  ⚠️ `deepseek-chat` / `deepseek-reasoner` 已于 **2026-07-24 废弃**，别再写这两个名字。
- **thinking 默认关闭**：V4 系列默认开启 thinking 且 effort=high，推理 token 同样计费、延迟翻倍。
  RAG 问答用不上推理链，代码里显式发了 `{"thinking": {"type": "disabled"}}`。
- 未配 key 或接口异常时**优雅降级**：照常展示检索结果，只多一条提示，不报错。
- 想完全本地可把 `llm.provider` 改成 `ollama` 走本地模型（2 核机器上会很慢）。
- **回答准确性（配方/用量类）**：做过两层加固——
  1. SYSTEM_PROMPT 强制「片段中的具体数字（用量/时间/比例）必须原样保留」，配方/做法类必须写出具体用量；
  2. 检索侧做**同笔记补全**（`retriever._complete_note_chunks`）：配方类笔记的用量常在图 OCR chunk 里，
     rerank 只用前 200 字符打分，OCR 噪声开头会让它被挤出 top5，补全保证「笔记有一个 chunk 进 topN，
     其他 chunk 一并喂给 LLM」。每笔记最多补 3 条，只在向量候选里补。
- 回归验证：`PYTHONPATH=src python scripts/_qa_usage_test.py`（秋葵/沙茶酱/山药三个 case，
  断言回答保留具体用量，可 `... shacha` 只跑单个）。

## 性能优化（检索 35-75s → 9-14s）

i5-5200U 双核 CPU 上，检索链路实测优化前 35-75s、优化后 9-14s（重复问题缓存命中 0s）。三板斧：

1. **OCR 噪声清洗**（`chunk.clean_ocr_noise`）：视频帧 OCR 的 `<|LOC_n|>` 定位标记、纯 emoji 装饰行、
   纯字母乱码串在**分块时**剔除（保守策略，数字数据行/中英混合行保留）。重建索引后 LOC 标记 62→0，
   chunks 234→218。`vault/` 原始 Markdown 不动，只影响索引。
2. **rerank 候选裁剪**：候选数必须由配置 `rerank.top_k_in` 控制（曾因 `max(top_k_out*4, cfg)` 在 k=5
   时强制 20 候选，每对 ~1.2s CPU 推理拖垮检索）；打分截断 160 字符。
3. **缓存**：query 编码缓存（embedder 内建）+ 检索结果 LRU 缓存（`rerank.cache_ttl`，默认 5 分钟），
   重复提问秒回。

配套：`retriever.warmup()` 在服务启动时预热 embedder + reranker（首次查询不再等模型加载）；
`/api/health` 健康检查端点（Docker healthcheck 用，不触发模型推理）。

## 调试模式（所有 Web 项目标配）

报错时页面显示错误卡片，三个操作：**查看错误详情**（完整 traceback，可展开/收起）、**复制错误报告**、**去 WorkBuddy 修复**。

- 报错**自动落盘**到 `data/debug/last_error.txt`（含时间/接口/问题/堆栈），在 WorkBuddy 说「修复上次的错误」即可直接读取修复，无需复制粘贴。
- 开关：`config.yaml` 的 `serve.debug`（个人局域网默认 `true`；若端口映射到公网，**务必改 `false`**，生产模式只返回错误消息、不暴露堆栈）。
- 端到端验证：`PYTHONPATH=src python scripts/_debug_mode_test.py`（假 retriever 构造 500，验证两种模式的响应差异）。

## 部署（Docker，手机随时访问）

镜像只跑**查询 + 问答**服务（不含同步采集）；同步留在宿主机跑（采集需要登录态/反检测脚本，容器 IP 易触发小红书风控，职责分离设计）。

```bash
# 一键启动（本机或云服务器）
docker compose up -d --build
# 浏览器打开 http://<服务器IP>:8765
```

- **数据全挂载**：`./data`（sqlite + lancedb 索引 + 模型 bge-m3/reranker + vault）。升级镜像不丢数据；换机部署把 `data/` 整个拷过去即可。
- **首次启动会预热模型**（embedder + reranker，约 1-5 分钟），之后查询秒级响应。
- 部署到公网前务必做三件事：① `serve.debug` 改 `false`；② 加访问密码（未内置，用 Nginx/Caddy basic auth）；③ 同步改手动触发。
- 服务器内存建议 ≥4GB（embedder ~2GB + reranker ~2GB）。

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
