---
name: xhs-rag
description: 小红书收藏夹 RAG —— 把用户自己在小红书收藏/点赞过的笔记变成可搜索、可问答、能回溯原帖的个人知识库。当用户说「建立小红书 RAG / 问我的小红书收藏 / 检索我收藏的笔记 / 把收藏夹做成知识库 / 用小红书收藏回答我」或需要对自己的小红书收藏做语义检索、带引用问答、MCP 接入时使用。只采集本人账号数据，Markdown 落盘本地优先。
---

# xhs-rag · 小红书收藏夹 RAG

把用户**自己账号**收藏/点赞过的小红书笔记，采集到本地并做成 RAG：
图片 OCR（RapidOCR 本地 + API 降级）、视频抽帧 OCR + ASR（SenseVoiceSmall/funasr）、
bge-m3 embedding + BM25+RRF 混合检索 + bge-reranker 重排（LanceDB）、
Web UI 问答 / 命令行问答 / MCP server（search/ask/stats 三工具）三种消费方式。

- 合规边界：仅采集**本人账号**的收藏，低频、不外传、不商用。
- 隐私边界：`.env`(key)、`data/`(登录态/图片/视频/库)、`vault/`(收藏 md)、`config/config.yaml`(含 user_id)
  全部 gitignore，不入库；检索链路本地模型，数据不出本机。

## 何时用

- 用户想基于自己的小红书收藏提问、检索、做知识库（「我收藏过的 X 怎么说」「检索我收藏夹里关于 Y 的笔记」）。
- 用户想把收藏夹 RAG 暴露给 AI 客户端（WorkBuddy 等）直问 —— 走 MCP。
- 初次搭建走 `setup` 一键命令；日常增量走 `sync`；查询走 `ask`/`serve`/MCP。

## 环境要求

- Python ≥ 3.11（3.13 实测 OK）
- ffmpeg（视频抽帧/ASR）
- 可选 API key：SILICONFLOW（OCR/ASR 降级）、ZHIPU/DeepSeek（问答 LLM），见 `.env.example`
- 本地检索模型 bge-m3/bge-reranker（torch CPU + flagembedding + modelscope 手动装，见 pyproject 注释）

## 快速使用

```bash
# 0. 安装（仓库根目录）
pip install -e ".[ocr,index,serve]"
pip install "mcp<2"          # 仅 MCP 需要（v1 API；2.x 改名 MCPServer，本项目按 v1 写）
playwright install chromium  # 首次扫码登录用
# 本地检索模型（按需，CPU 即可）：
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
#   pip install flagembedding modelscope

# 1. 配置（可选，仅 API 降级/LLM 问答需要；不配也能跑，本地 RapidOCR + 检索兜底）
cp .env.example .env         # 填 key
python scripts/verify_keys.py

# 2. 一键建库（登录态失效会自动弹码 → 全流水线 → 完事）
xhs setup
#    或分步：xhs login → xhs sync

# 3. 消费
xhs serve              # Web UI http://0.0.0.0:8765（手机同局域网可访问）
xhs ask "月子里怎么吃"  # 命令行问答，带 [n] 引用
xhs mcp                # MCP stdio server（接入见 README「MCP 接入（M9）」）
```

全部命令等价 `python -m xhs_rag.cli <cmd>`；Windows 可双击 `scripts/run.bat`。
`xhs doctor` 可先做环境自检。

## MCP 接入要点（把 RAG 暴露给 AI 客户端）

注册到 `~/.workbuddy/mcp.json`（其他客户端按其规范等效配置），然后对 `xhs-rag` 点「信任」：

```json
{
  "mcpServers": {
    "xhs-rag": {
      "command": "<你的 venv python 绝对路径>",
      "args": ["-m", "xhs_rag.cli", "mcp"],
      "cwd": "<本仓库绝对路径>"
    }
  }
}
```

工具：`search(query, k=5)` 语义检索带原帖链接；`ask(query)` 检索 + LLM 带 `[n]` 引用回答；
`stats()` 收藏库统计。三个实现坑已在代码注释/README 写明：stdout 只走 JSON-RPC（日志改道 stderr）、
模型预热必须在 `mcp.run()` 前的主线程（anyio worker 内加载 torch 会死锁）、mcp 版本钉 `<2`。

## 调参与排错

- `xhs doctor`：环境自检（Python/Chromium/ffmpeg/key）。
- 检索慢：`rerank.top_k_in`（候选池）是主杠杆，默认 6；语料 >1k chunks 后可调大 12-20。
  `scripts/bench_pool.py` 交叉实验决定。小语料(<2 万行)走 flat 扫描不建 ANN。
- 登录态失效：`xhs check` 看状态，`xhs login` 重新扫码；别删 `data/browser_profile/`（删=换设备，易触发风控）。
- 更多：README（架构/性能/调试模式/Docker 部署/隐私）。

## 注意事项

- 只在用户授权其本人小红书账号时采集；不做他人内容抓取。
- 输出引用真实笔记链接，LLM 回答带 `[n]` 角标可回溯原帖。
- 数据诚实：片段中的具体数字（用量/时间/比例）原样保留，见 README「LLM 问答（M8）」加固说明。
