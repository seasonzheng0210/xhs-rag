# xhs-rag Web 服务镜像 —— 只跑「查询 + LLM 问答」,不含同步采集
#
# 设计决策:
#   - 同步功能(Playwright 模拟登录小红书)不进容器: 采集需要登录态/反检测
#     脚本/宿主网络, 且容器 IP 容易触发风控。采集留在宿主机/单独机器跑,
#     容器专注对外提供检索服务 —— 职责分离, 面试可讲。
#   - 模型(bge-m3 2.2GB + reranker 2.3GB)不打进镜像, 挂载宿主 data/models,
#     镜像保持 ~2.5GB; 服务器首次部署可拷本机 data/ 目录过去。
#   - 数据(sqlite + lancedb + vault + thumbs)全部在 data/ 下, 挂载进容器。
FROM python:3.12-slim

WORKDIR /app

# ── 1) torch CPU 版(单独一层, 体积大但 Docker 缓存友好) ──
# Linux 默认 pip 会拉 CUDA 版 torch(2.5GB+), 必须显式指定 CPU index
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# ── 2) 项目依赖(按 pyproject 可选组, 缓存友好: 代码改动不触发重装) ──
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir ".[ocr,index,schedule]" \
 && pip install --no-cache-dir FlagEmbedding modelscope \
 # 锁定与本地数据兼容的版本: lancedb 索引是 0.37.1 写入的,
 # 新版本可能改文件格式; pyarrow 同理
 && pip install --no-cache-dir lancedb==0.37.1 pyarrow==25.0.1

# ── 3) 项目代码 ──
COPY src/ ./src/
COPY config/ ./config/
COPY assets/ ./assets/

ENV PYTHONPATH=/app/src
ENV XHS_RAG_CONFIG=/app/config/config.yaml

EXPOSE 8765

# 首次启动: 模型预热(embedder+reranker, 约 1-5 分钟, 视 CPU 而定)
CMD ["python", "-m", "xhs_rag.cli", "serve"]
