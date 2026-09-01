#!/usr/bin/env python3
"""
API 密钥连通性自检 —— 零依赖，标准库即可运行。

用法：
    python scripts/verify_keys.py

检查项：
    1. .env 是否读到 key，格式是否合理
    2. 硅基流动：是否完成实名认证（★ 最关键，未实名则全部功能停用）
    3. 各供应商 embedding / rerank 免费模型是否真的可调用
    4. 免费 OCR / ASR 模型 id 是否在线

退出码：0 = 主力供应商可用；1 = 有阻塞项
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"

OK = "[ OK ]"
FAIL = "[FAIL]"
WARN = "[WARN]"
SKIP = "[SKIP]"

# 平台要求的实名状态：硅基流动未实名 = 全部功能停用
SILICONFLOW_CRITICAL = True

PROVIDERS = {
    "siliconflow": {
        "env": "SILICONFLOW_API_KEY",
        "name": "硅基流动（主力 / 免费）",
        "base": "https://api.siliconflow.cn/v1",
        "prefix": "sk-",
        "embedding": "BAAI/bge-m3",
        "rerank": "BAAI/bge-reranker-v2-m3",
        "watch_models": [
            "PaddlePaddle/PaddleOCR-VL-1.5",
            "Qwen/Qwen3-ASR-1.7B",
        ],
    },
    "bailian": {
        "env": "DASHSCOPE_API_KEY",
        "name": "阿里云百炼（备份 / 约 ¥19）",
        "base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "prefix": "sk-",
        "embedding": "text-embedding-v4",
        "rerank": "qwen3-rerank",
        "watch_models": ["qwen3-vl-flash", "fun-asr"],
    },
    "zhipu": {
        "env": "ZHIPU_API_KEY",
        "name": "智谱（末位备份 / 约 ¥192）",
        "base": "https://open.bigmodel.cn/api/paas/v4",
        "prefix": "",
        "embedding": "embedding-3",
        "rerank": "rerank",
        "watch_models": ["glm-4v-flash"],
    },
}


def load_env(path: Path) -> dict:
    """极简 .env 解析，不依赖 python-dotenv。"""
    env = {}
    if not path.exists():
        return env
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def mask(key: str) -> str:
    if not key:
        return "(空)"
    if len(key) <= 10:
        return key[:3] + "*" * (len(key) - 3)
    return f"{key[:7]}...{key[-4:]} ({len(key)} 位)"


def http_json(url: str, key: str, payload=None, timeout=30):
    """返回 (ok, status, data_or_errortext, elapsed_ms)。"""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    req.add_header("Authorization", f"Bearer {key}")
    req.add_header("Content-Type", "application/json")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return True, resp.status, json.loads(body), int((time.time() - t0) * 1000)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return False, e.code, body[:400], int((time.time() - t0) * 1000)
    except Exception as e:  # 网络层：超时 / DNS / 连接重置
        return False, 0, f"{type(e).__name__}: {e}", int((time.time() - t0) * 1000)


def test_embedding(base: str, key: str, model: str):
    ok, status, data, ms = http_json(
        f"{base}/embeddings", key,
        {"model": model, "input": ["小红书收藏夹 RAG 连通性测试"], "encoding_format": "float"},
    )
    if not ok:
        return False, f"HTTP {status} {data}"
    try:
        vec = data["data"][0]["embedding"]
        dim = len(vec)
    except Exception:
        return False, f"响应结构异常：{str(data)[:200]}"
    usage = data.get("usage", {}).get("total_tokens", "?")
    return True, f"dim={dim}, tokens={usage}, {ms}ms"


def test_rerank(base: str, key: str, model: str):
    """
    用「强相关 / 强不相关」对照对验证，并检查排序与分数可分性。

    ★ 实测结论（2026-08-30，硅基流动 bge-reranker-v2-m3）：
      分数是 sigmoid 归一化到 [0,1]，且两极化 —— 强相关 0.9997 / 不相关 0.000016。
      因此检索时**只能用相对排序，不能设绝对阈值**（设 0.5 会大量误杀）。
    """
    docs = ["苹果是一种水果", "今天股市大涨"]
    ok, status, data, ms = http_json(
        f"{base}/rerank", key,
        {"model": model, "query": "苹果是什么", "documents": docs, "top_n": 2},
    )
    if not ok:
        return False, f"HTTP {status} {data}"
    try:
        res = data["results"]
        scores = {r["index"]: r["relevance_score"] for r in res}
        top1 = res[0]["index"]
    except Exception:
        return False, f"响应结构异常：{str(data)[:200]}"

    if top1 != 0:
        return False, f"排序错误：top1 命中不相关文档（{scores}）"
    if scores.get(0, 0) - scores.get(1, 1) < 0.5:
        return False, f"分数区分度不足，rerank 可能未生效（{scores}）"
    return True, (f"排序正确，区分度 {scores[0]:.4f} vs {scores[1]:.6f}，{ms}ms"
                  f"（sigmoid 归一化，勿设绝对阈值）")


def check_siliconflow_verification(base: str, key: str):
    """
    硅基流动实名状态 —— 未实名则 2026-05-15 起停用全部平台功能。

    注：2026-08-30 实测 /v1/user/info 已返回 410 deprecated，无法直接查询。
    改用「能否成功调用免费模型」反推：平台公告明确未实名账户停用全部功能，
    因此免费 embedding 能调通 ≈ 实名已通过。
    """
    ok, status, data, ms = http_json(
        f"{base}/embeddings", key,
        {"model": "BAAI/bge-m3", "input": ["实名状态探测"], "encoding_format": "float"},
        timeout=20,
    )
    if ok:
        return True, "反推：免费模型可调用 → 账户功能未停用 → 实名已通过"
    txt = str(data)
    if status == 403 or "real" in txt.lower() or "verif" in txt.lower() or "实名" in txt:
        return False, f"疑似未实名：{txt[:200]}"
    return None, f"无法确认（HTTP {status} {txt[:200]}）"


def check_models_online(base: str, key: str, wanted: list):
    ok, status, data, ms = http_json(f"{base}/models", key, timeout=25)
    if not ok:
        return None, f"无法列出模型（HTTP {status} {str(data)[:120]}）"
    items = data.get("data", []) if isinstance(data, dict) else []
    online = {it.get("id", ""): it for it in items if isinstance(it, dict)}
    found = [m for m in wanted if m in online]
    missing = [m for m in wanted if m not in online]
    return found, missing


def main():
    print("=" * 68)
    print(" xhs-rag  API 密钥连通性自检")
    print("=" * 68)

    env = load_env(ENV_FILE)
    if not env:
        print(f"{FAIL} 读不到 {ENV_FILE}，先创建它再跑")
        return 1

    filled = {k: v for k, v in env.items() if v and not v.startswith("<")}
    print(f"\n.env 路径：{ENV_FILE}")
    print(f"已填写：{len(filled)} 个变量\n")

    results = {}
    blocking = []

    for pid, cfg in PROVIDERS.items():
        key = filled.get(cfg["env"], "")
        print("-" * 68)
        print(f"■ {cfg['name']}")
        print(f"  变量 {cfg['env']}  ->  {mask(key)}")

        if not key:
            print(f"  {SKIP} 未填写，跳过\n")
            results[pid] = "skip"
            continue

        if cfg["prefix"] and not key.startswith(cfg["prefix"]):
            print(f"  {WARN} 格式可疑：通常应以 '{cfg['prefix']}' 开头，继续尝试…")

        # 实名状态（仅硅基流动关心，且是硬门槛）
        if pid == "siliconflow":
            verified, detail = check_siliconflow_verification(cfg["base"], key)
            if verified is True:
                print(f"  {OK} 实名认证：已通过")
            elif verified is False:
                print(f"  {FAIL} 实名认证：未通过 —— 2026-05-15 起未实名将停用全部平台功能")
                print(f"        且限流会降到 10 RPM，本项目等于不可用")
                blocking.append("硅基流动未完成实名认证")
            else:
                print(f"  {WARN} 实名状态无法确认：{detail}")

        ok_emb, msg_emb = test_embedding(cfg["base"], key, cfg["embedding"])
        print(f"  {OK if ok_emb else FAIL} Embedding  {cfg['embedding']}")
        print(f"         {msg_emb}")

        ok_rer, msg_rer = test_rerank(cfg["base"], key, cfg["rerank"])
        print(f"  {OK if ok_rer else FAIL} Rerank     {cfg['rerank']}")
        print(f"         {msg_rer}")

        if cfg.get("watch_models"):
            found, missing = check_models_online(cfg["base"], key, cfg["watch_models"])
            if found is None:
                print(f"  {WARN} 模型清单 {missing}")
            else:
                for m in found:
                    print(f"  {OK} 模型在线   {m}")
                for m in missing:
                    print(f"  {WARN} 模型缺失   {m}（可能已下架或改名，需核对定价页）")

        results[pid] = "ok" if (ok_emb and ok_rer) else "fail"
        if results[pid] == "fail":
            blocking.append(f"{cfg['name']} 调用失败")
        print()

    # ── 汇总 ──────────────────────────────────────────────
    print("=" * 68)
    print(" 汇总")
    print("=" * 68)
    for pid, cfg in PROVIDERS.items():
        st = results.get(pid, "skip")
        label = {"ok": "可用", "fail": "不可用", "skip": "未配置"}[st]
        print(f"  {cfg['name']:<28} {label}")

    print()
    if results.get("siliconflow") == "ok":
        print(f"{OK} 主力供应商可用 —— 可以开 M0 了")
        # 剩余变量提示
        llm = filled.get("DEEPSEEK_API_KEY", "")
        if not llm:
            print(f"{WARN} DEEPSEEK_API_KEY 未填：不影响抓取和检索，"
                  f"只影响 Web UI 的「对话式问答」功能，M6 前补上即可")
        return 0

    print(f"{FAIL} 阻塞项：")
    for b in blocking:
        print(f"  · {b}")
    if results.get("siliconflow") == "fail" and results.get("bailian") == "ok":
        print(f"\n{WARN} 硅基流动不可用，但百炼可用 —— 可临时切百炼，全量成本约 ¥19")
    return 1


if __name__ == "__main__":
    sys.exit(main())
