"""配置加载：config/config.yaml + .env。

设计要点：
- 项目根目录 = 本文件的上四级（src/xhs_rag/core/config.py → core → xhs_rag → src → 项目根）
- config.yaml 不存在时自动回退 config.example.yaml，避免每次换机器都要先复制
- 所有相对路径统一按项目根解析成绝对路径，杜绝「换个 cwd 就找不到文件」
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# 全项目唯一的项目根基准，其他模块一律 from .core.config import ROOT，不要各算各的
ROOT = Path(__file__).resolve().parents[3]

DEFAULT_REL_PATHS = {
    "paths.data_dir": "data",
    "paths.db": "data/xhs.db",
    "paths.jsonl_dir": "data/jsonl",
    "paths.images_dir": "data/images",
    "paths.lancedb_dir": "data/lancedb",
    "paths.vault_dir": "vault",
    "paths.browser_profile": "data/browser_profile",
    "paths.storage_state": "data/auth/storage_state.json",
}


class Config:
    """带点号路径访问的配置容器。"""

    def __init__(self, data: dict, source: Path):
        self._data = data
        self.source = source
        self.root = ROOT

    # ── 基础访问 ──────────────────────────────────────────
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @property
    def raw(self) -> dict:
        return self._data

    # ── 路径 ──────────────────────────────────────────────
    def path(self, dotted: str) -> Path:
        """取配置项并解析为绝对路径（相对路径一律相对项目根）。"""
        value = self.get(dotted) or DEFAULT_REL_PATHS.get(dotted)
        if value is None:
            raise KeyError(f"配置项不存在且无默认值: {dotted}")
        p = Path(os.path.expandvars(str(value)))
        return p if p.is_absolute() else (ROOT / p)

    def ensure_dirs(self, *dotted: str) -> None:
        for key in dotted:
            p = self.path(key)
            # 带后缀的路径当文件处理，建父目录
            (p.parent if p.suffix else p).mkdir(parents=True, exist_ok=True)

    # ── 便捷属性 ──────────────────────────────────────────
    @property
    def api_key(self) -> str:
        """主力供应商的 key（从 .env 读，变量名由 config 的 api_key_env 决定）。"""
        env_name = self.get("providers.siliconflow.api_key_env", "SILICONFLOW_API_KEY")
        return os.environ.get(env_name, "").strip()


def load_config(path: Path | None = None) -> Config:
    """加载配置。顺序：显式指定 → config/config.yaml → config/config.example.yaml"""
    load_dotenv(ROOT / ".env", override=False)

    candidates = [path] if path else [ROOT / "config" / "config.yaml",
                                      ROOT / "config" / "config.example.yaml"]
    for cand in candidates:
        if cand and cand.exists():
            data = yaml.safe_load(cand.read_text(encoding="utf-8")) or {}
            return Config(data, cand)
    raise FileNotFoundError(
        f"找不到配置文件，请复制 config/config.example.yaml 为 config/config.yaml（根目录 {ROOT}）"
    )


def save_user_id(config: Config, user_id: str) -> None:
    """首次登录成功后把 user_id 写回 config.yaml（不存在则新建）。"""
    target = ROOT / "config" / "config.yaml"
    if target.exists():
        data = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    else:
        data = yaml.safe_load(
            (ROOT / "config" / "config.example.yaml").read_text(encoding="utf-8")
        ) or {}

    data.setdefault("auth", {})["user_id"] = user_id
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
