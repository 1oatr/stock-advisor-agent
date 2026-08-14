"""src/llm_config.py — 全局 LLM 配置（唯一事实来源）

LLM 开关对整个软件生效：关闭后所有 LLM 调用点一律禁止访问 LLM API，
软件完全运行本地逻辑（规则引擎 / 11 技能 LocalFusion / RL / 关键词舆情）。

涉及的 LLM 调用点（全部必须先检查 llm_enabled()）：
  - agent/planner.py    工具调度（聊天时 DeepSeek 理解意图）
  - agent/core.py       AgentCore 开关（CLI/聊天共用）
  - tools/llm_skills_tool.py  llm_analyze 深度分析
  - tools/predict_tool.py     三路融合中的 LLM 路
  - skills/news_sentiment.py  新闻舆情（有 key 才调 LLM）

配置持久化: data/llm_config.json  {api_base, model, api_key, enabled}
  - enabled 未显式保存过时，默认 = 是否持有 API Key（与历史行为一致）
"""

import json
import os
from pathlib import Path
from typing import Optional

_DEFAULT_API_BASE = "https://api.deepseek.com"
_DEFAULT_MODEL = "deepseek-v4-flash"
_OVERRIDE_PATH: Optional[str] = None  # 测试可注入


def _config_path() -> Path:
    if _OVERRIDE_PATH:
        return Path(_OVERRIDE_PATH)
    return Path(__file__).resolve().parent.parent / "data" / "llm_config.json"


def set_config_path(path: str):
    """注入配置文件路径（仅测试用）"""
    global _OVERRIDE_PATH
    _OVERRIDE_PATH = path


def _read_saved() -> dict:
    try:
        path = _config_path()
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                saved = json.load(f) or {}
            return saved
    except (json.JSONDecodeError, IOError, OSError):
        pass
    return {}


def llm_config() -> dict:
    """完整配置视图（键与 WebUI services.get_llm_config 历史版本一致）"""
    saved = _read_saved()
    api_key = (saved.get("api_key") or "").strip()
    has_key = bool(api_key or os.getenv("DEEPSEEK_API_KEY"))
    return {
        "api_base": (saved.get("api_base") or _DEFAULT_API_BASE).strip(),
        "model": (saved.get("model") or _DEFAULT_MODEL).strip(),
        "api_key": api_key,
        "enabled": saved["enabled"] if "enabled" in saved else has_key,
        "has_api_key": has_key,
    }


def llm_enabled() -> bool:
    """全局 LLM 开关。关闭时任何模块都禁止调用 LLM API。"""
    return bool(llm_config()["enabled"])


def llm_enabled_explicit() -> Optional[bool]:
    """用户是否在 data/llm_config.json 显式保存过 enabled 开关。

    Returns:
        True/False 若显式保存过；None 若从未保存（走默认逻辑）。
    """
    saved = _read_saved()
    return saved["enabled"] if "enabled" in saved else None


def llm_api_key() -> str:
    """优先取配置保存的 key，其次环境变量 DEEPSEEK_API_KEY"""
    return llm_config()["api_key"] or os.getenv("DEEPSEEK_API_KEY") or ""


def llm_api_base() -> str:
    return llm_config()["api_base"]


def llm_model() -> str:
    return llm_config()["model"]


def save_config(params: dict) -> dict:
    """保存配置到 data/llm_config.json，返回完整视图"""
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = _read_saved()
    for k in ("api_base", "model", "api_key", "enabled"):
        if k in params:
            v = params[k]
            cfg[k] = str(v).strip() if isinstance(v, str) else v
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return llm_config()
