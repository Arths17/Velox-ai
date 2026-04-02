"""Configuration management for nano claude (multi-provider)."""
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false
import os
import json
from pathlib import Path
from typing import Any

CONFIG_DIR   = Path.home() / ".nano_claude"
CONFIG_FILE  = CONFIG_DIR  / "config.json"
HISTORY_FILE = CONFIG_DIR  / "input_history.txt"
SESSIONS_DIR = CONFIG_DIR  / "sessions"

DEFAULTS: dict[str, Any] = {
    "model":            "openrouter/openai/gpt-4o-mini",
    "max_tokens":       8192,
    "permission_mode":  "auto",   # auto | accept-all | manual
    "verbose":          False,
    "thinking":         False,
    "thinking_budget":  10000,
    "max_agent_turns":  12,
    "dream_mode":       True,
    "dream_min_turns":  6,
    "dream_interval_turns": 4,
    "dream_keep_recent_messages": 14,
    "custom_base_url":  "",       # for "custom" provider
    # Per-provider API keys (optional; env vars take priority)
    # "anthropic_api_key": "sk-ant-..."
    # "openai_api_key":    "sk-..."
    # "gemini_api_key":    "..."
    # "kimi_api_key":      "..."
    # "qwen_api_key":      "..."
    # "zhipu_api_key":     "..."
    # "deepseek_api_key":  "..."
}


def _load_dotenv_into_env() -> None:
    """Load .env from cwd as best-effort local overrides."""
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return
    try:
        for raw in env_path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            key = k.strip()
            val = v.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception:
        pass


def load_config() -> dict[str, Any]:
    CONFIG_DIR.mkdir(exist_ok=True)
    SESSIONS_DIR.mkdir(exist_ok=True)
    _load_dotenv_into_env()
    cfg = dict(DEFAULTS)
    if CONFIG_FILE.exists():
        try:
            cfg.update(json.loads(CONFIG_FILE.read_text()))
        except Exception:
            pass
    # Backward-compat: legacy single api_key → anthropic_api_key
    if cfg.get("api_key") and not cfg.get("anthropic_api_key"):
        cfg["anthropic_api_key"] = cfg.pop("api_key")
    # Also accept ANTHROPIC_API_KEY env for backward-compat
    if not cfg.get("anthropic_api_key"):
        cfg["anthropic_api_key"] = os.environ.get("ANTHROPIC_API_KEY", "")
    return cfg


def save_config(cfg: dict[str, Any]) -> None:
    CONFIG_DIR.mkdir(exist_ok=True)
    data = dict(cfg)
    CONFIG_FILE.write_text(json.dumps(data, indent=2))


def current_provider(cfg: dict[str, Any]) -> str:
    from providers import detect_provider
    return detect_provider(cfg.get("model", "claude-opus-4-6"))


def has_api_key(cfg: dict[str, Any]) -> bool:
    """Check whether the active provider has an API key configured."""
    from providers import get_api_key
    pname = current_provider(cfg)
    key = get_api_key(pname, cfg)
    return bool(key)


def calc_cost(model: str, in_tokens: int, out_tokens: int) -> float:
    from providers import calc_cost as _cc
    return _cc(model, in_tokens, out_tokens)
