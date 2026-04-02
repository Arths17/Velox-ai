"""
Multi-provider support for nano claude.

Supported providers:
  anthropic  — Claude (claude-opus-4-6, claude-sonnet-4-6, ...)
  openai     — GPT (gpt-4o, o3-mini, ...)
  gemini     — Google Gemini (gemini-2.0-flash, gemini-1.5-pro, ...)
  kimi       — Moonshot AI (moonshot-v1-8k/32k/128k)
  qwen       — Alibaba DashScope (qwen-max, qwen-plus, ...)
  zhipu      — Zhipu GLM (glm-4, glm-4-plus, ...)
  deepseek   — DeepSeek (deepseek-chat, deepseek-reasoner, ...)
  ollama     — Local Ollama (llama3.3, qwen2.5-coder, ...)
  lmstudio   — Local LM Studio (any loaded model)
  custom     — Any OpenAI-compatible endpoint

Model string formats:
  "claude-opus-4-6"          auto-detected → anthropic
  "gpt-4o"                   auto-detected → openai
  "ollama/qwen2.5-coder"     explicit provider prefix
  "custom/my-model"          uses CUSTOM_BASE_URL from config
"""
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportAttributeAccessIssue=false
from __future__ import annotations
import json
import re
from typing import Any, Generator

# ── Provider registry ──────────────────────────────────────────────────────

PROVIDERS: dict[str, dict[str, Any]] = {
    "anthropic": {
        "type":       "anthropic",
        "api_key_env": "ANTHROPIC_API_KEY",
        "models": [
            "claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001",
            "claude-opus-4-5", "claude-sonnet-4-5",
            "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022",
        ],
    },
    "openai": {
        "type":       "openai",
        "api_key_env": "OPENAI_API_KEY",
        "base_url":   "https://api.openai.com/v1",
        "models": [
            "gpt-4o", "gpt-4o-mini", "gpt-4-turbo",
            "o3-mini", "o1", "o1-mini",
        ],
    },
    "gemini": {
        "type":       "openai",
        "api_key_env": "GEMINI_API_KEY",
        "base_url":   "https://generativelanguage.googleapis.com/v1beta/openai/",
        "models": [
            "gemini-2.5-pro-preview-03-25",
            "gemini-2.0-flash", "gemini-2.0-flash-lite",
            "gemini-1.5-pro", "gemini-1.5-flash",
        ],
    },
    "openrouter": {
        "type":       "openai",
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url":   "https://openrouter.ai/api/v1",
        "models": [
            "openai/gpt-4o-mini",
            "openai/gpt-4.1-mini",
            "anthropic/claude-3.7-sonnet",
            "google/gemini-2.0-flash-001",
        ],
    },
    "kimi": {
        "type":       "openai",
        "api_key_env": "MOONSHOT_API_KEY",
        "base_url":   "https://api.moonshot.cn/v1",
        "models": [
            "moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k",
            "kimi-latest",
        ],
    },
    "qwen": {
        "type":       "openai",
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url":   "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "models": [
            "qwen-max", "qwen-plus", "qwen-turbo", "qwen-long",
            "qwen2.5-72b-instruct", "qwen2.5-coder-32b-instruct",
            "qwq-32b",
        ],
    },
    "zhipu": {
        "type":       "openai",
        "api_key_env": "ZHIPU_API_KEY",
        "base_url":   "https://open.bigmodel.cn/api/paas/v4/",
        "models": [
            "glm-4-plus", "glm-4", "glm-4-flash", "glm-4-air",
            "glm-z1-flash",
        ],
    },
    "deepseek": {
        "type":       "openai",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url":   "https://api.deepseek.com/v1",
        "models": [
            "deepseek-chat", "deepseek-coder", "deepseek-reasoner",
        ],
    },
    "ollama": {
        "type":       "openai",
        "api_key_env": None,
        "base_url":   "http://localhost:11434/v1",
        "api_key":    "ollama",
        "models": [
            "llama3.3", "llama3.2", "phi4", "mistral", "mixtral",
            "qwen2.5-coder", "deepseek-r1", "gemma3",
        ],
    },
    "lmstudio": {
        "type":       "openai",
        "api_key_env": None,
        "base_url":   "http://localhost:1234/v1",
        "api_key":    "lm-studio",
        "models": [],   # dynamic, depends on loaded model
    },
    "custom": {
        "type":       "openai",
        "api_key_env": "CUSTOM_API_KEY",
        "base_url":   None,   # read from config["custom_base_url"]
        "models": [],
    },
}

# Cost per million tokens (approximate, fallback to 0 for unknown)
COSTS = {
    "claude-opus-4-6":          (15.0, 75.0),
    "claude-sonnet-4-6":        (3.0,  15.0),
    "claude-haiku-4-5-20251001": (0.8,  4.0),
    "gpt-4o":                   (2.5,  10.0),
    "gpt-4o-mini":              (0.15,  0.6),
    "o3-mini":                  (1.1,   4.4),
    "gemini-2.0-flash":         (0.075, 0.3),
    "gemini-1.5-pro":           (1.25,  5.0),
    "gemini-2.5-pro-preview-03-25": (1.25, 10.0),
    "moonshot-v1-8k":           (1.0,   3.0),
    "moonshot-v1-32k":          (2.4,   7.0),
    "moonshot-v1-128k":         (8.0,  24.0),
    "qwen-max":                 (2.4,   9.6),
    "qwen-plus":                (0.4,   1.2),
    "deepseek-chat":            (0.27,  1.1),
    "deepseek-reasoner":        (0.55,  2.19),
    "glm-4-plus":               (0.7,   0.7),
}

ROUTER_FAST_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
}

ROUTER_REASONING_MODELS = {
    "openai": "o1",
    "anthropic": "claude-opus-4-6",
}

# Auto-detection: prefix → provider name
_PREFIXES = [
    ("claude-",       "anthropic"),
    ("gpt-",          "openai"),
    ("o1",            "openai"),
    ("o3",            "openai"),
    ("gemini-",       "gemini"),
    ("moonshot-",     "kimi"),
    ("kimi-",         "kimi"),
    ("qwen",          "qwen"),  # qwen-max, qwen2.5-...
    ("qwq-",          "qwen"),
    ("glm-",          "zhipu"),
    ("deepseek-",     "deepseek"),
    ("openrouter/",   "openrouter"),
    ("llama",         "ollama"),
    ("mistral",       "ollama"),
    ("phi",           "ollama"),
    ("gemma",         "ollama"),
]


def detect_provider(model: str) -> str:
    """Return provider name for a model string.
    Supports 'provider/model' explicit format, or auto-detect by prefix."""
    if "/" in model:
        return model.split("/", 1)[0]
    for prefix, pname in _PREFIXES:
        if model.lower().startswith(prefix):
            return pname
    return "openai"   # fallback


def bare_model(model: str) -> str:
    """Strip 'provider/' prefix if present."""
    return model.split("/", 1)[1] if "/" in model else model


def get_api_key(provider_name: str, config: dict[str, Any]) -> str:
    prov = PROVIDERS.get(provider_name, {})
    # 1. Check config dict (e.g. config["kimi_api_key"])
    cfg_key = config.get(f"{provider_name}_api_key", "")
    if cfg_key:
        return cfg_key
    # 2. Check env var
    env_var = prov.get("api_key_env")
    if env_var:
        import os
        value = os.environ.get(env_var, "")
        if value:
            return value
        if provider_name == "gemini":
            return os.environ.get("GOOGLE_API_KEY", "")
        if provider_name == "openrouter":
            return os.environ.get("OR_API_KEY", "")
        return ""
    # 3. Hardcoded (for local providers)
    return prov.get("api_key", "")


def calc_cost(model: str, in_tok: int, out_tok: int) -> float:
    ic, oc = COSTS.get(bare_model(model), (0.0, 0.0))
    return (in_tok * ic + out_tok * oc) / 1_000_000


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
            else:
                text = getattr(block, "text", "")
                if text:
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def _latest_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return _message_text(message)
    return ""


def _complexity_score(messages: list[dict[str, Any]], system: str) -> float:
    text = f"{system}\n{_latest_user_text(messages)}".lower()
    score = 0.0
    length = len(text)
    score += min(length / 900.0, 3.0)

    keyword_weights = {
        "architect": 1.5,
        "architecture": 1.5,
        "refactor": 1.5,
        "migration": 1.5,
        "rag": 1.0,
        "vector": 1.0,
        "sandbox": 1.0,
        "security": 1.2,
        "performance": 1.0,
        "optimize": 1.0,
        "multi-file": 1.2,
        "cross-file": 1.2,
        "whole repo": 1.2,
        "bug": 0.8,
        "traceback": 0.8,
        "test": 0.7,
        "review": 0.7,
        "dependency": 1.0,
        "install": 0.7,
        "error": 0.8,
    }
    for keyword, weight in keyword_weights.items():
        if keyword in text:
            score += weight

    if re.search(r"\b\d+\s+files?\b", text):
        score += 1.0
    if len(messages) > 12:
        score += 0.8
    if text.count("\n") > 20:
        score += 0.5
    return score


def _available_model(model: str, config: dict[str, Any]) -> bool:
    provider = detect_provider(model)
    if PROVIDERS.get(provider, {}).get("type") == "openai" and provider not in ("ollama", "lmstudio", "custom"):
        return bool(get_api_key(provider, config))
    if provider == "anthropic":
        return bool(get_api_key(provider, config))
    return True


def resolve_router_model(model: str, messages: list[dict[str, Any]], system: str, config: dict[str, Any]) -> tuple[str, str]:
    requested = model
    if requested not in {"router", "auto", "auto-router"} and not config.get("router_enabled", False):
        return requested, "router disabled"

    complexity = _complexity_score(messages, system)
    simple_model = config.get("router_fast_model") or ROUTER_FAST_MODELS.get(
        detect_provider(config.get("model", requested)),
        "gpt-4o-mini",
    )
    reasoning_model = config.get("router_reasoning_model") or ROUTER_REASONING_MODELS.get(
        detect_provider(config.get("model", requested)),
        "claude-opus-4-6",
    )
    threshold = float(config.get("router_threshold", 3.0))
    chosen = reasoning_model if complexity >= threshold else simple_model

    if not _available_model(chosen, config):
        fallback = requested if requested not in {"router", "auto", "auto-router"} else config.get("model", chosen)
        if _available_model(fallback, config):
            return fallback, f"router fallback → {fallback}"
        other = simple_model if chosen == reasoning_model else reasoning_model
        if _available_model(other, config):
            return other, f"router fallback → {other}"
        return chosen, f"router unavailable keys; using {chosen}"

    return chosen, f"router score={complexity:.1f} threshold={threshold:.1f}"


# ── Tool schema conversion ─────────────────────────────────────────────────

def tools_to_openai(tool_schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic-style tool schemas to OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name":        t["name"],
                "description": t["description"],
                "parameters":  t["input_schema"],
            },
        }
        for t in tool_schemas
    ]


# ── Message format conversion ──────────────────────────────────────────────
#
# Internal "neutral" message format:
#   {"role": "user",      "content": "text"}
#   {"role": "assistant", "content": "text", "tool_calls": [
#       {"id": "...", "name": "...", "input": {...}}
#   ]}
#   {"role": "tool", "tool_call_id": "...", "name": "...", "content": "..."}

def messages_to_anthropic(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert neutral messages → Anthropic API format."""
    result = []
    i = 0
    while i < len(messages):
        m = messages[i]
        role = m["role"]

        if role == "user":
            result.append({"role": "user", "content": m["content"]})
            i += 1

        elif role == "assistant":
            blocks = []
            text = m.get("content", "")
            if text:
                blocks.append({"type": "text", "text": text})
            for tc in m.get("tool_calls", []):
                blocks.append({
                    "type":  "tool_use",
                    "id":    tc["id"],
                    "name":  tc["name"],
                    "input": tc["input"],
                })
            result.append({"role": "assistant", "content": blocks})
            i += 1

        elif role == "tool":
            # Collect consecutive tool results into one user message
            tool_blocks = []
            while i < len(messages) and messages[i]["role"] == "tool":
                t = messages[i]
                tool_blocks.append({
                    "type":        "tool_result",
                    "tool_use_id": t["tool_call_id"],
                    "content":     t["content"],
                })
                i += 1
            result.append({"role": "user", "content": tool_blocks})

        else:
            i += 1

    return result


def messages_to_openai(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert neutral messages → OpenAI API format."""
    result = []
    for m in messages:
        role = m["role"]

        if role == "user":
            result.append({"role": "user", "content": m["content"]})

        elif role == "assistant":
            msg: dict = {"role": "assistant", "content": m.get("content") or None}
            tcs = m.get("tool_calls", [])
            if tcs:
                msg["tool_calls"] = [
                    {
                        "id":   tc["id"],
                        "type": "function",
                        "function": {
                            "name":      tc["name"],
                            "arguments": json.dumps(tc["input"], ensure_ascii=False),
                        },
                    }
                    for tc in tcs
                ]
            result.append(msg)

        elif role == "tool":
            result.append({
                "role":         "tool",
                "tool_call_id": m["tool_call_id"],
                "content":      m["content"],
            })

    return result


# ── Streaming adapters ─────────────────────────────────────────────────────

class TextChunk:
    def __init__(self, text: str): self.text = text

class ThinkingChunk:
    def __init__(self, text: str): self.text = text

class AssistantTurn:
    """Completed assistant turn with text + tool_calls."""
    def __init__(self, text: str, tool_calls: list[dict[str, Any]], in_tokens: int, out_tokens: int):
        self.text        = text
        self.tool_calls  = tool_calls   # list of {id, name, input}
        self.in_tokens   = in_tokens
        self.out_tokens  = out_tokens


def stream_anthropic(
    api_key: str,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]],
    config: dict[str, Any],
) -> Generator[Any, None, None]:
    """Stream from Anthropic API. Yields TextChunk/ThinkingChunk, then AssistantTurn."""
    import anthropic as _ant
    client = _ant.Anthropic(api_key=api_key)

    kwargs = {
        "model":      model,
        "max_tokens": config.get("max_tokens", 8192),
        "system":     system,
        "messages":   messages_to_anthropic(messages),
        "tools":      tool_schemas,
    }
    if config.get("thinking"):
        kwargs["thinking"] = {
            "type":          "enabled",
            "budget_tokens": config.get("thinking_budget", 10000),
        }

    tool_calls: list[dict[str, Any]] = []
    text = ""

    with client.messages.stream(**kwargs) as stream:
        for event in stream:
            etype = getattr(event, "type", None)
            if etype == "content_block_delta":
                delta = event.delta
                dtype = getattr(delta, "type", None)
                if dtype == "text_delta":
                    text += delta.text
                    yield TextChunk(delta.text)
                elif dtype == "thinking_delta":
                    yield ThinkingChunk(delta.thinking)

        final = stream.get_final_message()
        for block in final.content:
            if block.type == "tool_use":
                tool_calls.append({
                    "id":    block.id,
                    "name":  block.name,
                    "input": block.input,
                })

        yield AssistantTurn(
            text, tool_calls,
            final.usage.input_tokens,
            final.usage.output_tokens,
        )


def stream_openai_compat(
    api_key: str,
    base_url: str,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]],
    config: dict[str, Any],
) -> Generator[Any, None, None]:
    """Stream from any OpenAI-compatible API. Yields TextChunk, then AssistantTurn."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key or "dummy", base_url=base_url)

    oai_messages = [{"role": "system", "content": system}] + messages_to_openai(messages)

    kwargs: dict = {
        "model":    model,
        "messages": oai_messages,
        "stream":   True,
    }
    if tool_schemas and not config.get("no_tools"):
        kwargs["tools"] = tools_to_openai(tool_schemas)
        # "auto" requires vLLM --enable-auto-tool-choice; omit if server doesn't support it
        if not config.get("disable_tool_choice"):
            kwargs["tool_choice"] = "auto"
    if config.get("max_tokens"):
        kwargs["max_tokens"] = config["max_tokens"]

    text = ""
    tool_buf: dict[int, dict[str, str]] = {}   # index → {id, name, args_str}
    in_tok = 0
    out_tok = 0

    stream = client.chat.completions.create(**kwargs)
    for chunk in stream:
        if not chunk.choices:
            # usage-only chunk (some providers send this last)
            if hasattr(chunk, "usage") and chunk.usage:
                in_tok  = chunk.usage.prompt_tokens
                out_tok = chunk.usage.completion_tokens
            continue

        choice = chunk.choices[0]
        delta  = choice.delta

        if delta.content:
            text += delta.content
            yield TextChunk(delta.content)

        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in tool_buf:
                    tool_buf[idx] = {"id": "", "name": "", "args": ""}
                if tc.id:
                    tool_buf[idx]["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        tool_buf[idx]["name"] += tc.function.name
                    if tc.function.arguments:
                        tool_buf[idx]["args"] += tc.function.arguments

        # Some providers include usage in the last chunk
        if hasattr(chunk, "usage") and chunk.usage:
            in_tok  = chunk.usage.prompt_tokens  or in_tok
            out_tok = chunk.usage.completion_tokens or out_tok

    tool_calls = []
    for idx in sorted(tool_buf):
        v = tool_buf[idx]
        try:
            inp = json.loads(v["args"]) if v["args"] else {}
        except json.JSONDecodeError:
            inp = {"_raw": v["args"]}
        tool_calls.append({"id": v["id"] or f"call_{idx}", "name": v["name"], "input": inp})

    yield AssistantTurn(text, tool_calls, in_tok, out_tok)


def stream(
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]],
    config: dict[str, Any],
) -> Generator[Any, None, None]:
    """
    Unified streaming entry point.
    Auto-detects provider from model string.
    Yields: TextChunk | ThinkingChunk | AssistantTurn
    """
    model, route_reason = resolve_router_model(model, messages, system, config)
    config["_active_model"] = model
    config["_model_route_reason"] = route_reason

    provider_name = detect_provider(model)
    model_name    = bare_model(model)
    prov          = PROVIDERS.get(provider_name, PROVIDERS["openai"])
    api_key       = get_api_key(provider_name, config)

    if prov["type"] == "anthropic":
        yield from stream_anthropic(api_key, model_name, system, messages, tool_schemas, config)
    else:
        import os as _os
        if provider_name == "custom":
            base_url = (config.get("custom_base_url")
                        or _os.environ.get("CUSTOM_BASE_URL", ""))
            if not base_url:
                raise ValueError(
                    "custom provider requires a base_url. "
                    "Set CUSTOM_BASE_URL env var or run: /config custom_base_url=http://..."
                )
        else:
            base_url = prov.get("base_url", "https://api.openai.com/v1")
        yield from stream_openai_compat(
            api_key, base_url, model_name, system, messages, tool_schemas, config
        )
