# Velox AI

Frontier-Grade Terminal Agent for engineering workflows.

Velox AI is a model-agnostic, privacy-first terminal coding agent designed for high-speed iteration, controlled autonomy, and local-first execution. It supports multiple model providers through a single runtime and keeps critical session intelligence on your machine.

## Why Velox

- Model-agnostic runtime: Anthropic, OpenAI-compatible APIs, OpenRouter, local endpoints.
- Privacy-first operation: local session files, explicit permission gates, and user-controlled persistence.
- Tool-native execution: read, write, edit, grep, glob, bash, web fetch/search.
- Production-inspired architecture: bounded agent loop, deterministic state persistence, and background memory consolidation.

## Core Capabilities

- Multi-turn autonomous execution with tool-use feedback loops.
- Safe permission model:
  - auto: read-only and safe commands auto-approved
  - manual: always ask
  - accept-all: no prompt (dangerous)
- Session durability:
  - autosave on completed turns
  - explicit export/import via slash commands
  - latest session pointer for continuity
- Dream Mode (background consolidation):
  - asynchronously compresses older conversation context into durable local memory notes
  - keeps live interaction fast while preserving long-horizon context

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

1. Set your provider key via environment variable.

```bash
export OPENROUTER_API_KEY="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"
```

2. Run interactive mode.

```bash
python nano_claude.py
```

3. Run one-shot mode.

```bash
python nano_claude.py --print "analyze this repository and propose a refactor plan"
```

## Provider Configuration

Velox defaults to:

- model: `openrouter/openai/gpt-4o-mini`

You can switch at runtime:

```text
/model openrouter/anthropic/claude-3.7-sonnet
/model gpt-4o
/model ollama/qwen2.5-coder
```

Supported key env vars include:

- `OPENROUTER_API_KEY`
- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `GEMINI_API_KEY` or `GOOGLE_API_KEY`
- `DEEPSEEK_API_KEY`
- `MOONSHOT_API_KEY`
- `DASHSCOPE_API_KEY`
- `ZHIPU_API_KEY`

Velox also loads local `.env` values from the current project directory.

## Session & Memory Design

Velox stores runtime artifacts under `~/.nano_claude/`:

- `sessions/`: persistent session snapshots
- `sessions/latest_session_id.txt`: latest known session pointer
- `dream/`: background consolidation notes

This design is optimized for:

- restart resilience
- low-latency turn execution
- local control of sensitive context

## Dream Mode

Dream Mode is a background consolidation pipeline inspired by high-performance agent systems:

- non-blocking execution in a daemon thread
- per-session lock to prevent overlapping runs
- turn-based scheduling thresholds
- durable append-only memory notes

Controls:

```text
/dream
/dream on
/dream off
```

## Slash Commands

- `/help`
- `/clear`
- `/model [name]`
- `/config [key=value]`
- `/save [path]`
- `/load [path]`
- `/history`
- `/dream [on|off]`
- `/context`
- `/cost`
- `/verbose`
- `/thinking`
- `/permissions [auto|accept-all|manual]`
- `/cwd [path]`
- `/exit`

## Security Notes

- Keep API keys in environment variables, not source files.
- `.env` is ignored by git in this project.
- Source-map reference datasets are ignored by git to avoid accidental publication.
- Use `manual` permission mode for high-sensitivity repos.

## Architecture Highlights

Velox adopts proven patterns from modern terminal-agent systems while remaining lightweight:

- Agent Loop:
  - streaming tool-aware loop with explicit turn budget (`max_agent_turns`)
  - deterministic message ledger across user/assistant/tool roles
- Prompt Efficiency:
  - CLAUDE.md prompt component caching by working-directory and mtime
  - environment loading and model routing decisions per turn
- Session Persistence:
  - autosave on completed turns
  - compact JSON snapshots for deterministic replay
- Background Consolidation:
  - asynchronous Dream Mode with session-level lock and threshold scheduling

## Project Philosophy

Velox AI is built for engineers who want frontier-grade capability without surrendering control.

- Run from the terminal.
- Keep memory local.
- Choose your model.
- Move fast with guardrails.
