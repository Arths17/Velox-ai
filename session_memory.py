"""Session persistence and background dream consolidation for Velox."""
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from config import CONFIG_DIR, SESSIONS_DIR

DREAM_DIR = CONFIG_DIR / "dream"
DREAM_LOCKS: dict[str, threading.Lock] = {}
_DREAM_GUARD = threading.Lock()


class SessionStateLike(Protocol):
    session_id: str
    started_at: str
    turn_count: int
    total_input_tokens: int
    total_output_tokens: int
    last_saved_at: str
    last_dream_turn: int
    messages: list[dict[str, Any]]


def _safe_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            blocks = []
            for b in content:
                if isinstance(b, dict):
                    blocks.append(b)
                else:
                    blocks.append(getattr(b, "model_dump", lambda: str(b))())
            out.append({**m, "content": blocks})
        else:
            out.append(m)
    return out


def persist_session(state: SessionStateLike, reason: str = "turn") -> Path:
    """Persist current in-memory state to a deterministic session file."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = SESSIONS_DIR / f"{state.session_id}.json"
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "session_id": state.session_id,
        "started_at": state.started_at,
        "last_saved_at": now,
        "reason": reason,
        "turn_count": state.turn_count,
        "total_input_tokens": state.total_input_tokens,
        "total_output_tokens": state.total_output_tokens,
        "last_dream_turn": getattr(state, "last_dream_turn", 0),
        "messages": _safe_messages(state.messages),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    state.last_saved_at = now
    (SESSIONS_DIR / "latest_session_id.txt").write_text(state.session_id)
    return path


def hydrate_state(state: SessionStateLike, data: dict[str, Any]) -> None:
    """Load persisted fields onto an AgentState instance."""
    state.messages = data.get("messages", [])
    state.turn_count = int(data.get("turn_count", 0))
    state.total_input_tokens = int(data.get("total_input_tokens", 0))
    state.total_output_tokens = int(data.get("total_output_tokens", 0))
    state.session_id = data.get("session_id") or state.session_id
    state.started_at = data.get("started_at") or state.started_at
    state.last_saved_at = data.get("last_saved_at", "")
    state.last_dream_turn = int(data.get("last_dream_turn", 0))


def _dream_lock(session_id: str) -> threading.Lock:
    with _DREAM_GUARD:
        if session_id not in DREAM_LOCKS:
            DREAM_LOCKS[session_id] = threading.Lock()
        return DREAM_LOCKS[session_id]


def _render_dream_note(messages: list[dict[str, Any]], keep_recent: int) -> str:
    older = messages[:-keep_recent] if len(messages) > keep_recent else []
    if not older:
        return ""

    bullets: list[str] = []
    for m in older[-24:]:
        role = m.get("role", "unknown")
        text = str(m.get("content", "")).replace("\n", " ").strip()
        if not text:
            continue
        bullets.append(f"- [{role}] {text[:220]}")

    if not bullets:
        return ""

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    body = "\n".join(bullets)
    return f"## Consolidation {stamp}\n\n{body}\n"


def maybe_schedule_dream(state: SessionStateLike, config: dict[str, Any]) -> str | None:
    """Schedule a best-effort background memory consolidation job."""
    if not config.get("dream_mode", True):
        return None

    min_turns = int(config.get("dream_min_turns", 6))
    interval = int(config.get("dream_interval_turns", 4))
    keep_recent = int(config.get("dream_keep_recent_messages", 14))

    if state.turn_count < min_turns:
        return None
    if state.turn_count - getattr(state, "last_dream_turn", 0) < interval:
        return None

    sid = state.session_id
    lock = _dream_lock(sid)
    if lock.locked():
        return "running"

    snapshot = list(state.messages)
    state.last_dream_turn = state.turn_count

    def _job() -> None:
        if not lock.acquire(blocking=False):
            return
        try:
            note = _render_dream_note(snapshot, keep_recent)
            if not note:
                return
            DREAM_DIR.mkdir(parents=True, exist_ok=True)
            path = DREAM_DIR / f"{sid}.md"
            prev = path.read_text() if path.exists() else ""
            path.write_text((prev + "\n" + note).strip() + "\n")
        finally:
            lock.release()

    threading.Thread(target=_job, name=f"dream-{sid[:8]}", daemon=True).start()
    return "scheduled"
