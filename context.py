"""System context: CLAUDE.md, git info, cwd injection."""
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Any

_CLAUDE_MD_CACHE: dict[str, Any] = {
    "cwd": "",
    "global_mtime": 0.0,
    "project_mtime": 0.0,
    "content": "",
}

SYSTEM_PROMPT_TEMPLATE = """\
You are Nano Claude Code, Created by SAIL Lab (Safe AI and Robot Learning Lab), an AI coding assistant running in the terminal.
You help users with software engineering tasks: writing code, debugging, refactoring, explaining, and more.

# Available Tools
- **Read**: Read file contents with line numbers
- **Write**: Create or overwrite files
- **Edit**: Replace text in a file (exact string replacement)
- **Bash**: Execute shell commands
- **Glob**: Find files by pattern (e.g. **/*.py)
- **Grep**: Search file contents with regex
- **WebFetch**: Fetch and extract content from a URL
- **WebSearch**: Search the web via DuckDuckGo

# Guidelines
- Be concise and direct. Lead with the answer.
- Prefer editing existing files over creating new ones.
- Do not add unnecessary comments, docstrings, or error handling.
- When reading files before editing, use line numbers to be precise.
- Always use absolute paths for file operations.
- For multi-step tasks, work through them systematically.
- If a task is unclear, ask for clarification before proceeding.

# Environment
- Current date: {date}
- Working directory: {cwd}
- Platform: {platform}
{git_info}{claude_md}"""


def get_git_info() -> str:
    """Return git branch/status summary if in a git repo."""
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
        status = subprocess.check_output(
            ["git", "status", "--short"],
            stderr=subprocess.DEVNULL, text=True).strip()
        log = subprocess.check_output(
            ["git", "log", "--oneline", "-5"],
            stderr=subprocess.DEVNULL, text=True).strip()
        parts = [f"- Git branch: {branch}"]
        if status:
            lines = status.split('\n')[:10]
            parts.append("- Git status:\n" + "\n".join(f"  {l}" for l in lines))
        if log:
            parts.append("- Recent commits:\n" + "\n".join(f"  {l}" for l in log.split('\n')))
        return "\n".join(parts) + "\n"
    except Exception:
        return ""


def get_claude_md() -> str:
    """Load CLAUDE.md from cwd or parents, and ~/.claude/CLAUDE.md."""
    content_parts = []

    # Global CLAUDE.md
    global_md = Path.home() / ".claude" / "CLAUDE.md"
    global_mtime = global_md.stat().st_mtime if global_md.exists() else 0.0
    if global_md.exists():
        try:
            content_parts.append(f"[Global CLAUDE.md]\n{global_md.read_text()}")
        except Exception:
            pass

    # Project CLAUDE.md (walk up from cwd)
    p = Path.cwd()
    project_file = None
    for _ in range(10):
        candidate = p / "CLAUDE.md"
        if candidate.exists():
            project_file = candidate
            try:
                content_parts.append(f"[Project CLAUDE.md: {candidate}]\n{candidate.read_text()}")
            except Exception:
                pass
            break
        parent = p.parent
        if parent == p:
            break
        p = parent

    project_mtime = project_file.stat().st_mtime if project_file and project_file.exists() else 0.0
    cwd = str(Path.cwd())

    if (
        _CLAUDE_MD_CACHE["cwd"] == cwd
        and _CLAUDE_MD_CACHE["global_mtime"] == global_mtime
        and _CLAUDE_MD_CACHE["project_mtime"] == project_mtime
    ):
        return str(_CLAUDE_MD_CACHE["content"])

    if not content_parts:
        content = ""
    else:
        content = "\n# Memory / CLAUDE.md\n" + "\n\n".join(content_parts) + "\n"

    _CLAUDE_MD_CACHE["cwd"] = cwd
    _CLAUDE_MD_CACHE["global_mtime"] = global_mtime
    _CLAUDE_MD_CACHE["project_mtime"] = project_mtime
    _CLAUDE_MD_CACHE["content"] = content
    return content


def build_system_prompt() -> str:
    import platform
    return SYSTEM_PROMPT_TEMPLATE.format(
        date=datetime.now().strftime("%Y-%m-%d %A"),
        cwd=str(Path.cwd()),
        platform=platform.system(),
        git_info=get_git_info(),
        claude_md=get_claude_md(),
    )
