"""Bundled local MCP server for Mini-Agent.

This server provides a few zero-dependency tools so MCP works out of the box
in a fresh Python environment without requiring Node.js or extra API keys.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP


mcp = FastMCP(
    "mini_agent_local",
    instructions="Local utility tools bundled with Mini-Agent for quick MCP verification.",
)


def _workspace_root() -> Path:
    return Path.cwd().resolve()


def _resolve_workspace_path(path: str) -> Path:
    root = _workspace_root()
    candidate = (root / path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("Path must stay within the current workspace.") from exc
    return candidate


def _display_path(path: Path) -> str:
    root = _workspace_root()
    if path == root:
        return "."
    return str(path.relative_to(root))


@mcp.tool(name="local_time")
def local_time() -> str:
    """Return the current local timestamp on this machine."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


@mcp.tool(name="list_workspace")
def list_workspace(path: str = ".", limit: int = 20) -> str:
    """List files and directories under a workspace-relative path."""
    target = _resolve_workspace_path(path)
    capped_limit = max(1, min(limit, 100))

    if not target.exists():
        return f"Path not found: {path}"

    if target.is_file():
        size = target.stat().st_size
        return f"FILE {_display_path(target)} ({size} bytes)"

    entries = sorted(target.iterdir(), key=lambda entry: (entry.is_file(), entry.name.lower()))
    lines = [f"Directory: {_display_path(target)}"]

    for entry in entries[:capped_limit]:
        kind = "DIR " if entry.is_dir() else "FILE"
        suffix = "/" if entry.is_dir() else ""
        lines.append(f"{kind} {_display_path(entry)}{suffix}")

    remaining = len(entries) - capped_limit
    if remaining > 0:
        lines.append(f"... {remaining} more entries")

    return "\n".join(lines)


@mcp.tool(name="read_text_head")
def read_text_head(path: str, max_chars: int = 1200) -> str:
    """Read a short UTF-8 preview from a workspace-relative text file."""
    target = _resolve_workspace_path(path)
    if not target.exists() or not target.is_file():
        return f"File not found: {path}"

    capped_chars = max(1, min(max_chars, 8000))

    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"File is not valid UTF-8 text: {path}"

    if len(text) <= capped_chars:
        return text

    return text[:capped_chars] + "\n...[truncated]"


if __name__ == "__main__":
    mcp.run()
