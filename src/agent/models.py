from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Skill:
    """Skill loaded from a SKILL.md file."""

    key: str
    name: str
    description: str
    when_to_use: List[str]
    arguments: List[str]
    examples: List[str]
    allowed_tools: List[str]
    user_invocable: bool
    disable_model_invocation: bool
    path: Path
    body: str
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        return self.name or self.key

    @property
    def searchable_text(self) -> str:
        parts = [self.key, self.name, self.description, " ".join(self.when_to_use), " ".join(self.arguments)]
        return " ".join(part for part in parts if part).lower()


@dataclass
class ToolResult:
    """Result returned by an agent tool."""

    tool: str
    command: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    blocked: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.blocked

    def render_text(self) -> str:
        parts = []
        if self.stdout:
            parts.append(self.stdout.rstrip())
        if self.stderr:
            parts.append(self.stderr.rstrip())
        if not parts:
            parts.append(f"{self.tool} exited with code {self.exit_code}.")
        return "\n".join(parts)


@dataclass
class RuntimeResponse:
    """Response produced by the agent runtime."""

    message: str
    exit_requested: bool = False
    clear_screen: bool = False
    tool_result: Optional[ToolResult] = None


@dataclass
class ContextItem:
    """A file or command output injected into the current session context."""

    source: str
    content: str
