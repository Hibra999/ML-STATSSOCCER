from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class SessionState:
    session_id: str
    created_at: str
    updated_at: str
    messages: List[Dict[str, str]] = field(default_factory=list)
    commands: List[Dict[str, object]] = field(default_factory=list)
    active_skills: List[str] = field(default_factory=list)
    compact_summary: str = ""
    current_league: Optional[str] = None
    current_model: Optional[str] = None


class SessionStore:
    """JSON-backed session memory for the terminal agent."""

    def __init__(self, repo_root: Path, session_id: Optional[str] = None):
        self.repo_root = Path(repo_root).resolve()
        self.session_dir = self.repo_root / ".mlstatssoccer" / "sessions"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.state = self._load_or_create(session_id)

    @property
    def path(self) -> Path:
        return self.session_dir / f"{self.state.session_id}.json"

    def add_message(self, role: str, content: str) -> None:
        self.state.messages.append({"role": role, "content": content, "timestamp": utc_now()})
        self.save()

    def add_command(self, command: str, exit_code: int, tool: str = "cli") -> None:
        self.state.commands.append({"tool": tool, "command": command, "exit_code": exit_code, "timestamp": utc_now()})
        self.save()

    def add_active_skill(self, name: str) -> None:
        if name not in self.state.active_skills:
            self.state.active_skills.append(name)
            self.save()

    def set_current(self, league_id: Optional[str] = None, model_id: Optional[str] = None) -> None:
        if league_id is not None:
            self.state.current_league = league_id
        if model_id is not None:
            self.state.current_model = model_id
        self.save()

    def compact(self, keep_last: int = 12) -> str:
        old_messages = self.state.messages[:-keep_last]
        if old_messages:
            lines = [self.state.compact_summary] if self.state.compact_summary else []
            for message in old_messages:
                content = message.get("content", "").strip().replace("\n", " ")
                if content:
                    lines.append(f"{message.get('role', 'unknown')}: {content[:180]}")
            self.state.compact_summary = "\n".join(line for line in lines if line).strip()
            self.state.messages = self.state.messages[-keep_last:]
            self.save()
        return self.state.compact_summary

    def history_text(self, limit: int = 30) -> str:
        messages = self.state.messages[-limit:]
        if not messages:
            return "No session messages yet."
        return "\n".join(f"{item['role']}: {item['content']}" for item in messages)

    def status(self) -> Dict[str, object]:
        return {
            "session_id": self.state.session_id,
            "messages": len(self.state.messages),
            "commands": len(self.state.commands),
            "active_skills": self.state.active_skills,
            "current_league": self.state.current_league,
            "current_model": self.state.current_model,
            "summary_chars": len(self.state.compact_summary),
            "path": str(self.path),
        }

    def save(self) -> None:
        self.state.updated_at = utc_now()
        self.path.write_text(json.dumps(asdict(self.state), indent=2) + "\n", encoding="utf-8")

    def _load_or_create(self, session_id: Optional[str]) -> SessionState:
        if session_id:
            path = self.session_dir / f"{session_id}.json"
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                return SessionState(**data)
        now = utc_now()
        generated = session_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        state = SessionState(session_id=generated, created_at=now, updated_at=now)
        path = self.session_dir / f"{state.session_id}.json"
        suffix = 1
        while path.exists():
            state.session_id = f"{generated}-{suffix}"
            path = self.session_dir / f"{state.session_id}.json"
            suffix += 1
        return state
