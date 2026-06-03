from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List

from src.agent.models import ContextItem


AT_REFERENCE_RE = re.compile(r"(?<!\S)@([^\s]+)")


class ProjectContext:
    """Builds lightweight project context for the agent runtime."""

    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root).resolve()

    def load_references(self, message: str, max_bytes: int = 20000) -> List[ContextItem]:
        items = []
        for raw_path in AT_REFERENCE_RE.findall(message):
            try:
                path = self._resolve(raw_path)
            except ValueError as exc:
                items.append(ContextItem(source=f"@{raw_path}", content=str(exc)))
                continue
            if path.exists() and path.is_file():
                content = path.read_text(encoding="utf-8", errors="replace")
                if len(content) > max_bytes:
                    content = content[:max_bytes] + "\n\n[truncated]"
                items.append(ContextItem(source=f"@{raw_path}", content=content))
            else:
                items.append(ContextItem(source=f"@{raw_path}", content=f"File not found: {raw_path}"))
        return items

    def describe(self, active_context: Iterable[ContextItem]) -> str:
        lines = [
            f"Repository: {self.repo_root}",
            "Context sources:",
        ]
        context = list(active_context)
        if not context:
            lines.append("- none loaded in this turn")
        else:
            for item in context:
                lines.append(f"- {item.source}: {len(item.content)} chars")
        return "\n".join(lines)

    def _resolve(self, path: str) -> Path:
        candidate = (self.repo_root / path).resolve()
        if candidate == self.repo_root or self.repo_root in candidate.parents:
            return candidate
        raise ValueError(f"Path escapes repository root: {path}")
