from __future__ import annotations

from typing import Iterable, List, Tuple

try:
    from prompt_toolkit.completion import Completer, Completion
except ImportError:  # pragma: no cover - exercised only on minimal environments.
    Completer = object
    Completion = None


class AgentCommandCompleter(Completer):
    """Autocomplete slash commands and skill invocations in agent mode."""

    def __init__(self, slash_commands: Iterable[str], skill_names: Iterable[str]):
        self.slash_commands = _unique(slash_commands)
        self.skill_names = sorted(_unique(skill_names), key=str.lower)

    def complete_line(self, line_before_cursor: str) -> List[Tuple[str, int, str]]:
        command_line = line_before_cursor.lstrip()
        if not command_line.startswith("/"):
            return []

        if command_line.startswith("/skill "):
            return self._complete_skill_name(command_line)

        if " " in command_line:
            return []

        return self._complete_slash_command(command_line)

    def get_completions(self, document, complete_event):
        if Completion is None:
            return

        for text, start_position, display_meta in self.complete_line(document.current_line_before_cursor):
            yield Completion(text, start_position=start_position, display=text, display_meta=display_meta)

    def _complete_slash_command(self, prefix: str) -> List[Tuple[str, int, str]]:
        return [
            (command, -len(prefix), "command")
            for command in self.slash_commands
            if _matches_prefix(command, prefix)
        ]

    def _complete_skill_name(self, command_line: str) -> List[Tuple[str, int, str]]:
        prefix = command_line[len("/skill "):]
        if " " in prefix:
            return []

        return [
            (skill_name, -len(prefix), "skill")
            for skill_name in self.skill_names
            if _matches_prefix(skill_name, prefix)
        ]


def _unique(values: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _matches_prefix(value: str, prefix: str) -> bool:
    return value.lower().startswith(prefix.lower())
