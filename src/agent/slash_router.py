from __future__ import annotations

import shlex
from typing import Callable, Dict, List

from src.agent.models import RuntimeResponse
from src.agent.prompts import HELP_TEXT


CLI_DELEGATE_COMMANDS = {"league", "data", "model", "predict", "analysis", "explain", "config", "resources"}
SLASH_COMMANDS = [
    "/help",
    "/exit",
    "/quit",
    "/clear",
    "/status",
    "/context",
    "/skills",
    "/skill",
    "/reload-skills",
    "/history",
    "/compact",
    "/run",
    "/league",
    "/data",
    "/model",
    "/predict",
    "/analysis",
    "/explain",
    "/config",
    "/resources",
]


class SlashRouter:
    """Dispatches slash commands for AgentRuntime."""

    def __init__(self, runtime):
        self.runtime = runtime
        self._handlers: Dict[str, Callable[[List[str]], RuntimeResponse]] = {
            "help": self._help,
            "exit": self._exit,
            "quit": self._exit,
            "clear": self._clear,
            "status": self._status,
            "context": self._context,
            "skills": self._skills,
            "skill": self._skill,
            "reload-skills": self._reload_skills,
            "history": self._history,
            "compact": self._compact,
            "run": self._run,
        }

    @property
    def slash_commands(self) -> List[str]:
        return list(SLASH_COMMANDS)

    def route(self, line: str) -> RuntimeResponse:
        try:
            parts = shlex.split(line[1:])
        except ValueError as exc:
            return RuntimeResponse(f"Invalid command syntax: {exc}")
        if not parts:
            return RuntimeResponse(HELP_TEXT)

        command, args = parts[0], parts[1:]
        if command in CLI_DELEGATE_COMMANDS:
            return self.runtime.run_cli([command] + args)
        handler = self._handlers.get(command)
        if handler is None:
            return RuntimeResponse(f"Unknown slash command: /{command}. Use /help.")
        return handler(args)

    def _help(self, args: List[str]) -> RuntimeResponse:
        return RuntimeResponse(HELP_TEXT)

    def _exit(self, args: List[str]) -> RuntimeResponse:
        return RuntimeResponse("Leaving agent mode.", exit_requested=True)

    def _clear(self, args: List[str]) -> RuntimeResponse:
        return RuntimeResponse("", clear_screen=True)

    def _status(self, args: List[str]) -> RuntimeResponse:
        return RuntimeResponse(self.runtime.status_text())

    def _context(self, args: List[str]) -> RuntimeResponse:
        return RuntimeResponse(self.runtime.context_text())

    def _skills(self, args: List[str]) -> RuntimeResponse:
        return RuntimeResponse(self.runtime.skills_text())

    def _skill(self, args: List[str]) -> RuntimeResponse:
        if not args:
            return RuntimeResponse("Usage: /skill <name> [args...]")
        return self.runtime.invoke_skill(args[0], args[1:])

    def _reload_skills(self, args: List[str]) -> RuntimeResponse:
        self.runtime.skills.reload()
        return RuntimeResponse(f"Reloaded {len(self.runtime.skills.list_user_invocable())} skills.")

    def _history(self, args: List[str]) -> RuntimeResponse:
        return RuntimeResponse(self.runtime.session.history_text())

    def _compact(self, args: List[str]) -> RuntimeResponse:
        summary = self.runtime.session.compact()
        if not summary:
            return RuntimeResponse("Nothing to compact yet.")
        return RuntimeResponse("Compacted older session messages into the session summary.")

    def _run(self, args: List[str]) -> RuntimeResponse:
        if not args:
            return RuntimeResponse("Usage: /run <cli command | safe shell command>")
        if args[0] in CLI_DELEGATE_COMMANDS:
            return self.runtime.run_cli(args)
        return self.runtime.run_bash(" ".join(shlex.quote(part) for part in args))
