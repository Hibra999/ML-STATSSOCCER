from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional


RISKY_SHELL_COMMANDS = {"rm", "rmdir", "mv", "chmod", "chown", "dd", "mkfs", "truncate"}
RISKY_GIT_SUBCOMMANDS = {"reset", "checkout", "clean", "restore", "rebase"}
RISKY_WORDS = {"delete", "remove", "destroy", "purge"}
RISKY_CLI_PAIRS = {
    ("league", "delete"),
    ("model", "delete"),
}


@dataclass
class PermissionDecision:
    allowed: bool
    risky: bool = False
    reason: str = ""


class PermissionPolicy:
    """Small permission layer for commands launched from the agent."""

    def __init__(
            self,
            auto_confirm: bool = False,
            confirmer: Optional[Callable[[str], bool]] = None,
    ):
        self.auto_confirm = auto_confirm
        self.confirmer = confirmer

    def check_cli(self, argv: Iterable[str]) -> PermissionDecision:
        parts = [part for part in argv if part]
        if is_risky_cli(parts):
            return self._confirm(f"Risky CLI command: {' '.join(shlex.quote(part) for part in parts)}")
        return PermissionDecision(allowed=True)

    def check_shell(self, command: str) -> PermissionDecision:
        if has_shell_metacharacters(command):
            return PermissionDecision(
                allowed=False,
                risky=True,
                reason="Shell operators are not allowed by the MVP agent. Run a single command with arguments.",
            )

        try:
            parts = shlex.split(command)
        except ValueError as exc:
            return PermissionDecision(allowed=False, reason=str(exc))

        if not parts:
            return PermissionDecision(allowed=False, reason="Empty command.")
        if is_risky_shell(parts):
            return self._confirm(f"Risky shell command: {command}")
        return PermissionDecision(allowed=True)

    def _confirm(self, message: str) -> PermissionDecision:
        if self.auto_confirm:
            return PermissionDecision(allowed=True, risky=True)
        if self.confirmer and self.confirmer(message):
            return PermissionDecision(allowed=True, risky=True)
        return PermissionDecision(allowed=False, risky=True, reason="Command requires explicit confirmation.")


def is_risky_cli(argv: List[str]) -> bool:
    if len(argv) >= 2 and (argv[0], argv[1]) in RISKY_CLI_PAIRS:
        return True
    lowered = [part.lower() for part in argv]
    if any(word in RISKY_WORDS for word in lowered):
        return True
    return False


def is_risky_shell(argv: List[str]) -> bool:
    command = argv[0].lower()
    if command in RISKY_SHELL_COMMANDS:
        return True
    if command == "git" and len(argv) > 1 and argv[1].lower() in RISKY_GIT_SUBCOMMANDS:
        return True
    lowered = [part.lower() for part in argv]
    return any(word in RISKY_WORDS for word in lowered)


def has_shell_metacharacters(command: str) -> bool:
    return any(token in command for token in ["|", "&&", "||", ";", "`", "$(", ">", "<"])
