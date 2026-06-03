from __future__ import annotations

import io
import os
import shlex
import subprocess
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Callable, Iterable, List, Optional

from src.agent.models import ToolResult
from src.agent.permissions import PermissionPolicy


SAFE_GIT_COMMANDS = {"status", "diff", "log", "show", "branch"}


class ToolRuntime:
    """Runtime for tools available to the terminal agent."""

    def __init__(
            self,
            repo_root: Path,
            permission_policy: Optional[PermissionPolicy] = None,
            cli_runner: Optional[Callable[[List[str]], int]] = None,
    ):
        self.repo_root = Path(repo_root).resolve()
        self.permission_policy = permission_policy or PermissionPolicy()
        self._cli_runner = cli_runner

    def run_cli(self, argv: Iterable[str]) -> ToolResult:
        args = [str(part) for part in argv if str(part)]
        command = "python cli.py " + " ".join(shlex.quote(part) for part in args)
        decision = self.permission_policy.check_cli(args)
        if not decision.allowed:
            return ToolResult("cli", command, 130, stderr=decision.reason, blocked=True)

        stdout = io.StringIO()
        stderr = io.StringIO()
        previous_cwd = Path.cwd()
        try:
            os.chdir(self.repo_root)
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = self._run_cli(args)
        finally:
            os.chdir(previous_cwd)
        return ToolResult("cli", command, int(code or 0), stdout.getvalue(), stderr.getvalue())

    def run_bash(self, command: str) -> ToolResult:
        decision = self.permission_policy.check_shell(command)
        if not decision.allowed:
            return ToolResult("bash", command, 130, stderr=decision.reason, blocked=True)

        args = shlex.split(command)
        try:
            completed = subprocess.run(
                args,
                cwd=self.repo_root,
                text=True,
                capture_output=True,
                check=False,
            )
        except FileNotFoundError as exc:
            return ToolResult("bash", command, 127, stderr=str(exc))
        return ToolResult("bash", command, completed.returncode, completed.stdout, completed.stderr)

    def read_file(self, path: str, max_bytes: int = 20000) -> ToolResult:
        try:
            file_path = self._resolve_repo_path(path)
        except ValueError as exc:
            return ToolResult("read", path, 1, stderr=str(exc))
        if not file_path.exists() or not file_path.is_file():
            return ToolResult("read", path, 1, stderr=f"File not found: {path}")
        content = file_path.read_text(encoding="utf-8", errors="replace")
        truncated = content[:max_bytes]
        if len(content) > max_bytes:
            truncated += "\n\n[truncated]"
        return ToolResult("read", path, 0, stdout=truncated)

    def list_files(self, path: str = ".") -> ToolResult:
        try:
            base = self._resolve_repo_path(path)
        except ValueError as exc:
            return ToolResult("list", path, 1, stderr=str(exc))
        if not base.exists():
            return ToolResult("list", path, 1, stderr=f"Path not found: {path}")
        if base.is_file():
            return ToolResult("list", path, 0, stdout=str(base.relative_to(self.repo_root)))
        entries = sorted(item.relative_to(self.repo_root).as_posix() for item in base.iterdir())
        return ToolResult("list", path, 0, stdout="\n".join(entries))

    def grep(self, pattern: str, path: str = ".") -> ToolResult:
        try:
            base = self._resolve_repo_path(path)
        except ValueError as exc:
            return ToolResult("grep", f"grep {pattern} {path}", 1, stderr=str(exc))
        command = ["rg", "-n", pattern, str(base)]
        try:
            completed = subprocess.run(
                command,
                cwd=self.repo_root,
                text=True,
                capture_output=True,
                check=False,
            )
        except FileNotFoundError:
            completed = subprocess.run(
                ["grep", "-R", "-n", pattern, str(base)],
                cwd=self.repo_root,
                text=True,
                capture_output=True,
                check=False,
            )
        return ToolResult("grep", f"grep {pattern} {path}", completed.returncode, completed.stdout, completed.stderr)

    def write_file(self, path: str, content: str, confirmed: bool = False) -> ToolResult:
        if not confirmed:
            return ToolResult("write", path, 130, stderr="Writing files requires explicit confirmation.", blocked=True)
        try:
            file_path = self._resolve_repo_path(path)
        except ValueError as exc:
            return ToolResult("write", path, 1, stderr=str(exc))
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return ToolResult("write", path, 0, stdout=f"Wrote {file_path.relative_to(self.repo_root)}")

    def git(self, argv: Iterable[str]) -> ToolResult:
        args = [str(part) for part in argv if str(part)]
        if not args:
            return ToolResult("git", "git", 1, stderr="Missing git subcommand.")
        if args[0] not in SAFE_GIT_COMMANDS:
            return ToolResult("git", "git " + " ".join(args), 130, stderr="Only safe git commands are allowed.", blocked=True)
        return self.run_bash("git " + " ".join(shlex.quote(part) for part in args))

    def _run_cli(self, args: List[str]) -> int:
        if self._cli_runner is not None:
            return self._cli_runner(args)
        from src.cli.app import run

        return run(args)

    def _resolve_repo_path(self, path: str) -> Path:
        candidate = (self.repo_root / path).resolve()
        if candidate == self.repo_root or self.repo_root in candidate.parents:
            return candidate
        raise ValueError(f"Path escapes repository root: {path}")
