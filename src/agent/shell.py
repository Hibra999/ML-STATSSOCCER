from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.agent.completion import AgentCommandCompleter
from src.agent.prompts import WELCOME_TEXT
from src.agent.runtime import AgentRuntime

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.shortcuts import CompleteStyle
except ImportError:  # pragma: no cover - exercised only on minimal environments.
    CompleteStyle = None
    PromptSession = None
    FileHistory = None
    KeyBindings = None

try:
    from rich.console import Console
    from rich.panel import Panel
except ImportError:  # pragma: no cover - project requirements include rich.
    Console = None
    Panel = None


class AgentShell:
    """Interactive terminal shell for the agent runtime."""

    def __init__(self, repo_root: Optional[Path] = None, runtime: Optional[AgentRuntime] = None):
        self.repo_root = Path(repo_root or Path.cwd()).resolve()
        self.console = Console() if Console else None
        self.runtime = runtime or AgentRuntime(repo_root=self.repo_root, confirmer=self._confirm)
        self._session = self._build_prompt_session()

    def run(self) -> int:
        self._print_panel(WELCOME_TEXT, title="Agent")
        while True:
            try:
                message = self._read_message()
            except (EOFError, KeyboardInterrupt):
                self._print("Leaving agent mode.")
                return 0

            response = self.runtime.handle_message(message)
            if response.clear_screen:
                self._clear()
            elif response.message:
                self._print_panel(response.message, title="Response")
            if response.exit_requested:
                return 0

    def _build_prompt_session(self):
        if PromptSession is None:
            return None
        history_path = self.repo_root / ".mlstatssoccer" / "agent_history"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        completer = AgentCommandCompleter(self.runtime.slash_commands, self.runtime.skill_names)
        session_options = {
            "history": FileHistory(str(history_path)),
            "completer": completer,
            "complete_while_typing": True,
            "reserve_space_for_menu": 8,
            "key_bindings": self._build_key_bindings(),
        }
        if CompleteStyle is not None:
            session_options["complete_style"] = CompleteStyle.COLUMN
        return PromptSession(**session_options)

    def _build_key_bindings(self):
        if KeyBindings is None:
            return None

        bindings = KeyBindings()

        @bindings.add("/")
        def _(event):
            buffer = event.current_buffer
            buffer.insert_text("/")
            line = buffer.document.current_line_before_cursor.lstrip()
            if line.startswith("/"):
                buffer.start_completion(select_first=False)

        return bindings

    def _read_message(self) -> str:
        if self._session is None:
            return input("mls-agent> ")

        chunks = []
        prompt = "mls-agent> "
        while True:
            line = self._session.prompt(prompt)
            if line.endswith("\\"):
                chunks.append(line[:-1])
                prompt = "        ... "
                continue
            chunks.append(line)
            return "\n".join(chunks)

    def _confirm(self, message: str) -> bool:
        self._print_panel(f"{message}\nType 'yes' to continue.", title="Confirmation")
        answer = input("confirm> ").strip().lower()
        return answer == "yes"

    def _clear(self) -> None:
        if self.console:
            self.console.clear()
        else:
            print("\033[2J\033[H", end="")

    def _print(self, message: str) -> None:
        if self.console:
            self.console.print(message)
        else:
            print(message)

    def _print_panel(self, message: str, title: str) -> None:
        if self.console and Panel:
            self.console.print(Panel(message, title=title, border_style="cyan"))
        else:
            print(f"== {title} ==")
            print(message)
