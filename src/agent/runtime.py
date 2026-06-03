from __future__ import annotations

import shlex
from difflib import get_close_matches
from pathlib import Path
from typing import Iterable, List, Optional

from src.agent.context import ProjectContext
from src.agent.models import ContextItem, RuntimeResponse
from src.agent.permissions import PermissionPolicy
from src.agent.prompts import CLI_ONLY_TEXT
from src.agent.session import SessionStore
from src.agent.skills import SkillRegistry
from src.agent.slash_router import SlashRouter
from src.agent.tools import ToolRuntime

ANALYSIS_TYPES = [
    "descriptive",
    "distributions",
    "variance",
    "correlation",
    "boruta",
    "coefficients",
    "impurity",
    "rules",
]
MODEL_TYPES = [
    "logistic",
    "discriminant",
    "decision-tree",
    "random-forest",
    "xgboost",
    "knn",
    "naive-bayes",
    "svm",
    "dnn",
]
EXPLAIN_TYPES = ["boundary", "pdp", "waterfall", "shap", "extra"]


class AgentRuntime:
    """Coordinates slash commands, skills, tools, context and session memory."""

    def __init__(
            self,
            repo_root: Optional[Path] = None,
            session_id: Optional[str] = None,
            auto_confirm: bool = False,
            confirmer=None,
            cli_runner=None,
    ):
        self.repo_root = Path(repo_root or Path.cwd()).resolve()
        self.session = SessionStore(self.repo_root, session_id=session_id)
        self.skills = SkillRegistry(self.repo_root)
        self.context = ProjectContext(self.repo_root)
        self.active_context: List[ContextItem] = []
        self.permissions = PermissionPolicy(auto_confirm=auto_confirm, confirmer=confirmer)
        self.tools = ToolRuntime(self.repo_root, permission_policy=self.permissions, cli_runner=cli_runner)
        self.router = SlashRouter(self)

    @property
    def slash_commands(self) -> List[str]:
        return self.router.slash_commands

    @property
    def skill_names(self) -> List[str]:
        return self.skills.invocation_names()

    def handle_message(self, message: str) -> RuntimeResponse:
        message = message.rstrip()
        if not message:
            return RuntimeResponse("")

        if message.startswith("/"):
            response = self.router.route(message)
            self._record_response(message, response)
            return response

        if message.startswith("!"):
            response = self.run_bash(message[1:].strip())
            self._record_response(message, response)
            return response

        self.session.add_message("user", message)
        self.active_context = self.context.load_references(message)
        matches = self.skills.match(message)
        lines = []
        if self.active_context:
            lines.append("Loaded context:")
            lines.extend(f"- {item.source}: {len(item.content)} chars" for item in self.active_context)

        if matches:
            lines.append("Skills sugeridas:")
            for skill, score in matches:
                lines.append(f"- {skill.display_name}: {skill.description}")
                if skill.examples:
                    lines.append(f"  Ejemplo: {skill.examples[0]}")
            lines.append("Ejecuta una con /skill <name> [args...] o usa el slash command del ejemplo.")
        else:
            lines.append("No identifique una skill especifica para ese texto.")
            lines.append("Usa /skills para ver tareas disponibles y ejemplos.")

        lines.append(CLI_ONLY_TEXT.strip())
        response = RuntimeResponse("\n".join(lines).strip() or CLI_ONLY_TEXT.strip())
        self.session.add_message("assistant", response.message)
        return response

    def run_cli(self, argv: Iterable[str]) -> RuntimeResponse:
        result = self.tools.run_cli(argv)
        self.session.add_command(result.command, result.exit_code, tool=result.tool)
        message = result.render_text()
        return RuntimeResponse(message, tool_result=result)

    def run_bash(self, command: str) -> RuntimeResponse:
        result = self.tools.run_bash(command)
        self.session.add_command(result.command, result.exit_code, tool=result.tool)
        return RuntimeResponse(result.render_text(), tool_result=result)

    def invoke_skill(self, name: str, args: List[str]) -> RuntimeResponse:
        skill = self.skills.get(name)
        if skill is None:
            matches = self.skills.match(name)
            suggestions = [skill.display_name for skill, _ in matches]
            suggestions.extend(get_close_matches(name, self.skill_names, n=5, cutoff=0.55))
            suggestions = list(dict.fromkeys(suggestions))
            suffix = "\nType /skill <name> and use Tab/autocomplete to see available skills."
            if suggestions:
                suffix = "\nDid you mean: " + ", ".join(suggestions)
            return RuntimeResponse(f'Unknown skill "{name}". Use /skills.{suffix}')

        self.session.add_active_skill(skill.display_name)
        self.active_context = [ContextItem(source=f"skill:{skill.display_name}", content=skill.body)]
        lines = [
            f"Skill: {skill.display_name}",
            skill.description,
        ]
        if args:
            lines.append(f"Arguments: {' '.join(shlex.quote(arg) for arg in args)}")
        if skill.arguments:
            lines.append("Expected arguments: " + ", ".join(skill.arguments))
        if skill.allowed_tools:
            lines.append("Allowed tools: " + ", ".join(skill.allowed_tools))
        if skill.examples:
            lines.append("")
            lines.append("Examples:")
            lines.extend(f"- {example}" for example in skill.examples)
        preview = self._skill_preview(skill.display_name, args)
        if preview:
            lines.append("")
            lines.append(preview)
        lines.append("")
        lines.append(skill.body)
        response = RuntimeResponse("\n".join(lines).strip())
        self.session.add_message("assistant", response.message)
        return response

    def _skill_preview(self, skill_name: str, args: List[str]) -> str:
        if skill_name in {"downloadleague", "league"}:
            return "\n\n".join([
                self._cli_preview("Catalogo de ligas descargables", ["league", "list", "--catalog"]),
                self._cli_preview("Ligas guardadas", ["league", "list"]),
            ])

        if skill_name == "loadleague":
            if args:
                return self._cli_preview(f'Liga "{args[0]}"', ["league", "show", args[0], "--rows", "20"])
            return self._cli_preview("Ligas guardadas", ["league", "list"])

        if skill_name == "train":
            sections = [self._static_preview("Tipos de modelo disponibles", MODEL_TYPES)]
            if args:
                sections.append(self._cli_preview(f'Modelos guardados para "{args[0]}"', ["model", "list", args[0]]))
            else:
                sections.append(self._cli_preview("Ligas guardadas", ["league", "list"]))
            return "\n\n".join(sections)

        if skill_name in {"evaluate", "explain", "predict", "fixtures"}:
            sections = []
            if skill_name == "explain":
                sections.append(self._static_preview("Graficas disponibles", EXPLAIN_TYPES))
            if skill_name == "fixtures":
                sections.append("Opciones de fixtures:\n- Desde archivo: /predict fixtures <league_id> --model <model_id> --input fixtures.csv\n- Desde FootyStats: /predict fixtures <league_id> --model <model_id> --date YYYY-MM-DD --headless")
            if args:
                sections.append(self._cli_preview(f'Modelos guardados para "{args[0]}"', ["model", "list", args[0]]))
            else:
                sections.append(self._cli_preview("Ligas guardadas", ["league", "list"]))
            return "\n\n".join(sections)

        if skill_name == "analysis":
            sections = [self._static_preview("Analisis disponibles", ANALYSIS_TYPES)]
            if args:
                sections.append(self._cli_preview(f'Liga "{args[0]}"', ["league", "show", args[0], "--rows", "5"]))
            else:
                sections.append(self._cli_preview("Ligas guardadas", ["league", "list"]))
            return "\n\n".join(sections)

        if skill_name == "install":
            return "Comprobaciones utiles:\n- python --version\n- python -c \"import prompt_toolkit; print(prompt_toolkit.__version__)\"\n- pip install -r requirements.txt"

        if skill_name == "troubleshoot":
            return self._cli_preview("Configuracion del navegador", ["config", "browser", "show"])

        return ""

    def _cli_preview(self, title: str, argv: List[str]) -> str:
        result = self.tools.run_cli(argv)
        self.session.add_command(result.command, result.exit_code, tool=result.tool)
        return f"{title}:\n{result.render_text()}"

    def _static_preview(self, title: str, values: List[str]) -> str:
        return f"{title}:\n" + "\n".join(f"- {value}" for value in values)

    def skills_text(self) -> str:
        skills = self.skills.list_user_invocable()
        if not skills:
            return "No skills found. Expected skills under skills/*/SKILL.md."
        lines = ["Discovered skills:", ""]
        for index, skill in enumerate(skills):
            args = f" ({', '.join(skill.arguments)})" if skill.arguments else ""
            lines.append(f"/skill {skill.display_name}{args}")
            shortcuts = [f"/skill {skill.display_name}"]
            direct_names = [skill.display_name, *skill.aliases]
            shortcuts.extend(f"/{name}" for name in direct_names if f"/{name}" not in self.slash_commands)
            if shortcuts:
                lines.append("  Shortcuts: " + ", ".join(shortcuts))
            if skill.description:
                lines.append(f"  {skill.description}")
            if skill.examples:
                lines.append(f"  Example: {skill.examples[0]}")
            if index < len(skills) - 1:
                lines.append("")
        return "\n".join(lines)

    def status_text(self) -> str:
        status = self.session.status()
        lines = [
            "Agent status:",
            f"- repo: {self.repo_root}",
            f"- session: {status['session_id']}",
            f"- messages: {status['messages']}",
            f"- commands: {status['commands']}",
            f"- skills: {len(self.skills.list_user_invocable())}",
            f"- active skills: {', '.join(status['active_skills']) or 'none'}",
            f"- current league: {status['current_league'] or 'unset'}",
            f"- current model: {status['current_model'] or 'unset'}",
            "- mode: CLI-only skills runtime",
        ]
        return "\n".join(lines)

    def context_text(self) -> str:
        lines = [self.context.describe(self.active_context)]
        if self.session.state.compact_summary:
            lines.extend(["", "Compact summary:", self.session.state.compact_summary])
        return "\n".join(lines)

    def _record_response(self, user_message: str, response: RuntimeResponse) -> None:
        self.session.add_message("user", user_message)
        if response.message:
            self.session.add_message("assistant", response.message)
