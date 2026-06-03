HELP_TEXT = """\
Available slash commands:

Type / to autocomplete commands and direct skill shortcuts.

/help                         Show this help
/exit, /quit                  Leave agent mode
/clear                        Clear the terminal
/status                       Show session and runtime status
/context                      Show loaded project/session context
/skills                       List discovered skills
/skill <name> [args...]       Load and invoke a skill manually
/reload-skills                Reload skills from disk
/history                      Show recent session messages
/compact                      Compact older session messages into a summary
/run <command>                Run a CLI command, or a safe shell command if it is not a CLI root
/league ...                   Delegate to `python cli.py league ...`
/model ...                    Delegate to `python cli.py model ...`
/predict ...                  Delegate to `python cli.py predict ...`
/analysis ...                 Delegate to `python cli.py analysis ...`
/explain ...                  Delegate to `python cli.py explain ...`

Context shortcuts:
@path                         Load a repository file as context for the current turn
!command                      Run one safe shell command from the repository root
"""


WELCOME_TEXT = """\
ML-STATSSOCCER Agent Mode

Use slash commands to orchestrate the existing CLI. Start with /help or /skills.
"""


CLI_ONLY_TEXT = """\
Este agente no usa un LLM. Funciona como CLI interactivo con slash commands, skills, contexto y herramientas seguras.
Usa /skills para ver tareas disponibles o /help para ver todos los comandos.
"""
