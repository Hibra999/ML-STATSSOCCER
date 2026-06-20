# ML-STATSSOCCER Agent

The agent is an interactive terminal layer above the existing CLI. It does not replace normal commands, does not call an LLM, and does not add any GUI dependency.

## Nota Para Agentes En Este VPS

Este VPS tiene recursos limitados. No ejecutes pruebas, entrenamientos, builds pesados ni comandos de verificacion largos en este entorno, aunque existan instrucciones de testing en la documentacion del proyecto. Haz cambios de codigo con revision estatica ligera y deja que las pruebas se ejecuten en una maquina local con mejor hardware.

Para Mundial, el flujo vigente usa solo el dataset internacional `all_matches.csv` desde 2014. El ETL debe dividir por tiempo en 80/10/10: train inicial, validacion intermedia y test final. El entrenamiento de boosting queda en perfil de features `balanced` por defecto: maximo 480 columnas, sin `train.csv/test.csv`, sin familia `kaggle_`, con `history` compacto y ventanas 3/5/10. Si un agente toca esta parte, debe conservar esos defaults salvo que el usuario pida explicitamente modo completo.

Cuando termines cambios en este repositorio, commitea y sube todo a Git:

```bash
git add <archivos modificados>
git commit -m "mensaje claro"
git push origin main
```

## Start

```bash
python cli.py agent
python cli.py chat
```

## Architecture

```text
src/agent/
  shell.py          prompt_toolkit/Rich REPL
  runtime.py        message orchestration
  slash_router.py   slash command dispatch
  skills.py         SKILL.md discovery and parsing
  tools.py          CLI, bash, read, grep, list, write, git tools
  session.py        JSON session memory
  context.py        @file context loading
  prompts.py        built-in prompt/help text
  models.py         shared dataclasses and provider protocol
  permissions.py    risky command detection
```

The CLI delegation path is `AgentRuntime -> ToolRuntime.run_cli() -> src.cli.app.run(argv)`. This keeps the existing command handlers as the source of truth.

## Slash Commands

```text
/help
/exit
/quit
/clear
/status
/context
/skills
/skill <name> [args...]
/reload-skills
/history
/compact
/run <cli command>
/league ...
/model ...
/predict ...
/analysis ...
/explain ...
```

Examples:

```text
/skills
/skill train epl-2018 random-forest
/league list --catalog
/model list epl-2018
/predict fixtures epl-2018 --model rf-result --input fixtures.csv
/status
/exit
```

## Skills

Project skills live in:

```text
skills/<name>/SKILL.md
```

Local user skills can be added without changing the repo:

```text
.mlstatssoccer/skills/<name>/SKILL.md
```

Each skill uses YAML frontmatter:

```markdown
---
name: train
description: Entrena modelos ML para una liga usando el CLI existente.
when_to_use:
  - cuando el usuario quiera entrenar un modelo
arguments:
  - league_id
  - model_type
examples:
  - /skill train epl-2018 random-forest
  - /model train epl-2018 random-forest --id rf-result
allowed_tools:
  - cli
  - read
user_invocable: true
disable_model_invocation: false
---
```

## Context And Session Memory

- `@path` loads a repository file as context for the current turn.
- `!command` executes one safe shell command from the repository root.
- `/history` shows recent messages.
- `/compact` summarizes older session messages.
- `/status` shows session id, command count, active skills and model provider status.

Session files are stored in `.mlstatssoccer/sessions/`.

## Permissions

The agent blocks or asks confirmation for risky operations:

- `rm`, `rmdir`, `dd`, `truncate`
- `git reset`, `git checkout`, `git clean`, `git restore`, `git rebase`
- `league delete`
- `model delete`
- commands containing destructive words such as `delete`, `remove`, `purge`

The MVP does not run compound shell commands with pipes, redirects, command substitution or shell separators.
