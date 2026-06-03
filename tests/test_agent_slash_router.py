from pathlib import Path

from src.agent.runtime import AgentRuntime


def test_slash_router_delegates_cli_commands(tmp_path: Path):
    calls = []

    def fake_cli(argv):
        calls.append(argv)
        print("delegated")
        return 0

    runtime = AgentRuntime(repo_root=tmp_path, cli_runner=fake_cli)
    response = runtime.handle_message("/league list --catalog")

    assert calls == [["league", "list", "--catalog"]]
    assert "delegated" in response.message


def test_slash_router_runs_skill(tmp_path: Path):
    skill_dir = tmp_path / "skills" / "train"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: train
description: Entrena modelos.
when_to_use:
  - entrenamiento
arguments:
  - league_id
allowed_tools:
  - cli
examples:
  - /model train epl-2018 random-forest --id rf-result
user_invocable: true
disable_model_invocation: false
---

# Train

Use `model train`.
""",
        encoding="utf-8",
    )
    calls = []

    def fake_cli(argv):
        calls.append(argv)
        print("models")
        return 0

    runtime = AgentRuntime(repo_root=tmp_path, cli_runner=fake_cli)

    response = runtime.handle_message("/skill train epl-2018 random-forest")

    assert "Skill: train" in response.message
    assert "Examples:" in response.message
    assert "epl-2018 random-forest" in response.message
    assert "Tipos de modelo disponibles" in response.message
    assert calls == [["model", "list", "epl-2018"]]


def test_slash_router_runs_direct_skill_shortcut(tmp_path: Path):
    skill_dir = tmp_path / "skills" / "downloadleague"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: downloadleague
description: Descarga ligas.
aliases:
  - createleague
examples:
  - /league list --catalog
user_invocable: true
---

# Download
""",
        encoding="utf-8",
    )
    calls = []

    def fake_cli(argv):
        calls.append(argv)
        print("catalog")
        return 0

    runtime = AgentRuntime(repo_root=tmp_path, cli_runner=fake_cli)

    response = runtime.handle_message("/downloadleague epl-2018")

    assert "Skill: downloadleague" in response.message
    assert "Arguments: epl-2018" in response.message
    assert "Catalogo de ligas descargables" in response.message
    assert "Ligas guardadas" in response.message
    assert calls == [["league", "list", "--catalog"], ["league", "list"]]


def test_slash_router_suggests_unknown_skill_like_command(tmp_path: Path):
    skill_dir = tmp_path / "skills" / "downloadleague"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: downloadleague
description: Descarga ligas.
user_invocable: true
---

# Download
""",
        encoding="utf-8",
    )
    runtime = AgentRuntime(repo_root=tmp_path)

    response = runtime.handle_message("/downlodleague")

    assert "Unknown slash command" in response.message
    assert "/downloadleague" in response.message


def test_loadleague_lists_or_shows_available_leagues(tmp_path: Path):
    skill_dir = tmp_path / "skills" / "loadleague"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: loadleague
description: Carga ligas.
user_invocable: true
---

# Load
""",
        encoding="utf-8",
    )
    calls = []

    def fake_cli(argv):
        calls.append(argv)
        print("league output")
        return 0

    runtime = AgentRuntime(repo_root=tmp_path, cli_runner=fake_cli)

    list_response = runtime.handle_message("/loadleague")
    show_response = runtime.handle_message("/loadleague epl-2018")

    assert "Ligas guardadas" in list_response.message
    assert 'Liga "epl-2018"' in show_response.message
    assert calls == [["league", "list"], ["league", "show", "epl-2018", "--rows", "20"]]


def test_trainmodel_alias_shows_model_options(tmp_path: Path):
    skill_dir = tmp_path / "skills" / "train"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: train
description: Entrena modelos.
aliases:
  - trainmodel
user_invocable: true
---

# Train
""",
        encoding="utf-8",
    )
    calls = []

    def fake_cli(argv):
        calls.append(argv)
        print("saved models")
        return 0

    runtime = AgentRuntime(repo_root=tmp_path, cli_runner=fake_cli)

    response = runtime.handle_message("/trainmodel epl-2018")

    assert "Skill: train" in response.message
    assert "Tipos de modelo disponibles" in response.message
    assert "- random-forest" in response.message
    assert 'Modelos guardados para "epl-2018"' in response.message
    assert calls == [["model", "list", "epl-2018"]]


def test_analysis_skill_shows_analysis_types_and_leagues(tmp_path: Path):
    skill_dir = tmp_path / "skills" / "analysis"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: analysis
description: Analisis.
aliases:
  - stats
user_invocable: true
---

# Analysis
""",
        encoding="utf-8",
    )
    calls = []

    def fake_cli(argv):
        calls.append(argv)
        print("saved leagues")
        return 0

    runtime = AgentRuntime(repo_root=tmp_path, cli_runner=fake_cli)

    response = runtime.handle_message("/stats")

    assert "Analisis disponibles" in response.message
    assert "- variance" in response.message
    assert "Ligas guardadas" in response.message
    assert calls == [["league", "list"]]


def test_model_based_skills_show_models_for_league(tmp_path: Path):
    for name in ["evaluate", "fixtures", "explain"]:
        skill_dir = tmp_path / "skills" / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"""---
name: {name}
description: Skill {name}.
user_invocable: true
---

# {name}
""",
            encoding="utf-8",
        )
    calls = []

    def fake_cli(argv):
        calls.append(argv)
        print("models")
        return 0

    runtime = AgentRuntime(repo_root=tmp_path, cli_runner=fake_cli)

    evaluate_response = runtime.handle_message("/evaluate epl-2018")
    fixtures_response = runtime.handle_message("/fixtures epl-2018")
    explain_response = runtime.handle_message("/skill explain epl-2018")

    assert 'Modelos guardados para "epl-2018"' in evaluate_response.message
    assert "Opciones de fixtures" in fixtures_response.message
    assert "Graficas disponibles" in explain_response.message
    assert calls == [
        ["model", "list", "epl-2018"],
        ["model", "list", "epl-2018"],
        ["model", "list", "epl-2018"],
    ]


def test_skill_invocation_suggests_unknown_name(tmp_path: Path):
    skill_dir = tmp_path / "skills" / "loadleague"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: loadleague
description: Carga ligas.
user_invocable: true
---

# Load
""",
        encoding="utf-8",
    )
    runtime = AgentRuntime(repo_root=tmp_path)

    response = runtime.handle_message("/skill lodleague")

    assert 'Unknown skill "lodleague"' in response.message
    assert "loadleague" in response.message
