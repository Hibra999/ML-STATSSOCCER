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
    runtime = AgentRuntime(repo_root=tmp_path)

    response = runtime.handle_message("/skill train epl-2018 random-forest")

    assert "Skill: train" in response.message
    assert "Examples:" in response.message
    assert "epl-2018 random-forest" in response.message
