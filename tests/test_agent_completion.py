from pathlib import Path

from src.agent.completion import AgentCommandCompleter
from src.agent.runtime import AgentRuntime


def completion_texts(completer: AgentCommandCompleter, line: str):
    return [text for text, _, _ in completer.complete_line(line)]


def test_slash_completion_lists_all_root_options():
    completer = AgentCommandCompleter(["/help", "/skills", "/status"], ["train"])

    texts = completion_texts(completer, "/")

    assert "/help" in texts
    assert "/skills" in texts
    assert "/status" in texts
    assert "/skill train" in texts


def test_slash_completion_filters_as_user_types():
    completer = AgentCommandCompleter(["/help", "/skills", "/status", "/model"], ["train"])

    texts = completion_texts(completer, "/s")

    assert "/skills" in texts
    assert "/status" in texts
    assert "/skill train" in texts
    assert "/help" not in texts
    assert "/model" not in texts


def test_skill_argument_completion_filters_skill_names():
    completer = AgentCommandCompleter(["/skill", "/skills"], ["analysis", "train", "troubleshoot"])

    texts = completion_texts(completer, "/skill tr")

    assert texts == ["train", "troubleshoot"]


def test_completion_ignores_non_slash_text_and_command_arguments():
    completer = AgentCommandCompleter(["/help", "/model"], ["train"])

    assert completion_texts(completer, "hola") == []
    assert completion_texts(completer, "/model list") == []


def test_skills_text_spaces_entries(tmp_path: Path):
    for name in ["train", "predict"]:
        skill_dir = tmp_path / "skills" / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"""---
name: {name}
description: Skill {name}.
examples:
  - /{name} example
user_invocable: true
---

# {name}
""",
            encoding="utf-8",
        )

    runtime = AgentRuntime(repo_root=tmp_path)
    text = runtime.skills_text()

    assert "/skill predict" in text
    assert "/skill train" in text
    assert "\n\n/skill train" in text
