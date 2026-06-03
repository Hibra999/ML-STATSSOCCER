from pathlib import Path

from src.agent.skills import SkillRegistry, load_skill, parse_skill_markdown


def test_parse_skill_markdown_reads_frontmatter():
    raw = """---
name: train
description: Train models.
aliases:
  - trainmodel
when_to_use:
  - when training is requested
arguments:
  - league_id
allowed_tools:
  - cli
examples:
  - /model train epl-2018 random-forest --id rf-result
user_invocable: true
disable_model_invocation: false
---

# Body
"""

    metadata, body = parse_skill_markdown(raw)

    assert metadata["name"] == "train"
    assert metadata["aliases"] == ["trainmodel"]
    assert metadata["arguments"] == ["league_id"]
    assert metadata["examples"] == ["/model train epl-2018 random-forest --id rf-result"]
    assert "# Body" in body


def test_skill_registry_discovers_and_matches_project_skill(tmp_path: Path):
    skill_dir = tmp_path / "skills" / "train"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        """---
name: train
description: Entrena modelos random forest.
aliases:
  - trainmodel
when_to_use:
  - cuando mencione entrenamiento
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
""",
        encoding="utf-8",
    )

    skill = load_skill(skill_file)
    registry = SkillRegistry(tmp_path)

    assert skill.name == "train"
    assert skill.aliases == ["trainmodel"]
    assert skill.examples == ["/model train epl-2018 random-forest --id rf-result"]
    assert registry.get("train").description.startswith("Entrena")
    assert registry.get("trainmodel").description.startswith("Entrena")
    assert "trainmodel" in registry.invocation_names()
    assert registry.match("quiero entrenar random forest")[0][0].name == "train"
