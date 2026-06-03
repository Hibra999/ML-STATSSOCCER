from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from src.agent.models import Skill

try:
    import yaml
except ImportError:  # pragma: no cover - project requirements include PyYAML.
    yaml = None


FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.DOTALL)


class SkillRegistry:
    """Discovers and loads agent skills from project and local skill folders."""

    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root).resolve()
        self.skill_roots = [
            self.repo_root / "skills",
            self.repo_root / ".mlstatssoccer" / "skills",
        ]
        self.skills: Dict[str, Skill] = {}
        self.reload()

    def reload(self) -> None:
        loaded: Dict[str, Skill] = {}
        for root in self.skill_roots:
            if not root.exists():
                continue
            for path in sorted(root.glob("*/SKILL.md")):
                skill = load_skill(path)
                keys = {skill.key, normalize_skill_key(skill.name)}
                keys.update(normalize_skill_key(alias) for alias in skill.aliases)
                for key in keys:
                    if key:
                        loaded[key] = skill
        self.skills = loaded

    def list_user_invocable(self) -> List[Skill]:
        unique = {skill.path: skill for skill in self.skills.values()}
        return sorted(unique.values(), key=lambda skill: skill.display_name)

    def invocation_names(self) -> List[str]:
        names = set()
        for skill in self.list_user_invocable():
            names.add(skill.display_name)
            names.update(skill.aliases)
        return sorted((name for name in names if name), key=str.lower)

    def get(self, name: str) -> Optional[Skill]:
        return self.skills.get(normalize_skill_key(name))

    def match(self, text: str, limit: int = 3) -> List[Tuple[Skill, int]]:
        query_terms = tokenize(text)
        if not query_terms:
            return []

        scored: Dict[Path, Tuple[Skill, int]] = {}
        for skill in self.skills.values():
            text_terms = set(tokenize(skill.searchable_text))
            score = sum(3 if term in tokenize(skill.name) else 1 for term in query_terms if term in text_terms)
            if score <= 0:
                continue
            existing = scored.get(skill.path)
            if existing is None or score > existing[1]:
                scored[skill.path] = (skill, score)
        return sorted(scored.values(), key=lambda item: item[1], reverse=True)[:limit]


def load_skill(path: Path) -> Skill:
    raw = path.read_text(encoding="utf-8")
    metadata, body = parse_skill_markdown(raw)
    key = normalize_skill_key(str(metadata.get("name") or path.parent.name))
    return Skill(
        key=key,
        name=str(metadata.get("name") or path.parent.name),
        description=str(metadata.get("description") or ""),
        aliases=as_list(metadata.get("aliases")),
        when_to_use=as_list(metadata.get("when_to_use")),
        arguments=as_list(metadata.get("arguments")),
        examples=as_list(metadata.get("examples")),
        allowed_tools=as_list(metadata.get("allowed_tools")),
        user_invocable=as_bool(metadata.get("user_invocable"), default=True),
        disable_model_invocation=as_bool(metadata.get("disable_model_invocation"), default=False),
        path=path,
        body=body.strip(),
        metadata=metadata,
    )


def parse_skill_markdown(raw: str) -> Tuple[Dict[str, object], str]:
    match = FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw
    metadata_text, body = match.groups()
    if yaml is not None:
        metadata = yaml.safe_load(metadata_text) or {}
    else:
        metadata = parse_simple_yaml(metadata_text)
    if not isinstance(metadata, dict):
        metadata = {}
    return metadata, body


def parse_simple_yaml(raw: str) -> Dict[str, object]:
    """Tiny fallback parser for the skill frontmatter shape used in this repo."""

    result: Dict[str, object] = {}
    current_key = None
    for line in raw.splitlines():
        if not line.strip():
            continue
        if line.startswith("  - ") and current_key:
            result.setdefault(current_key, [])
            value = line[4:].strip()
            if isinstance(result[current_key], list):
                result[current_key].append(value)
            continue
        if ":" in line:
            key, value = line.split(":", maxsplit=1)
            current_key = key.strip()
            value = value.strip()
            if value == "":
                result[current_key] = []
            elif value.lower() in {"true", "false"}:
                result[current_key] = value.lower() == "true"
            else:
                result[current_key] = value.strip('"')
    return result


def normalize_skill_key(value: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", value.strip().lower().replace("_", "-")).strip("-")


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9][a-z0-9-]*", text.lower())


def as_list(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    return [str(value)]


def as_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
