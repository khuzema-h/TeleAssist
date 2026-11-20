"""Object catalog for the Lab Franka cube-stack scene."""

from __future__ import annotations

from typing import Any

# Lab ObjectTableSceneCfg: cube_1=blue, cube_2=red, cube_3=green.
STACK_CUBE_SPECS: tuple[dict[str, Any], ...] = (
    {"id": "blue", "name": "cube_1", "phrase": "blue block", "affordance": "pick_up", "grasp_style": "auto"},
    {"id": "red", "name": "cube_2", "phrase": "red block", "affordance": "pick_up", "grasp_style": "auto"},
    {"id": "green", "name": "cube_3", "phrase": "green block", "affordance": "pick_up", "grasp_style": "auto"},
)


def object_id(spec: dict[str, Any]) -> str:
    return str(spec["id"])


def object_phrase(spec: dict[str, Any]) -> str:
    phrase = spec.get("phrase")
    if isinstance(phrase, str) and phrase.strip():
        return phrase.strip()
    return f"{object_id(spec)} block"
