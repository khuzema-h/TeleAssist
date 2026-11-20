"""Object catalog for the TeleAssist table scene (mug only for now)."""

from __future__ import annotations

from typing import Any

# Plate temporarily removed; mug matches pseudo_vla/kitchen.py.
TABLE_OBJECT_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "mug",
        "name": "mug",
        "phrase": "mug",
        "kind": "asset",
        "affordance": "hold",
        "grasp_style": "side",
    },
)

# Backward-compatible alias used by older scan imports.
STACK_CUBE_SPECS = TABLE_OBJECT_SPECS


def object_id(spec: dict[str, Any]) -> str:
    return str(spec["id"])


def object_phrase(spec: dict[str, Any]) -> str:
    phrase = spec.get("phrase")
    if isinstance(phrase, str) and phrase.strip():
        return phrase.strip()
    return str(object_id(spec))
