"""Pinned model and real-case profile metadata for reviewer workflows."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

from .workflows import WorkflowError


def release_manifest() -> dict[str, Any]:
    """Return the versioned manifest packaged with gidind-ai."""
    return json.loads(files("gidind_ai").joinpath("release_models.json").read_text())


def release_model(model_id: str) -> dict[str, Any]:
    """Resolve one named model from the immutable release manifest."""
    for model in release_manifest()["models"]:
        if model["id"] == model_id:
            return model
    choices = ", ".join(model["id"] for model in release_manifest()["models"])
    raise WorkflowError(f"Unknown release model {model_id!r}; choose one of: {choices}.")


def real_case_profile(case_id: str) -> dict[str, Any]:
    """Resolve one approved real-case profile from the release manifest."""
    for case in release_manifest()["real_cases"]:
        if case["id"] == case_id:
            return case
    choices = ", ".join(case["id"] for case in release_manifest()["real_cases"])
    raise WorkflowError(f"Unknown real-case profile {case_id!r}; choose one of: {choices}.")
