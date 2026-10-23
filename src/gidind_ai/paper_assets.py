"""Resolution and validation of immutable synthetic-benchmark assets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from .workflows import WorkflowError


@dataclass(frozen=True)
class PaperAssetsInfo:
    """Validated source for one fixed paper benchmark split and sampler."""

    manifest_path: Path
    generator_path: Path
    source: str
    repository: str | None = None
    revision: str | None = None
    release_id: str | None = None


def paper_asset_release(asset_id: str) -> dict[str, Any]:
    """Resolve a named, immutable paper-asset release packaged with gidind-ai."""
    manifest = json.loads(files("gidind_ai").joinpath("paper_assets.json").read_text())
    for asset in manifest["assets"]:
        if asset["id"] == asset_id:
            return asset
    choices = ", ".join(asset["id"] for asset in manifest["assets"])
    raise WorkflowError(f"Unknown paper asset {asset_id!r}; choose one of: {choices}.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"Could not read paper-asset manifest {path}: {exc}") from exc
    if manifest.get("schema_version") != 1:
        raise WorkflowError("Unsupported paper-asset manifest schema.")
    return manifest


def _validate_assets(manifest: dict[str, Any], generator_path: Path) -> None:
    record = manifest.get("files", {}).get("gmm/unit_cell_generators.pkl")
    if not isinstance(record, dict) or not isinstance(record.get("sha256"), str):
        raise WorkflowError("Paper-asset manifest lacks the sampler SHA-256 checksum.")
    if not generator_path.is_file():
        raise WorkflowError(f"Paper sampler asset does not exist: {generator_path}")
    if _sha256(generator_path) != record["sha256"]:
        raise WorkflowError("Paper sampler checksum does not match its manifest.")
    expected_sklearn = manifest.get("sampler", {}).get("scikit_learn_version")
    if expected_sklearn:
        try:
            import sklearn
        except ImportError as exc:  # pragma: no cover - dependency is declared by the package
            raise WorkflowError("The paper sampler requires scikit-learn.") from exc
        if sklearn.__version__ != expected_sklearn:
            raise WorkflowError(
                "Paper sampler requires scikit-learn "
                f"{expected_sklearn}, but {sklearn.__version__} is installed."
            )


def resolve_paper_assets(
    *,
    assets_dir: str | Path | None,
    assets_repo: str | None,
    revision: str | None = None,
) -> PaperAssetsInfo:
    """Resolve a local asset bundle or a pinned Hugging Face Dataset revision."""
    if bool(assets_dir) == bool(assets_repo):
        raise WorkflowError("Provide exactly one of --paper-assets or --paper-assets-repo.")
    if assets_dir:
        root = Path(assets_dir).expanduser().resolve()
        manifest_path = root / "manifest.json"
        generator_path = root / "gmm" / "unit_cell_generators.pkl"
        manifest = _load_manifest(manifest_path)
        _validate_assets(manifest, generator_path)
        return PaperAssetsInfo(
            manifest_path=manifest_path,
            generator_path=generator_path,
            source="local",
            release_id=manifest.get("release_id"),
        )

    if not revision:
        raise WorkflowError("--paper-assets-revision is required with --paper-assets-repo.")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise WorkflowError(
            "Paper-asset download requires `pip install 'gidind-ai[paper]'`."
        ) from exc
    try:
        manifest_path = Path(
            hf_hub_download(
                repo_id=assets_repo,
                repo_type="dataset",
                filename="manifest.json",
                revision=revision,
            )
        )
        generator_path = Path(
            hf_hub_download(
                repo_id=assets_repo,
                repo_type="dataset",
                filename="gmm/unit_cell_generators.pkl",
                revision=revision,
            )
        )
    except Exception as exc:  # pragma: no cover - exercised with a mocked client
        raise WorkflowError(
            f"Could not download paper assets from {assets_repo!r} at revision {revision!r}: {exc}"
        ) from exc
    manifest = _load_manifest(manifest_path)
    _validate_assets(manifest, generator_path)
    return PaperAssetsInfo(
        manifest_path=manifest_path,
        generator_path=generator_path,
        source="huggingface",
        repository=assets_repo,
        revision=revision,
        release_id=manifest.get("release_id"),
    )
