"""Fetch pinned public reviewer assets and verify every payload before reuse."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from .figure3_assets import sha256
from .release_models import release_model

ASSETS_REPOSITORY = "epfeiler/gidind-ai-paper-assets"
ASSETS_REVISION = "8a670198023ae5872a8379bdc448591112c26ba6"
PEAK_REVISION = "9c2aef88ad916c6d353e2664c4185fdebe159216"
MANIFEST_SHA256 = "79c14554ab2d24745453d55d635bb3517e86e481f4fc71b5fd2cb52570515dd5"
PEAK_MANIFEST_SHA256 = "a2a1958b5310522c5279244d94e42857b699a9cf872c2769c28d392da70297fc"


def prepare_assets(root: str | Path = ".assets", *, download=True):
    from huggingface_hub import hf_hub_download
    import shutil

    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    def get(repo, revision, name, expected=None, kind="dataset", target=None):
        if target is not None and target.is_file():
            if expected and sha256(target) != expected:
                raise ValueError(f"Local asset checksum mismatch: {target}")
            if expected:
                return target
        path = Path(
            hf_hub_download(
                repo_id=repo,
                repo_type=kind,
                revision=revision,
                filename=name,
                local_files_only=not download,
                token=False,
            )
        )
        if expected and sha256(path) != expected:
            raise ValueError(f"Checksum mismatch: {name}")
        if target is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            return target
        return path

    manifest_path = get(
        ASSETS_REPOSITORY,
        ASSETS_REVISION,
        "manifest.json",
        MANIFEST_SHA256,
        target=root / "paper/manifest.json",
    )
    manifest = json.loads(manifest_path.read_text())
    for name, record in manifest["files"].items():
        # Source COD pickle shards and old audit logs are not needed to run/train.
        if name.startswith("splits/") or name == "gmm/unit_cell_generators.pkl":
            get(
                ASSETS_REPOSITORY,
                ASSETS_REVISION,
                name,
                record["sha256"],
                target=root / "paper" / name,
            )
    for model_id in ["standard", "high_missingness"]:
        model = release_model(model_id)
        get(
            model["repository"],
            model["revision"],
            model["filename"],
            model["sha256"],
            kind="model",
            target=root / "models" / model["filename"],
        )
    peak_path = get(
        ASSETS_REPOSITORY,
        PEAK_REVISION,
        "real_cases/manifest.json",
        PEAK_MANIFEST_SHA256,
        target=root / "real_cases/manifest.json",
    )
    for case in json.loads(peak_path.read_text())["cases"]:
        original = case.get("source_sha256") or case["without_specular"]["source_sha256"]
        for name, digest in [
            (case["filename"], original),
            (case["without_specular"]["filename"], case["without_specular"]["sha256"]),
        ]:
            get(
                ASSETS_REPOSITORY,
                PEAK_REVISION,
                "real_cases/" + name,
                digest,
                target=root / "real_cases" / name,
            )
    # Training retains the original ImageNet initialization; inference loads all
    # project weights strictly and never needs this additional bootstrap download.
    from torchvision.models import EfficientNet_V2_S_Weights
    import torch

    weights = EfficientNet_V2_S_Weights.DEFAULT
    backbone = Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[-1]
    if not backbone.is_file():
        if not download:
            raise FileNotFoundError("Training initialization missing: rerun setup while online")
        weights.get_state_dict(progress=True, check_hash=True)
    if not sha256(backbone).startswith(backbone.stem.rsplit("-", 1)[-1]):
        raise ValueError("Training backbone checksum mismatch")
    info = dict(
        model="high_missingness",
        model_revision=release_model("high_missingness")["revision"],
        assets_revision=ASSETS_REVISION,
        peak_revision=PEAK_REVISION,
        root=str(root),
        public_downloads=True,
    )
    (root / "installed.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=".assets")
    p.add_argument("--offline", action="store_true")
    args = p.parse_args()
    print(json.dumps(prepare_assets(args.root, download=not args.offline), indent=2))


if __name__ == "__main__":
    main()
