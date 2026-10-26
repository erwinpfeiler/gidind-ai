"""Immutable Hugging Face inputs for the reviewer-facing Figure 3 benchmark.

No local experiment helpers, working-directory datasets, or sampler pickle are
required. Only the released seven test-cell tables and checkpoint are fetched.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SPLIT_REPOSITORY = "epfeiler/gidind-ai-paper-assets"
SPLIT_REVISION = "8a670198023ae5872a8379bdc448591112c26ba6"
MANIFEST_SHA256 = "79c14554ab2d24745453d55d635bb3517e86e481f4fc71b5fd2cb52570515dd5"
SYSTEMS = (
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "rhombohedral",
    "hexagonal",
    "cubic",
)
MODEL = {
    "id": "high_missingness",
    "repository": "epfeiler/gidind-ai",
    "revision": "f1cfccf0db3781747dfa7b184a448ac68e47adff",
    "filename": "real_cases_model.ckpt",
    "sha256": "be7f62e28de4ffe3ee6bfffa9569b238e445986f7de577f3ddd6cd2ce24398ce",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def assets(allow_download: bool = False) -> dict:
    """Download once online, or verify the shared cache without network access."""
    import os
    local = os.environ.get('GIDIND_AI_ASSETS')
    if local:
        root = Path(local)
        manifest_path = root/'paper/manifest.json'
        if sha256(manifest_path) != MANIFEST_SHA256:
            raise ValueError('Local paper manifest checksum mismatch')
        manifest = json.loads(manifest_path.read_text())
        paths, hashes = {}, {}
        for system in SYSTEMS:
            name=f'splits/test_unit_cells/{system}.npy'
            path=root/'paper'/name
            expected=manifest['files'][name]['sha256']
            if sha256(path)!=expected:raise ValueError(f'Checksum mismatch: {name}')
            paths[name], hashes[name]=str(path),expected
        checkpoint=root/'models'/MODEL['filename']
        if sha256(checkpoint)!=MODEL['sha256']:raise ValueError('Checkpoint checksum mismatch')
        return dict(paths=paths,hashes=hashes,model=dict(MODEL),checkpoint=str(checkpoint),
                    source_revision=SPLIT_REVISION,source_repository=SPLIT_REPOSITORY,
                    manifest_sha256=MANIFEST_SHA256)
    from huggingface_hub import hf_hub_download

    def fetch(repo, revision, filename, repo_type="model"):
        try:
            return Path(
                hf_hub_download(
                    repo_id=repo,
                    revision=revision,
                    filename=filename,
                    repo_type=repo_type,
                    local_files_only=not allow_download,
                )
            )
        except OSError as exc:
            raise RuntimeError(
                f"Missing/unavailable pinned asset {repo}/{filename}. Run "
                "`python scripts/benchmark_figure3.py cache` on an internet-connected "
                "login node, using the same HF_HOME as the compute job."
            ) from exc

    manifest_path = fetch(SPLIT_REPOSITORY, SPLIT_REVISION, "manifest.json", "dataset")
    if sha256(manifest_path) != MANIFEST_SHA256:
        raise ValueError("Figure 3 asset manifest checksum mismatch")
    manifest = json.loads(manifest_path.read_text())
    paths, hashes = {}, {}
    for system in SYSTEMS:
        name = f"splits/test_unit_cells/{system}.npy"
        path = fetch(SPLIT_REPOSITORY, SPLIT_REVISION, name, "dataset")
        expected = manifest["files"][name]["sha256"]
        if sha256(path) != expected:
            raise ValueError(f"Checksum mismatch: {name}")
        paths[name], hashes[name] = str(path), expected
    checkpoint = fetch(MODEL["repository"], MODEL["revision"], MODEL["filename"])
    if sha256(checkpoint) != MODEL["sha256"]:
        raise ValueError("Figure 3 checkpoint checksum mismatch")

    # The existing NN constructor initializes torchvision before loading the
    # complete project state dict. Cache its bootstrap weights on the login node.
    import torch
    from torchvision.models import EfficientNet_V2_S_Weights

    weights = EfficientNet_V2_S_Weights.DEFAULT
    backbone = Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[-1]
    if not backbone.is_file():
        if not allow_download:
            raise RuntimeError(
                "Torchvision bootstrap weights are not cached. Run the cache command "
                "online, using the same TORCH_HOME as the compute job."
            )
        weights.get_state_dict(progress=False, check_hash=True)
    backbone_hash = sha256(backbone)
    if not backbone_hash.startswith(backbone.stem.rsplit("-", 1)[-1]):
        raise ValueError("Torchvision bootstrap checksum mismatch")
    return {
        "paths": paths,
        "hashes": hashes,
        "model": dict(MODEL),
        "checkpoint": str(checkpoint),
        "source_revision": SPLIT_REVISION,
        "source_repository": SPLIT_REPOSITORY,
        "manifest_sha256": MANIFEST_SHA256,
        "torchvision_backbone": str(backbone),
        "torchvision_backbone_sha256": backbone_hash,
    }


def sampler(bundle, seed):
    """Match the original sampling algorithm, using only pinned HF test tables."""
    import numpy as np

    from unit_cell_generator import UnitCellGenerator

    obj = UnitCellGenerator(
        gmm_for_sampling=False,
        calc_intensities=False,
        seed=seed,
        optimize_gmm=False,
    )
    obj.test_crystals = {
        s: np.loadtxt(bundle["paths"][f"splits/test_unit_cells/{s}.npy"], dtype=np.float32)
        for s in SYSTEMS
    }
    return obj
