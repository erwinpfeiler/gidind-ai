"""Reviewer assets must be immutable and work without unpublished helpers."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from gidind_ai import figure3_assets as module


def digest(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def asset_files(tmp_path, monkeypatch):
    import huggingface_hub
    import torch
    import torchvision.models

    paths = {}
    records = {}
    for system in module.SYSTEMS:
        name = f"splits/test_unit_cells/{system}.npy"
        data = b"10 11 12 90 90 90\n"
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        paths[name] = path
        records[name] = {"sha256": digest(data)}
    manifest = json.dumps({"schema_version": 1, "files": records}).encode()
    paths["manifest.json"] = tmp_path / "manifest.json"
    paths["manifest.json"].write_bytes(manifest)
    checkpoint = b"fixture checkpoint bytes; never loaded as a model"
    paths[module.MODEL["filename"]] = tmp_path / module.MODEL["filename"]
    paths[module.MODEL["filename"]].write_bytes(checkpoint)
    monkeypatch.setattr(module, "MANIFEST_SHA256", digest(manifest))
    monkeypatch.setattr(module, "MODEL", {**module.MODEL, "sha256": digest(checkpoint)})
    bootstrap = b"fixture bootstrap; never loaded as a model"
    filename = f"bootstrap-{digest(bootstrap)[:8]}.pth"
    hub = tmp_path / "torch"
    (hub / "checkpoints").mkdir(parents=True)
    (hub / "checkpoints" / filename).write_bytes(bootstrap)
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(hub))
    monkeypatch.setattr(
        torchvision.models,
        "EfficientNet_V2_S_Weights",
        SimpleNamespace(DEFAULT=SimpleNamespace(url=f"https://example.invalid/{filename}")),
    )
    calls = []

    def fetch(**kwargs):
        calls.append(kwargs)
        return str(paths[kwargs["filename"]])

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fetch)
    return paths, calls


def test_only_pinned_test_assets_are_requested_and_offline_is_default(asset_files):
    _, calls = asset_files
    bundle = module.assets()
    assert len(bundle["paths"]) == 7
    assert len(calls) == 9  # manifest + seven tables + checkpoint
    for call in calls:
        assert call["local_files_only"] is True
        assert call["revision"] in {module.SPLIT_REVISION, module.MODEL["revision"]}
        assert "training" not in call["filename"]
    assert bundle["manifest_sha256"] == module.MANIFEST_SHA256


@pytest.mark.parametrize(
    "filename,match",
    [
        ("manifest.json", "manifest checksum"),
        ("splits/test_unit_cells/triclinic.npy", "Checksum mismatch"),
        ("real_cases_model.ckpt", "checkpoint checksum"),
    ],
)
def test_tampered_assets_fail_before_model_loading(asset_files, filename, match):
    paths, _ = asset_files
    paths[filename].write_bytes(paths[filename].read_bytes() + b"tampered")
    with pytest.raises(ValueError, match=match):
        module.assets()
