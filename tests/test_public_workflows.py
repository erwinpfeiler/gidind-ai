import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from gidind_ai.cli import main
from gidind_ai.workflows import (
    DEFAULT_MODEL_REPOSITORY,
    DEFAULT_MODEL_REVISION,
    WorkflowError,
    _build_regressor,
    _candidate_records,
    resolve_checkpoint,
    select_device,
)


def test_reviewer_inference_sets_a_loaded_model_to_evaluation_mode(monkeypatch, tmp_path):
    class FakeGenerator:
        def __init__(self, *args, **kwargs):
            self.device = None
            self.laue_indices_tensor = SimpleNamespace(to=lambda device: ("indices", device))

    class FakeModel:
        def __init__(self):
            self.is_in_evaluation_mode = False

        def eval(self):
            self.is_in_evaluation_mode = True
            return self

    class FakeRegressor:
        def __init__(self, *args):
            self.gixd_model = FakeModel()

        def load_checkpoint(self, checkpoint):
            self.loaded_checkpoint = checkpoint

    generator_module = ModuleType("gixd_data_generator")
    generator_module.GixdDataGenerator = FakeGenerator
    regressor_module = ModuleType("gixd_regressor")
    regressor_module.GixdRegressor = FakeRegressor
    monkeypatch.setitem(sys.modules, "gixd_data_generator", generator_module)
    monkeypatch.setitem(sys.modules, "gixd_regressor", regressor_module)

    regressor, _ = _build_regressor(tmp_path / "model.ckpt", "cpu")

    assert regressor.gixd_model.is_in_evaluation_mode


def test_resolve_checkpoint_requires_exactly_one_model_source(tmp_path):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")

    assert resolve_checkpoint(checkpoint=checkpoint, model_repo=None).path == checkpoint

    with pytest.raises(WorkflowError, match="exactly one"):
        resolve_checkpoint(checkpoint=None, model_repo=None)
    with pytest.raises(WorkflowError, match="exactly one"):
        resolve_checkpoint(checkpoint=checkpoint, model_repo="owner/model")


def test_missing_checkpoint_is_an_actionable_error(tmp_path):
    with pytest.raises(WorkflowError, match="does not exist"):
        resolve_checkpoint(checkpoint=tmp_path / "missing.ckpt", model_repo=None)


def test_local_paper_assets_verify_sampler_checksum(tmp_path):
    from gidind_ai.paper_assets import resolve_paper_assets

    sampler = tmp_path / "gmm" / "unit_cell_generators.pkl"
    sampler.parent.mkdir()
    sampler.write_bytes(b"trusted test sampler")
    import hashlib

    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "release_id": "test-assets",
                "files": {
                    "gmm/unit_cell_generators.pkl": {
                        "sha256": hashlib.sha256(sampler.read_bytes()).hexdigest()
                    }
                },
                "sampler": {},
            }
        )
    )
    info = resolve_paper_assets(assets_dir=tmp_path, assets_repo=None)

    assert info.source == "local"
    assert info.release_id == "test-assets"
    sampler.write_bytes(b"changed")
    with pytest.raises(WorkflowError, match="checksum"):
        resolve_paper_assets(assets_dir=tmp_path, assets_repo=None)


def test_benchmark_defaults_to_the_pinned_paper_asset_release(tmp_path, monkeypatch):
    calls = {}

    monkeypatch.setattr(
        "gidind_ai.cli.resolve_checkpoint",
        lambda **kwargs: SimpleNamespace(path=tmp_path / "model.ckpt"),
    )
    monkeypatch.setattr(
        "gidind_ai.cli.paper_asset_release",
        lambda asset_id: {"repository": "owner/assets", "revision": "a" * 40},
    )
    monkeypatch.setattr(
        "gidind_ai.cli.resolve_paper_assets",
        lambda **kwargs: calls.setdefault("assets", kwargs) or SimpleNamespace(),
    )
    monkeypatch.setattr(
        "gidind_ai.cli.run_benchmark", lambda **kwargs: calls.setdefault("run", kwargs)
    )

    assert main(["benchmark", "--output-dir", str(tmp_path), "--checkpoint", "local.ckpt"]) == 0
    assert calls["assets"] == {
        "assets_dir": None,
        "assets_repo": "owner/assets",
        "revision": "a" * 40,
    }
    assert calls["run"]["paper_assets"] is not None


def test_cli_uses_pinned_publication_model_by_default(tmp_path, monkeypatch):
    captured = {}

    def fake_resolve_checkpoint(**kwargs):
        captured.update(kwargs)
        raise WorkflowError("stop after resolving")

    monkeypatch.setattr("gidind_ai.cli.resolve_checkpoint", fake_resolve_checkpoint)
    status = main(["infer", "--input", "missing.csv", "--output-dir", str(tmp_path)])

    assert status == 2
    assert captured == {
        "checkpoint": None,
        "model_repo": DEFAULT_MODEL_REPOSITORY,
        "revision": DEFAULT_MODEL_REVISION,
        "filename": "default_model.ckpt",
        "model_id": None,
        "sha256": None,
    }


def test_naib_profile_resolves_the_separate_pinned_release_model(tmp_path, monkeypatch):
    captured = {}

    def fake_resolve_checkpoint(**kwargs):
        captured.update(kwargs)
        raise WorkflowError("stop after resolving")

    monkeypatch.setattr("gidind_ai.cli.resolve_checkpoint", fake_resolve_checkpoint)
    status = main(
        [
            "infer",
            "--input",
            "missing.csv",
            "--output-dir",
            str(tmp_path),
            "--case-profile",
            "naib_s11",
        ]
    )

    assert status == 2
    assert captured == {
        "checkpoint": None,
        "model_repo": "epfeiler/gidind-ai",
        "revision": "f1cfccf0db3781747dfa7b184a448ac68e47adff",
        "filename": "real_cases_model.ckpt",
        "model_id": "high_missingness",
        "sha256": "be7f62e28de4ffe3ee6bfffa9569b238e445986f7de577f3ddd6cd2ce24398ce",
    }


def test_release_manifest_and_reviewer_notebooks_pin_each_model():
    from gidind_ai.release_models import release_manifest

    manifest = release_manifest()
    models = {model["id"]: model for model in manifest["models"]}
    assert set(models) == {"standard", "high_missingness"}
    for model in models.values():
        assert len(model["revision"]) == 40
        assert len(model["sha256"]) == 64
        assert model["input_domain_angstrom_inverse"] == {"q_xy": [0.0, 2.0], "q_z": [0.0, 2.0]}
        assert model["refinement"]["q_loss_threshold"] == pytest.approx(1e-3)

    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/04_real_cases.ipynb").read_text())
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    for case_id in ["pq", "cu_bod_mof", "mapbbr3", "naib_s11"]:
        assert f"run_real_case(ROOT, '{case_id}'" in code
    for profile in manifest["real_cases"]:
        assert profile["model_id"] == "high_missingness"
        assert profile["q_scale"] == pytest.approx(0.9)
        assert profile["refinement"]["q_cutoff_angstrom_inverse"] == pytest.approx(0.06)
        assert profile["refinement"]["maxiter"] == 500
        assert profile["refinement"]["matching_method"] == "hybrid"
        assert profile["refinement"]["outer_loops"] == 20


def test_real_case_manifest_retains_the_historical_cu_bod_input():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "examples" / "real_cases" / "manifest.json").read_text())
    cases = {case["id"]: case for case in manifest["cases"]}

    assert cases["cu_bod_mof"]["q_scale"] == 1.0
    assert "Historical manually selected" in cases["cu_bod_mof"]["provenance"]
    assert cases["naib_s11"]["q_scale"] == 0.9
    assert (root / "examples" / "real_cases" / "cu_bod_mof.csv").read_text().splitlines() == [
        "q_xy,q_z",
        "0.578,0.000",
        "0.817,0.000",
        "1.157,0.000",
        "0.581,0.651",
        "0.822,0.650",
        "1.160,0.650",
        "1.290,0.000",
        "1.734,0.000",
        "1.295,0.655",
        "1.735,0.649",
    ]


def test_device_validation_is_actionable():
    with pytest.raises(WorkflowError, match="one of"):
        select_device("metal")


def test_peak_loader_accepts_a_two_column_csv(tmp_path):
    from load_peaks import load_peaks_qxy_qz

    input_path = tmp_path / "peaks.csv"
    input_path.write_text("q_xy,q_z\n0.2,0.3\n0.4,0.5\n")
    features = SimpleNamespace(max_q_xy=2.0, max_q_z=2.0)

    q_xy, q_z, scale = load_peaks_qxy_qz(input_path, features, q_scale=1.0)

    assert scale == 1.0
    assert q_xy.tolist() == pytest.approx([0.2, 0.4])
    assert q_z.tolist() == pytest.approx([0.3, 0.5])


def test_dynamic_peak_scaling_uses_the_shorter_axis_without_clipping():
    from load_peaks import load_peaks_qxy_qz

    features = SimpleNamespace(max_q_xy=2.0, max_q_z=2.0)
    q_xy, q_z, scale = load_peaks_qxy_qz(np.asarray([[1.0, 1.0], [4.0, 2.0]]), features)

    assert scale == pytest.approx(1.0)
    assert q_xy.tolist() == pytest.approx([1.0, 4.0])
    assert q_z.tolist() == pytest.approx([1.0, 2.0])


def test_refinement_peak_count_accepts_real_peak_list_shape():
    """Real inputs have shape (n_peaks), unlike batched synthetic inputs."""
    from gixd_regressor import _observed_peak_count

    assert _observed_peak_count(np.zeros(3)) == 3
    assert _observed_peak_count(np.zeros((1, 3))) == 3


def test_inference_records_restore_physical_length_scale():
    raw = {
        "refined0_a": 8.0,
        "refined0_b": 10.0,
        "refined0_c": 12.0,
        "refined0_alpha": 90.0,
        "refined0_beta": 91.0,
        "refined0_gamma": 92.0,
        "unc0_a": 0.8,
        "unc0_b": 1.0,
        "unc0_c": 1.2,
        "unc0_alpha": 0.1,
        "unc0_beta": 0.2,
        "unc0_gamma": 0.3,
        "is_converged0": True,
        "q_loss0": 0.01,
        "percent_matched0": 0.9,
    }
    record = _candidate_records(raw, [[6.0, 8.0, 10.0, 90, 90, 90]], [(1, 0, 0)], 2.0)[0]

    assert record["initial_cell_deg"][:3] == [3.0, 4.0, 5.0]
    assert record["refined_cell_deg"][:3] == [4.0, 5.0, 6.0]
    assert record["laplace_uncertainty"][:3] == [0.4, 0.5, 0.6]
