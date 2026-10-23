import json
import random
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import json5
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_CONFIGS = sorted((PROJECT_ROOT / "configs" / "training").glob("*.json5"))


def test_training_configurations_report_the_actual_optimizer_update_budget(monkeypatch, capsys):
    from gidind_ai.training import main

    assert len(TRAINING_CONFIGS) == 4
    observed_profiles = set()
    for config_path in TRAINING_CONFIGS:
        monkeypatch.setenv("GIDIND_AI_CONFIG", str(config_path))
        assert main(["--config", str(config_path), "--validate-only"]) == 0
        summary = json.loads(capsys.readouterr().out)
        observed_profiles.add(
            (summary["missing_peak_rate_max"], "with-specular" in summary["run_id"])
        )
        assert summary["planned_optimizer_steps"] == 319_447
        assert summary["seed"] == 42
        assert summary["deterministic_algorithms"] is True

    assert observed_profiles == {(0.15, False), (0.15, True), (0.30, False), (0.30, True)}


def test_training_configs_only_vary_the_declared_experiment_factors():
    configs = [json5.loads(path.read_text()) for path in TRAINING_CONFIGS]
    for config in configs:
        assert config["run"]["immutable"] is True
        assert config["features"]["training"]["missing_peak_rate"] in {0.15, 0.30}
        assert config["ml_hyperparameters"]["num_epochs"] == 817
        assert config["ml_hyperparameters"]["initial_lr"] == pytest.approx(0.001)
        assert config["ml_hyperparameters"]["seed"] == 42
        assert config["ml_hyperparameters"]["deterministic_algorithms"] is True
        assert config["ml_hyperparameters"]["checkpoint_every_n_epochs"] == 1

    signatures = []
    for config in configs:
        signature = deepcopy(config)
        signature.pop("run")
        signature["features"].pop("use_specular_scan")
        signature["features"]["training"].pop("missing_peak_rate")
        signature["paths"].pop("root_path")
        signatures.append(signature)
    assert all(signature == signatures[0] for signature in signatures[1:])


def test_training_dropout_masks_degenerate_reflections_after_rasterisation(monkeypatch):
    from gixd_data_generator import GixdDataGenerator

    generator = object.__new__(GixdDataGenerator)
    generator.data_mode = "training"
    generator.device = torch.device("cpu")
    generator.q_xy_max = 2.0
    generator.q_z_max = 2.0
    generator.image_size = 256
    generator.num_channels = 3
    generator.max_intensity = 1.0
    generator.feature_config = type(
        "Features",
        (),
        {
            "training_extra_peak_rate": 0.0,
            "training_missing_peak_rate": 0.0,
            "max_intensity": 1.0,
        },
    )()
    # Two q-point entries rasterise to the same observable image pixel.
    q_xy = torch.tensor([[0.4, 0.4]], dtype=torch.float64)
    q_z = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    clean_image = generator.calc_images_from_q_points(q_xy, q_z)
    assert torch.count_nonzero(clean_image) == 3

    generator.feature_config.training_missing_peak_rate = 1.0
    draws = iter((1.0, 0.0, 0.0, 1.0))

    def fixed_random(*shape, device=None, **kwargs):
        return torch.full(shape, next(draws), device=device)

    monkeypatch.setattr(torch, "rand", fixed_random)
    dropped_image = generator.calc_images_from_q_points(q_xy, q_z)
    assert torch.count_nonzero(dropped_image) == 0


def test_training_seed_replays_python_numpy_and_pytorch_streams():
    import numpy as np

    from gidind_ai.training_runtime import configure_training_seed

    configure_training_seed(42, deterministic=True)
    first = (random.random(), np.random.random(), torch.rand(1).item())
    configure_training_seed(42, deterministic=True)
    second = (random.random(), np.random.random(), torch.rand(1).item())

    assert second == pytest.approx(first)


def test_resume_checkpoint_must_belong_to_the_existing_experiment(tmp_path):
    from gidind_ai.training import _prepare_output_directory, _resolve_resume_checkpoint

    root = tmp_path / "experiment"
    model_dir = root / "model"
    model_dir.mkdir(parents=True)
    checkpoint = model_dir / "last.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    path_config = SimpleNamespace(
        root_path=str(root),
        model_path=str(model_dir),
        training_history_path=str(root / "history"),
        prediction_sanity_check_path=str(root / "prediction_sanity_check"),
        ann_toxic_cases_path=str(root / "ann_toxic_cases"),
        wrong_classifications_path=str(root / "wrong_classifications"),
        right_classifications_path=str(root / "right_classifications"),
        bad_predictions_path=str(root / "bad_predictions"),
        test_data_path=str(root / "test_unit_cells"),
    )

    assert _resolve_resume_checkpoint("latest", path_config) == checkpoint.resolve()
    assert _prepare_output_directory(path_config, checkpoint.resolve()) == root

    external_checkpoint = tmp_path / "other.ckpt"
    external_checkpoint.write_bytes(b"checkpoint")
    with pytest.raises(ValueError, match="must belong"):
        _resolve_resume_checkpoint(str(external_checkpoint), path_config)


def test_full_checkpoint_roundtrip_restores_all_training_random_streams(tmp_path):
    import numpy as np

    from gixd_regressor import GixdModule

    # Use the production checkpoint hooks without constructing the large model.
    generator = SimpleNamespace(
        rng=np.random.default_rng(123),
        unit_cell_generator=SimpleNamespace(rng=np.random.default_rng(456)),
    )
    module = SimpleNamespace(
        training_data_generator={"uniform": generator},
        validation_data_generator=None,
        test_data_generator=None,
        _uniform_generator=GixdModule._uniform_generator,
    )
    module._capture_generator_rng_states = lambda: GixdModule._capture_generator_rng_states(module)
    module._restore_generator_rng_states = lambda states: GixdModule._restore_generator_rng_states(module, states)

    checkpoint = {}
    GixdModule.on_save_checkpoint(module, checkpoint)
    checkpoint_path = tmp_path / "last.ckpt"
    torch.save(checkpoint, checkpoint_path)

    def draw():
        return (
            random.random(),
            np.random.random(),
            torch.rand(1).item(),
            generator.rng.random(),
            generator.unit_cell_generator.rng.random(),
        )

    expected = draw()
    draw()  # Simulate later work lost at the wall-time cutoff.
    GixdModule.on_load_checkpoint(
        module, torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    )
    assert draw() == expected
