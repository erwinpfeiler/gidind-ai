"""Launch one immutable, reproducible synthetic-training experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gidind_ai.training",
        description="Train gidind-ai from one version-controlled experiment configuration.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to one immutable configuration in configs/training/.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the experiment definition without creating outputs or using a GPU.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume from a rolling checkpoint in this experiment's output directory, or use 'latest'.",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


ACCUMULATE_GRAD_BATCHES = 1


def _expected_optimizer_steps(samples_per_epoch: int, batch_size: int) -> int:
    micro_batches = -(-samples_per_epoch // batch_size)
    return -(-micro_batches // ACCUMULATE_GRAD_BATCHES)


def _validate_config(config: Any) -> dict[str, Any]:
    run = config.get_property(["run"])
    if not isinstance(run, dict) or not run.get("id") or not run.get("model_name"):
        raise ValueError("The configuration needs run.id and run.model_name.")
    if run.get("immutable") is not True:
        raise ValueError("The configuration must declare run.immutable: true.")

    maximum_rate = config.get_property(["features", "training", "missing_peak_rate"])
    if not isinstance(maximum_rate, (int, float)) or not 0.0 <= maximum_rate <= 1.0:
        raise ValueError(
            "features.training.missing_peak_rate must be a number from 0 to 1."
        )

    from config_loader import MachineLearningHyperparameter

    hyperparameters = MachineLearningHyperparameter()
    if hyperparameters.num_epochs < 1:
        raise ValueError("ml_hyperparameters.num_epochs must be positive.")
    if hyperparameters.training_batch_size < 1 or hyperparameters.training_epoch_size < 1:
        raise ValueError("Training batch size and samples per epoch must be positive.")

    steps_per_epoch = _expected_optimizer_steps(
        hyperparameters.training_epoch_size, hyperparameters.training_batch_size
    )
    return {
        "run_id": str(run["id"]),
        "model_name": str(run["model_name"]),
        "steps_per_epoch": steps_per_epoch,
        "planned_optimizer_steps": steps_per_epoch * hyperparameters.num_epochs,
        "seed": hyperparameters.training_seed,
        "deterministic_algorithms": hyperparameters.deterministic_algorithms,
        "checkpoint_every_n_epochs": hyperparameters.checkpoint_every_n_epochs,
        "missing_peak_rate_max": float(maximum_rate),
    }


def _resolve_resume_checkpoint(resume: str | None, path_config: Any) -> Path | None:
    if resume is None:
        return None
    path = Path(path_config.model_path) / "last.ckpt" if resume == "latest" else Path(resume)
    path = path.expanduser().resolve()
    root = Path(path_config.root_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    if not path.is_relative_to(root):
        raise ValueError(
            f"Resume checkpoint must belong to this experiment's output directory: {root}"
        )
    return path


def _prepare_output_directory(path_config: Any, resume_checkpoint: Path | None) -> Path:
    root = Path(path_config.root_path)
    if root.exists() and any(root.iterdir()):
        if resume_checkpoint is None:
            raise FileExistsError(
                f"Refusing to reuse non-empty training output directory: {root}. "
                "Use --resume latest to continue this exact experiment."
            )
    elif resume_checkpoint is not None:
        raise FileNotFoundError(
            f"Cannot resume because this experiment output directory is empty: {root}"
        )
    root.mkdir(parents=True, exist_ok=True)
    for directory in (
        path_config.model_path,
        path_config.training_history_path,
        path_config.prediction_sanity_check_path,
        path_config.ann_toxic_cases_path,
        path_config.wrong_classifications_path,
        path_config.right_classifications_path,
        path_config.bad_predictions_path,
        path_config.test_data_path,
    ):
        Path(directory).mkdir(parents=True, exist_ok=True)
    return root


def _write_manifest(
    root: Path,
    config_path: Path,
    experiment: dict[str, Any],
    resume_checkpoint: Path | None,
) -> None:
    copied_config = root / "training_config.json5"
    manifest_path = root / "run_manifest.json"
    config_sha256 = _sha256(config_path)
    if resume_checkpoint is not None:
        if not copied_config.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(
                "Cannot resume without the original training_config.json5 and run_manifest.json."
            )
        if _sha256(copied_config) != config_sha256:
            raise ValueError("Refusing to resume with a configuration that differs from the original run.")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("config_sha256") != config_sha256:
            raise ValueError("The existing run manifest does not match the selected configuration.")
        manifest.setdefault("resume_events", []).append(
            {
                "resumed_at_utc": datetime.now(timezone.utc).isoformat(),
                "checkpoint": str(resume_checkpoint),
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        return

    shutil.copy2(config_path, copied_config)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        "config_file": config_path.name,
        "config_sha256": config_sha256,
        "experiment": experiment,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        _parser().error(f"configuration file does not exist: {config_path}")

    # This must precede all imports that can initialise CUDA.
    os.environ["GIDIND_AI_CONFIG"] = str(config_path)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    from config_loader import Config, PathConfig

    config = Config()
    try:
        experiment = _validate_config(config)
    except (KeyError, TypeError, ValueError) as exc:
        _parser().error(str(exc))

    path_config = PathConfig()
    try:
        resume_checkpoint = _resolve_resume_checkpoint(args.resume, path_config)
    except (FileNotFoundError, ValueError) as exc:
        _parser().error(str(exc))

    if args.validate_only:
        experiment["resume_checkpoint"] = str(resume_checkpoint) if resume_checkpoint else None
        print(json.dumps(experiment, indent=2, sort_keys=True))
        return 0

    from gidind_ai.training_runtime import configure_training_seed
    from gixd_data_generator import create_data_generators
    from gixd_regressor import GixdRegressor

    configure_training_seed(
        experiment["seed"], deterministic=experiment["deterministic_algorithms"]
    )
    try:
        root = _prepare_output_directory(path_config, resume_checkpoint)
        _write_manifest(root, config_path, experiment, resume_checkpoint)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        _parser().error(str(exc))

    training_generators, validation_generators, test_generators = create_data_generators(
        "uniform",
        ["cell", "top_layer"],
        num_phases=1,
        use_gmm=False,
        split_orientations=True,
        use_filtered_data_stream=True,
        calc_intensities=False,
    )
    regressor = GixdRegressor(
        training_data_generators=training_generators,
        validation_data_generators=validation_generators,
        test_data_generators=test_generators,
        num_phases=1,
        use_gmm=False,
    )
    regressor.train_gixd_model(
        model_name=experiment["model_name"],
        resume_checkpoint=str(resume_checkpoint) if resume_checkpoint else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
