"""Portable reviewer workflows built on gidind-ai's PyTorch core."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .mcmc import MCMCSettings


DEFAULT_MODEL_REPOSITORY = "epfeiler/gidind-ai"
"""Publication model repository used when a source is not specified."""

DEFAULT_MODEL_REVISION = "374aed64cf921eb47e3e03452c4337e0e636ff93"
"""Immutable Hugging Face commit containing the publication checkpoint."""


class WorkflowError(RuntimeError):
    """An actionable error in a public gidind-ai workflow."""


@dataclass(frozen=True)
class RefinementSettings:
    """Backward-compatible explicit controls for reviewer notebooks and API users."""

    q_cutoff: float = 0.06
    maxiter: int = 50
    outer_loops: int = 10
    matching_method: str = "simulated_fractional"
    q_loss_threshold: float = 5e-5
    percent_matched_threshold: float = 0.90
    optimizer_method: str = "cg"
    loss_type: str = "l2"
    torch_lbfgs_lr: float = 1.0

    def as_workflow_dict(self) -> dict[str, Any]:
        if self.q_cutoff <= 0 or self.q_loss_threshold <= 0:
            raise WorkflowError("Refinement q cutoff and q-loss threshold must be positive.")
        if self.maxiter < 1 or self.outer_loops < 1:
            raise WorkflowError("Refinement iteration counts must be at least one.")
        return {
            "algorithm": "gradient_laplace",
            "matching_method": self.matching_method,
            "q_cutoff_angstrom_inverse": self.q_cutoff,
            "outer_loops": self.outer_loops,
            "maxiter": self.maxiter,
            "q_loss_threshold": self.q_loss_threshold,
            "percent_matched_threshold": self.percent_matched_threshold,
            "optimizer_method": self.optimizer_method,
            "loss_type": self.loss_type,
            "torch_lbfgs_lr": self.torch_lbfgs_lr,
        }


@dataclass(frozen=True)
class CheckpointInfo:
    path: Path
    source: str
    repository: str | None = None
    revision: str | None = None
    filename: str | None = None
    model_id: str | None = None
    sha256: str | None = None


def resolve_checkpoint(
    *,
    checkpoint: str | Path | None,
    model_repo: str | None,
    revision: str | None = None,
    filename: str = "default_model.ckpt",
    model_filename: str | None = None,
    model_id: str | None = None,
    sha256: str | None = None,
) -> CheckpointInfo:
    """Resolve one explicit local checkpoint or download one from Hugging Face."""
    if model_filename is not None:
        if filename != "default_model.ckpt" and filename != model_filename:
            raise WorkflowError("Pass only one checkpoint filename.")
        filename = model_filename
    if bool(checkpoint) == bool(model_repo):
        raise WorkflowError("Provide exactly one of --checkpoint or --model-repo.")

    if checkpoint:
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise WorkflowError(f"Checkpoint does not exist: {path}")
        return CheckpointInfo(
            path=path,
            source="local",
            filename=path.name,
            model_id=model_id,
            sha256=sha256,
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise WorkflowError(
            "Hugging Face download requires `pip install 'gidind-ai[models]'`."
        ) from exc

    try:
        downloaded = hf_hub_download(
            repo_id=model_repo,
            filename=filename,
            revision=revision,
        )
    except Exception as exc:  # pragma: no cover - exercised with a mocked client
        rev = f" at revision {revision!r}" if revision else ""
        raise WorkflowError(f"Could not download {model_repo!r}{rev}: {exc}") from exc
    return CheckpointInfo(
        path=Path(downloaded),
        source="huggingface",
        repository=model_repo,
        revision=revision,
        filename=filename,
        model_id=model_id,
        sha256=sha256,
    )


def select_device(requested: str):
    import torch

    if requested not in {"auto", "cpu", "cuda"}:
        raise WorkflowError("--device must be one of: auto, cpu, cuda.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise WorkflowError("--device cuda was requested, but CUDA is unavailable.")
    return torch.device(requested)


def _json_value(value: Any) -> Any:
    """Convert NumPy/PyTorch scalars and arrays into JSON-safe values."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_value(value), indent=2, sort_keys=True) + "\n")


def _build_regressor(checkpoint: Path, device):
    """Construct only the geometry needed for portable real-case inference."""
    from gixd_data_generator import GixdDataGenerator
    from gixd_regressor import GixdRegressor

    generator = GixdDataGenerator(
        "uniform", "testing", load_unit_cell_generator=False, calc_intensities=False
    )
    generator.device = device
    generator.laue_indices_tensor = generator.laue_indices_tensor.to(device)
    generators = {"uniform": generator}
    regressor = GixdRegressor(generators, generators, generators)
    regressor.device = device
    regressor.load_checkpoint(checkpoint)
    regressor.gixd_model.eval()
    return regressor, generator


def _load_and_filter_peaks(input_path: Path, feature_config, q_scale: float | None):
    import torch

    from load_peaks import load_peaks_qxy_qz

    q_xy, q_z, scale_used = load_peaks_qxy_qz(
        sample_data=str(input_path), feature_config=feature_config, q_scale=q_scale, max_fill=0.97
    )
    in_bounds = np.logical_and(q_xy < feature_config.max_q_xy, q_z < feature_config.max_q_z)
    if not feature_config.use_specular_scan:
        in_bounds = np.logical_and(in_bounds, q_xy >= feature_config.specular_cutoff)
    q_xy, q_z = q_xy[in_bounds], q_z[in_bounds]
    if len(q_xy) == 0:
        raise WorkflowError("No valid peaks remain after scaling and model-domain filtering.")
    return (
        torch.as_tensor(q_xy, dtype=torch.float64),
        torch.as_tensor(q_z, dtype=torch.float64),
        scale_used,
    )


def _physical_cell(values, q_scale: float):
    values = [float(value) for value in values]
    return [value / q_scale for value in values[:3]] + values[3:]


def _physical_uncertainty(values, q_scale: float):
    values = [float(value) for value in values]
    return [value / q_scale for value in values[:3]] + values[3:]


def _candidate_records(result: dict[str, Any], predictions, miller_indices, q_scale: float):
    candidates = []
    for index, hkl in enumerate(miller_indices):
        refined_key = f"refined{index}_a"
        if refined_key not in result:
            continue
        candidates.append(
            {
                "orientation_index": index,
                "hkl": list(hkl),
                "initial_cell_deg": _physical_cell(predictions[index], q_scale),
                "refined_cell_deg": _physical_cell(
                    [
                        result[f"refined{index}_{name}"]
                        for name in ("a", "b", "c", "alpha", "beta", "gamma")
                    ],
                    q_scale,
                ),
                "laplace_uncertainty": _physical_uncertainty(
                    [
                        result[f"unc{index}_{name}"]
                        for name in ("a", "b", "c", "alpha", "beta", "gamma")
                    ],
                    q_scale,
                ),
                "is_converged": bool(result[f"is_converged{index}"]),
                "q_loss": result[f"q_loss{index}"],
                "percent_matched": result[f"percent_matched{index}"],
            }
        )
    return candidates


def _write_report(path: Path, result: dict[str, Any]) -> None:
    lines = ["gidind-ai real-case inference", ""]
    for candidate in result["candidates"]:
        lines.extend(
            [
                f"Try hkl = {tuple(candidate['hkl'])}",
                f"NN unit cell = {candidate['initial_cell_deg']}",
                f"Refined unit cell = {candidate['refined_cell_deg']}",
                f"Laplace uncertainty = {candidate['laplace_uncertainty']}",
                f"Is converged = {candidate['is_converged']}",
                f"q loss = {candidate['q_loss']}",
                f"Percent matched = {candidate['percent_matched']}",
                "",
            ]
        )
    path.write_text("\n".join(lines))


def _write_figures(
    output_dir: Path, raw_result: dict[str, Any], candidates: list[dict[str, Any]], q_scale: float
) -> None:
    import matplotlib.pyplot as plt

    for candidate in candidates:
        index = candidate["orientation_index"]
        init_plot = raw_result.get(f"init_plot_dict{index}")
        refined_plot = raw_result.get(f"refined_plot_dict{index}")
        if not init_plot or not refined_plot:
            continue
        figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        for axis, title, plot in zip(
            axes,
            ("Neural initialization", "Gradient/Laplace refinement"),
            (init_plot, refined_plot),
        ):
            sim_q = np.asarray(plot["sim_q"]) * q_scale
            axis.scatter(
                np.asarray(plot["test_qxy"]) * q_scale,
                np.asarray(plot["test_qz"]) * q_scale,
                c="tab:red",
                marker="x",
                label="observed",
            )
            if sim_q.size:
                axis.scatter(
                    sim_q[:, 0],
                    sim_q[:, 1],
                    facecolors="none",
                    edgecolors="tab:green",
                    label="simulated",
                )
            axis.set(title=title, xlabel="q_xy (Å⁻¹)", ylabel="q_z (Å⁻¹)")
            axis.legend()
        figure.suptitle(f"HKL {tuple(candidate['hkl'])}")
        figure.savefig(output_dir / f"reconstruction_{index:03d}.png", dpi=150)
        plt.close(figure)


def run_inference(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    checkpoint_info: CheckpointInfo,
    device_name: str = "auto",
    workers: int = 1,
    q_scale: float | None = None,
    figures: bool = True,
    refinement: dict[str, Any] | None = None,
    case_profile: str | None = None,
    refinement_backend: str = "auto",
    refinement_algorithm: str = "gradient_laplace",
    q_sigma: float | None = None,
    mcmc_settings: MCMCSettings | None = None,
) -> dict[str, Any]:
    """Run all orientation candidates for one peak-list input and write reviewer artifacts."""
    if refinement_algorithm not in {"gradient_laplace", "mcmc"}:
        raise WorkflowError("refinement_algorithm must be gradient_laplace or mcmc")
    if refinement_algorithm == "mcmc":
        from .mcmc import validate_sigma

        validate_sigma(q_sigma)
        assert q_sigma is not None
    if workers < 1:
        raise WorkflowError("--workers must be at least 1.")
    path = Path(input_path).expanduser().resolve()
    if not path.is_file():
        raise WorkflowError(f"Input file does not exist: {path}")
    if q_scale is not None and q_scale <= 0:
        raise WorkflowError("--q-scale must be positive.")
    if isinstance(refinement, RefinementSettings):
        refinement = refinement.as_workflow_dict()
    refinement = refinement or {
        "algorithm": "gradient_laplace",
        "matching_method": "simulated_fractional",
        "q_cutoff_angstrom_inverse": 0.06,
        "outer_loops": 10,
        "maxiter": 50,
        "q_loss_threshold": 0.00005,
    }
    q_cutoff = q_loss_threshold = 0.0
    if refinement_algorithm == "gradient_laplace":
        q_cutoff = float(refinement["q_cutoff_angstrom_inverse"])
        q_loss_threshold = float(refinement["q_loss_threshold"])
        if q_cutoff <= 0 or q_loss_threshold <= 0:
            raise WorkflowError("Refinement q cutoff and q-loss threshold must be positive.")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    import torch

    from gixd_utility import GixdUtility

    device = select_device(device_name)
    torch.set_num_threads(workers)
    regressor, generator = _build_regressor(checkpoint_info.path, device)
    q_xy, q_z, scale_used = _load_and_filter_peaks(path, regressor.feature_config, q_scale)
    with torch.no_grad():
        images = generator.calc_images_from_q_points(
            q_xy.unsqueeze(0).to(device), q_z.unsqueeze(0).to(device)
        )
        top_indices = torch.arange(
            len(regressor.feature_config.miller_indices), dtype=torch.int64, device=device
        )
        model_output = regressor.gixd_model(
            [images.repeat(len(top_indices), 1, 1, 1), top_indices]
        ).cpu()
        if isinstance(model_output, tuple):
            means, _, weights = model_output
            model_output = (means * weights.unsqueeze(-1)).sum(1)
        initial_cells = GixdUtility.denormalize_cell_deg(model_output).cpu().tolist()

    if refinement_algorithm == "mcmc":
        from .batched_refinement import Geometry
        from .mcmc import run_mcmc

        assert q_sigma is not None
        sampled = run_mcmc(
            initial_cells,
            torch.stack((q_xy, q_z), -1),
            regressor.feature_config.miller_indices,
            Geometry.from_generator(generator, device),
            q_sigma=q_sigma,
            q_scale=float(scale_used),
            min_height=float(generator.target_config.min_cell_height),
            settings=mcmc_settings,
            output_path=output / "posterior_samples.npz",
        )
        mcmc_result = {
            "input": {"path": str(path), "q_scale": float(scale_used), "num_peaks": len(q_xy)},
            "checkpoint": vars(checkpoint_info) | {"path": str(checkpoint_info.path)},
            "runtime": {
                "device": str(device),
                "workers": workers,
                "algorithm": "mcmc",
                "case_profile": case_profile,
                "refinement_execution": sampled,
            },
            "candidates": sampled["candidates"],
        }
        _write_json(output / "results.json", mcmc_result)
        _write_mcmc_report(output / "report.txt", sampled)
        if figures:
            _write_mcmc_figures(output, sampled)
        return mcmc_result

    from gixd_regressor import _unknown_refine_worker

    raw_result = _unknown_refine_worker(
        {
            "sample_index": 0,
            "pred_cell_deg": initial_cells,
            "q_xy": q_xy.numpy(),
            "q_z": q_z.numpy(),
        },
        data_generator=generator,
        feature_config=regressor.feature_config,
        device=device,
        q_cutoff_1=q_cutoff,
        q_cutoff_2=q_cutoff,
        matching_method=refinement["matching_method"],
        maxiter=int(refinement["maxiter"]),
        outer_loops=int(refinement["outer_loops"]),
        q_loss_threshold=q_loss_threshold,
        return_debug_output=figures,
        refinement_backend=refinement_backend,
    )
    candidates = _candidate_records(
        raw_result, initial_cells, regressor.feature_config.miller_indices, float(scale_used)
    )
    result: dict[str, Any] = {
        "input": {"path": str(path), "q_scale": float(scale_used), "num_peaks": len(q_xy)},
        "checkpoint": {
            "path": str(checkpoint_info.path),
            "source": checkpoint_info.source,
            "repository": checkpoint_info.repository,
            "revision": checkpoint_info.revision,
            "filename": checkpoint_info.filename,
            "model_id": checkpoint_info.model_id,
            "sha256": checkpoint_info.sha256,
        },
        "runtime": {
            "device": str(device),
            "workers": workers,
            "algorithm": "gradient_laplace",
            "case_profile": case_profile,
            "refinement": refinement,
            "refinement_execution": raw_result.get("refinement_metadata", {}),
        },
        "candidates": candidates,
    }
    _write_json(output / "results.json", result)
    _write_report(output / "report.txt", result)
    if figures:
        _write_figures(output, raw_result, candidates, float(scale_used))
    return result


def run_benchmark(
    *,
    output_dir: str | Path,
    checkpoint_info: CheckpointInfo,
    samples: int = 4,
    seed: int = 42,
    device_name: str = "auto",
    workers: int = 1,
    paper_assets=None,
    paper_assets_info=None,
    refinement_backend: str = "auto",
    refinement_algorithm: str = "gradient_laplace",
    q_sigma: float | None = None,
    mcmc_settings: MCMCSettings | None = None,
) -> dict[str, Any]:
    """Run deterministic synthetic cases using the selected refinement algorithm."""
    if refinement_algorithm not in {"gradient_laplace", "mcmc"}:
        raise WorkflowError("refinement_algorithm must be gradient_laplace or mcmc")
    if refinement_algorithm == "mcmc":
        from .mcmc import validate_sigma

        validate_sigma(q_sigma)
        assert q_sigma is not None
    if samples < 1:
        raise WorkflowError("--samples must be at least 1.")
    if workers < 1:
        raise WorkflowError("--workers must be at least 1.")
    if paper_assets is not None and paper_assets_info is not None:
        raise WorkflowError("Provide only one paper-asset source.")
    paper_assets = paper_assets or paper_assets_info
    if paper_assets is None:
        raise WorkflowError("A pinned paper-asset release is required for synthetic benchmarks.")
    import torch

    from gixd_data_generator import create_data_generators
    from gixd_regressor import GixdRegressor

    np.random.seed(seed)
    torch.manual_seed(seed)
    device = select_device(device_name)
    torch.set_num_threads(workers)
    generators = create_data_generators(
        "uniform",
        ["cell", "top_layer"],
        unit_cell_generator_path=str(paper_assets.generator_path),
    )
    for generator in generators[2].values():
        if hasattr(generator, "rng"):
            generator.rng = np.random.default_rng(seed)
        if hasattr(generator, "device"):
            generator.device = device
            generator.laue_indices_tensor = generator.laue_indices_tensor.to(device)
    regressor = GixdRegressor(*generators)
    regressor.device = device
    regressor.load_checkpoint(checkpoint_info.path)
    generator = generators[2]["uniform"]
    cell_targets, images, _, q_xy, q_z, top_indices = generator.generate_data_batch(samples)
    if refinement_algorithm == "mcmc":
        # Synthetic MCMC benchmarks use explicit, known measurement noise.
        valid_peaks = (q_xy > 0) | (q_z > 0)
        q_xy = torch.where(valid_peaks, q_xy + torch.randn_like(q_xy) * q_sigma, q_xy)
        q_z = torch.where(valid_peaks, q_z + torch.randn_like(q_z) * q_sigma, q_z)
        images = generator.calc_images_from_q_points(q_xy, q_z)
    initial_cells = regressor._predict_cells_from_images(images, top_indices)
    from .batched_refinement import Geometry, refine_batch, resolve_backend, result_rows

    mcmc_rows = None
    if refinement_algorithm == "mcmc":
        from .mcmc import run_mcmc

        assert q_sigma is not None
        mcmc_rows = []
        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        for index in range(samples):
            sampled = run_mcmc(
                np.asarray(initial_cells[index])[None],
                torch.stack((q_xy[index], q_z[index]), -1),
                [regressor.feature_config.miller_indices[int(top_indices[index])]],
                Geometry.from_generator(generator, device),
                q_sigma=q_sigma,
                min_height=float(generator.target_config.min_cell_height),
                settings=mcmc_settings,
                output_path=output / f"posterior_samples_{index}.npz",
            )
            mcmc_rows.append(sampled)
        execution = {"backend": "mcmc", "samples": mcmc_rows}
        backend = "mcmc"
    else:
        from polishing_fit import direct_inversion_pytorch

        backend = resolve_backend(refinement_backend, device)
    batch_rows = None
    if mcmc_rows is None:
        execution = {"backend": backend, "optimizer": "cg"}
    if backend == "batched":
        observations = torch.stack((q_xy, q_z), -1)
        result_batch = refine_batch(
            torch.as_tensor(initial_cells, dtype=torch.float64, device=device),
            observations,
            torch.as_tensor(regressor.feature_config.miller_indices, device=device)[
                top_indices.long()
            ],
            Geometry.from_generator(generator, device),
            matching_method="simulated_fractional",
            q_cutoff=0.06,
            maxiter=50,
            outer_loops=10,
        )
        batch_rows = result_rows(result_batch, observations)
        execution = {
            "backend": backend,
            "optimizer": result_batch["optimizer"],
            "timings": result_batch["timings"],
        }
    rows = []
    for index in range(samples):
        true_cell = np.concatenate((cell_targets[index, :3], np.rad2deg(cell_targets[index, 3:])))
        if mcmc_rows is not None:
            candidate = mcmc_rows[index]["candidates"][0]
            refined_cell = np.asarray(candidate.get("refined_cell_deg", [float("nan")] * 6))
            benchmark_candidate = candidate | {
                "sample_index": index,
                "true_cell_deg": true_cell,
                "initial_absolute_error": np.abs(initial_cells[index] - true_cell),
                "refined_absolute_error": np.abs(refined_cell - true_cell),
                "samples_path": mcmc_rows[index]["samples_path"],
            }
            rows.append(benchmark_candidate)
            continue
        refined = (
            batch_rows[index]
            if batch_rows is not None
            else direct_inversion_pytorch(
                torch.as_tensor(initial_cells[index], dtype=torch.float64, device=device),
                q_xy[index],
                q_z[index],
                regressor.feature_config.miller_indices[int(top_indices[index])],
                generator,
                method="cg",
                loss_type="l2",
                q_cutoff=0.06,
                maxiter=50,
                outer_loops=10,
                matching_method="simulated_fractional",
                return_uncertainties=True,
            )
        )
        refined_cell = refined["refined_predicted_cell_deg"].detach().cpu().numpy()
        rows.append(
            {
                "sample_index": index,
                "hkl": list(regressor.feature_config.miller_indices[int(top_indices[index])]),
                "true_cell_deg": true_cell,
                "initial_cell_deg": initial_cells[index],
                "refined_cell_deg": refined_cell,
                "initial_absolute_error": np.abs(initial_cells[index] - true_cell),
                "refined_absolute_error": np.abs(refined_cell - true_cell),
                "laplace_uncertainty": refined["refined_uncertainty_cell_deg"],
                "q_loss": refined["refined_q_loss"],
                "refinement_iterations": refined.get("iterations"),
                "refinement_loss_history": refined.get("loss_history", []),
            }
        )
    if mcmc_rows is not None:
        # Failed starts remain in rows, but do not turn the numerical summary into NaNs.
        for row in rows:
            if not np.isfinite(row["refined_absolute_error"]).all():
                row["refined_absolute_error"] = [None] * 6
    initial_errors = np.asarray([row["initial_absolute_error"] for row in rows])
    refined_errors = np.asarray([row["refined_absolute_error"] for row in rows], dtype=float)
    result: dict[str, Any] = {
        "seed": seed,
        "samples": samples,
        "algorithm": refinement_algorithm,
        "refinement_execution": execution,
        "checkpoint": {
            "source": checkpoint_info.source,
            "repository": checkpoint_info.repository,
            "revision": checkpoint_info.revision,
            "filename": checkpoint_info.filename,
            "model_id": checkpoint_info.model_id,
            "sha256": checkpoint_info.sha256,
        },
        "paper_assets": {
            "source": paper_assets.source,
            "repository": paper_assets.repository,
            "revision": paper_assets.revision,
            "release_id": paper_assets.release_id,
            "manifest_path": str(paper_assets.manifest_path),
            "sampler_path": str(paper_assets.generator_path),
        },
        "rows": rows,
        "summary": {
            "initial_mae": float(initial_errors.mean()),
            "refined_mae": _finite_mean(refined_errors),
            "initial_distance_mae": float(initial_errors[:, :3].mean()),
            "refined_distance_mae": _finite_mean(refined_errors[:, :3]),
            "initial_angle_mae": float(initial_errors[:, 3:].mean()),
            "refined_angle_mae": _finite_mean(refined_errors[:, 3:]),
        },
    }
    if mcmc_rows is not None:
        intervals = [
            (row, np.asarray(row["credible_interval_95"]))
            for row in rows
            if "credible_interval_95" in row
        ]
        coverage = [
            (row["true_cell_deg"] >= interval[:, 0]) & (row["true_cell_deg"] <= interval[:, 1])
            for row, interval in intervals
        ]
        result["summary"].update(
            {
                "successful_initializations": len(intervals),
                "converged_samples": sum(row["is_converged"] for row in rows),
                "marginal_95_coverage": np.mean(coverage, axis=0).tolist() if coverage else None,
                "coverage_note": "Descriptive coverage includes unconverged runs; inspect diagnostics.",
                "injected_q_sigma_angstrom_inverse": q_sigma,
            }
        )
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "benchmark.json", result)
    (output / "report.txt").write_text(
        (
            "gidind-ai ensemble MCMC synthetic benchmark\n\n"
            if refinement_algorithm == "mcmc"
            else "gidind-ai fast synthetic benchmark\n\n"
        )
        + f"seed: {seed}\nsamples: {samples}\n"
        f"initial MAE: {result['summary']['initial_mae']:.6g}\n"
        f"refined MAE: {_format_mae(result['summary']['refined_mae'])}\n"
    )
    return result


def _finite_mean(values):
    finite = np.asarray(values)[np.isfinite(values)]
    return float(finite.mean()) if finite.size else None


def _write_mcmc_report(path: Path, sampled: dict) -> None:
    lines = [
        "gidind-ai ensemble MCMC refinement",
        "",
        "Scores are maximum retained Gaussian residual log likelihoods, not evidence.",
        f"Physical q sigma: {sampled['q_sigma_angstrom_inverse']} inverse angstroms",
        "",
    ]
    for candidate in sampled["candidates"]:
        lines += [f"Try hkl = {tuple(candidate['hkl'])}", f"Status = {candidate['status']}"]
        if "refined_cell_deg" in candidate:
            lines += [
                f"Posterior mean cell = {candidate['refined_cell_deg']}",
                f"Posterior standard deviation = {candidate['posterior_std']}",
                f"95% intervals = {candidate['credible_interval_95']}",
                f"Goodness-of-fit log likelihood = {candidate['goodness_of_fit_log_likelihood']}",
                f"Diagnostics = {candidate['diagnostics']}",
            ]
        else:
            lines += [f"Initialization failures = {candidate['initialization_failures']}"]
        lines.append("")
    path.write_text("\n".join(lines))


def _write_mcmc_figures(output: Path, sampled: dict) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    with np.load(sampled["samples_path"]) as archive:
        draws = archive["samples"]
        for candidate in sampled["candidates"]:
            if candidate.get("goodness_of_fit_rank", 1000) > 5:
                continue
            index = candidate["orientation_index"]
            flat = draws[index].reshape(-1, 6)
            fig = Figure(figsize=(10, 6))
            FigureCanvasAgg(fig)
            axes = fig.subplots(2, 3)
            for parameter, (ax, label) in enumerate(
                zip(axes.flat, ("a (Å)", "b (Å)", "c (Å)", "α (°)", "β (°)", "γ (°)"))
            ):
                ax.hist(flat[:, parameter], bins=40, density=True)
                ax.set_xlabel(label)
            fig.suptitle(f"hkl {candidate['hkl']}: {candidate['status']}")
            fig.tight_layout()
            fig.savefig(output / f"posterior_orientation_{index}.png", dpi=140)
            fig.clear()


def _format_mae(value):
    return "unavailable" if value is None else f"{value:.6g}"
