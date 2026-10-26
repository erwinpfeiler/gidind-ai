"""Freeze common neural initializations and compare serial/batched refinement.

Run ``python -m gidind_ai.refinement_validation --help``. This runner does not
enable the CUDA release gate or replace published paper results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .batched_refinement import Geometry, refine_batch, result_rows, simulate, valid_cells
from .workflows import _build_regressor, _load_and_filter_peaks

# Frozen comparison profiles: independent of unpublished release tooling.
_PROFILES: dict[str, Any] = {
    "schema_version": 1,
    "models": [
        {
            "id": "standard",
            "repository": "epfeiler/gidind-ai",
            "revision": "374aed64cf921eb47e3e03452c4337e0e636ff93",
            "filename": "default_model.ckpt",
            "sha256": "b7efe2de22fdac9ece25b954634ffbc0a6df5570b01b80a2212a80cdae34dc2a",
            "intended_use": "PQ, Cu-BOD, MAPbBr3, and synthetic benchmarks.",
            "input_domain_angstrom_inverse": {"q_xy": [0.0, 2.0], "q_z": [0.0, 2.0]},
            "scaling": "Use one shared scale from the shorter q axis when no case-specific q_scale is declared; discard out-of-domain values rather than clipping.",
            "refinement": {
                "algorithm": "gradient_laplace",
                "matching_method": "simulated_fractional",
                "q_cutoff_angstrom_inverse": 0.06,
                "outer_loops": 10,
                "maxiter": 50,
                "q_loss_threshold": 5e-05,
            },
        },
        {
            "id": "high_missingness",
            "repository": "epfeiler/gidind-ai",
            "revision": "f1cfccf0db3781747dfa7b184a448ac68e47adff",
            "filename": "real_cases_model.ckpt",
            "sha256": "be7f62e28de4ffe3ee6bfffa9569b238e445986f7de577f3ddd6cd2ce24398ce",
            "intended_use": "NAIB-S11, whose missing reflections are better handled by the high-missingness model.",
            "training_provenance": {
                "specular_scan": False,
                "missing_peak_rate_max": 0.3,
                "missing_peak_masking": "post-rasterisation",
                "q_coordinate_jitter_sigma_angstrom_inverse": 0.005,
            },
            "input_domain_angstrom_inverse": {"q_xy": [0.0, 2.0], "q_z": [0.0, 2.0]},
            "scaling": "NAIB-S11 uses the documented fixed q_scale of 0.9.",
            "refinement": {
                "algorithm": "gradient_laplace",
                "matching_method": "one2one",
                "q_cutoff_angstrom_inverse": 0.15,
                "outer_loops": 10,
                "maxiter": 10,
                "q_loss_threshold": 5e-05,
            },
        },
    ],
    "real_cases": [
        {"id": "pq", "model_id": "standard", "q_scale": None},
        {
            "id": "cu_bod_mof",
            "model_id": "standard",
            "q_scale": 1.0,
            "refinement": {
                "algorithm": "gradient_laplace",
                "matching_method": "hybrid",
                "q_cutoff_angstrom_inverse": 0.05,
                "outer_loops": 10,
                "maxiter": 10,
                "q_loss_threshold": 5e-05,
            },
        },
        {"id": "mapbbr3", "model_id": "standard", "q_scale": None},
        {"id": "naib_s11", "model_id": "high_missingness", "q_scale": 0.9},
    ],
}


def release_model(model_id):
    return next(model for model in _PROFILES["models"] if model["id"] == model_id)


def real_case_profile(case_id):
    return next(case for case in _PROFILES["real_cases"] if case["id"] == case_id)


def prepare(peak_dir: Path, synthetic_per_system: int, seed: int) -> list[dict[str, Any]]:
    """Prepare once on CPU; every measured backend receives these exact cells."""
    from huggingface_hub import hf_hub_download

    from niggli import cell_to_niggli

    cases: list[dict[str, Any]] = []
    standard = None
    rng = np.random.default_rng(seed)
    for case_id in ("pq", "cu_bod_mof", "mapbbr3", "naib_s11"):
        profile = real_case_profile(case_id)
        model = release_model(profile["model_id"])
        checkpoint = Path(
            hf_hub_download(
                repo_id=model["repository"], revision=model["revision"], filename=model["filename"]
            )
        )
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != model["sha256"]:
            raise ValueError(f"Checkpoint checksum mismatch: {checkpoint}")
        if model["id"] == "standard" and standard is not None:
            regressor, generator = standard
        else:
            regressor, generator = _build_regressor(checkpoint, torch.device("cpu"))
            if model["id"] == "standard":
                standard = regressor, generator
        path = peak_dir / f"{case_id}_without_specular.csv"
        qxy, qz, scale = _load_and_filter_peaks(
            path, regressor.feature_config, profile.get("q_scale")
        )
        geometry = Geometry.from_generator(generator, "cpu")
        orientations = torch.tensor(regressor.feature_config.miller_indices)
        observations = torch.stack((qxy, qz), -1).to(torch.float64)
        image = generator.calc_images_from_q_points(qxy[None], qz[None])
        initial = regressor._predict_cells_from_images(
            image.repeat(len(orientations), 1, 1, 1), torch.arange(len(orientations))
        )
        cases.append(
            {
                "id": case_id,
                "kind": "experimental",
                "observed": observations.tolist(),
                "initial_cells": initial.tolist(),
                "orientations": orientations.tolist(),
                "q_scale": scale,
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "model_revision": model["revision"],
                "settings": profile.get("refinement", model["refinement"]),
            }
        )
    assert standard is not None
    regressor, generator = standard
    geometry = Geometry.from_generator(generator, "cpu")
    orientations = torch.tensor(regressor.feature_config.miller_indices)
    for system in (
        "triclinic",
        "monoclinic",
        "orthorhombic",
        "tetragonal",
        "rhombohedral",
        "hexagonal",
        "cubic",
    ):
        if not synthetic_per_system:
            break
        path = hf_hub_download(
            repo_id="epfeiler/gidind-ai-paper-assets",
            repo_type="dataset",
            revision="8a670198023ae5872a8379bdc448591112c26ba6",
            filename=f"splits/test_unit_cells/{system}.npy",
        )
        source = np.loadtxt(path)
        selected = 0
        for row_index in rng.permutation(len(source)):
            truth = torch.tensor(cell_to_niggli(source[row_index]), dtype=torch.float64)
            if not bool(valid_cells(truth)) or not bool(
                ((truth >= geometry.minimum) & (truth <= geometry.minimum + geometry.span)).all()
            ):
                continue
            orientation_index = int(rng.integers(len(orientations)))
            observations = simulate(truth, orientations[orientation_index], geometry.laue)
            mask = (
                (observations[:, 0] > geometry.specular_cutoff)
                & (observations[:, 0] < geometry.q_xy_max)
                & (observations[:, 1] < geometry.q_z_max)
            )
            observations = observations[mask]
            if len(observations) < 6:
                continue
            # Exact duplicates are not independent observations.
            observations = torch.unique(observations, dim=0)
            image = generator.calc_images_from_q_points(
                observations[:, 0][None], observations[:, 1][None]
            )
            initial = regressor._predict_cells_from_images(
                image.repeat(len(orientations), 1, 1, 1), torch.arange(len(orientations))
            )
            cases.append(
                {
                    "id": f"{system}_{selected:03d}",
                    "kind": "synthetic",
                    "system": system,
                    "truth": truth.tolist(),
                    "true_orientation_index": orientation_index,
                    "observed": observations.tolist(),
                    "initial_cells": initial.tolist(),
                    "orientations": orientations.tolist(),
                    "source_row_index": int(row_index),
                    "source_revision": "8a670198023ae5872a8379bdc448591112c26ba6",
                    "settings": release_model("standard")["refinement"],
                }
            )
            selected += 1
            if selected == synthetic_per_system:
                break
        if selected < synthetic_per_system:
            raise ValueError(f"Only {selected} eligible held-out cells found for {system}")
    return cases


def evaluate(case: dict[str, Any], backend: str, device: str, quick: bool) -> dict[str, Any]:
    from unittest.mock import patch

    import polishing_fit
    from gixd_data_generator import GixdDataGenerator
    from polishing_fit import direct_inversion_pytorch

    generator = GixdDataGenerator("uniform", "testing", load_unit_cell_generator=False)
    generator.device = torch.device(device)
    generator.laue_indices_tensor = generator.laue_indices_tensor.to(device)
    geometry = Geometry.from_generator(generator, device)
    initial = torch.tensor(case["initial_cells"], device=device, dtype=torch.float64)
    observed = torch.tensor(case["observed"], device=device, dtype=torch.float64)
    orientations = torch.tensor(case["orientations"], device=device)
    settings = case["settings"]
    options = {
        "q_cutoff": settings["q_cutoff_angstrom_inverse"],
        "maxiter": 5 if quick else settings["maxiter"],
        "outer_loops": 2 if quick else settings["outer_loops"],
        "matching_method": settings["matching_method"],
        "return_uncertainties": True,
    }
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    if backend == "batched":
        result = refine_batch(initial, observed, orientations, geometry, **options)
        rows = result_rows(result, observed)
        phases = result["timings"]
    else:
        # Instrument the reference externally; do not alter its optimizer.
        phases = dict.fromkeys(("matching_ms", "optimization_ms", "uncertainty_ms"), 0.0)

        timing_stack = []

        def timed(function, key):
            def wrapper(*args, **kwargs):
                if device == "cuda":
                    torch.cuda.synchronize()
                phase_start = time.perf_counter()
                timing_stack.append(0.0)
                value = function(*args, **kwargs)
                if device == "cuda":
                    torch.cuda.synchronize()
                duration = (time.perf_counter() - phase_start) * 1000
                children = timing_stack.pop()
                phases[key] += duration - children
                if timing_stack:
                    timing_stack[-1] += duration
                return value

            return wrapper

        with (
            patch.object(
                polishing_fit,
                "match_laue_indices",
                timed(polishing_fit.match_laue_indices, "matching_ms"),
            ),
            patch.object(
                polishing_fit,
                "torch_minimize",
                timed(polishing_fit.torch_minimize, "optimization_ms"),
            ),
            patch.object(polishing_fit, "hessian", timed(polishing_fit.hessian, "uncertainty_ms")),
            patch.object(polishing_fit, "ls_res", timed(polishing_fit.ls_res, "uncertainty_ms")),
        ):
            rows = [
                direct_inversion_pytorch(
                    initial[i],
                    observed[:, 0],
                    observed[:, 1],
                    tuple(orientation),
                    generator,
                    method="cg",
                    **options,
                )
                for i, orientation in enumerate(case["orientations"])
            ]

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    phases["other_ms"] = max(
        0.0, elapsed - sum(value for key, value in phases.items() if key != "total_ms")
    )
    cells = torch.stack([row["refined_predicted_cell_deg"].to(device) for row in rows])
    uncertainties = torch.stack([row["refined_uncertainty_cell_deg"].to(device) for row in rows])
    losses = torch.tensor([float(row["refined_q_loss"]) for row in rows], device=device)
    matches = torch.tensor(
        [len(row["refined_plot_dict"]["indices_match"]) for row in rows], device=device
    )
    accepted = (
        torch.isfinite(uncertainties).all(-1)
        & (uncertainties[:, :3] <= 0.1).all(-1)
        & (uncertainties[:, 3:] <= 1.0).all(-1)
        & (losses <= settings["q_loss_threshold"])
        & (matches / max(len(observed), 1) >= 0.9)
    )
    output = {
        "cells": cells.cpu().tolist(),
        "uncertainties": uncertainties.cpu().tolist(),
        "losses": losses.cpu().tolist(),
        "num_matches": matches.cpu().tolist(),
        "failed": (
            (matches == 0) | ~torch.isfinite(losses) | ~torch.isfinite(uncertainties).all(-1)
        )
        .cpu()
        .tolist(),
        "iterations": [row.get("iterations") for row in rows],
        "accepted": accepted.cpu().tolist(),
        "total_ms": elapsed,
        "phase_timings": phases,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
    }
    if case.get("truth"):
        error = (cells - cells.new_tensor(case["truth"])).abs()
        output["correct_cells"] = (
            ((error[:, :3] < 0.1).all(-1) & (error[:, 3:] < 0.1).all(-1)).cpu().tolist()
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peak-dir", type=Path, default=Path("gidind_peak_lists"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepared-inputs", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--synthetic-per-system", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", default="cpu,cuda", choices=("cpu", "cuda", "cpu,cuda"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Integration smoke test, not scientific/timing evidence",
    )
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0 or args.synthetic_per_system < 0 or args.threads < 1:
        parser.error("Invalid repetitions, sample count or CPU thread count")
    if "cuda" in args.devices and not args.prepare_only and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --devices cpu for a CPU-only diagnostic")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared = args.prepared_inputs or args.output_dir / "prepared_inputs.json"
    if args.prepared_inputs:
        cases = json.loads(prepared.read_text())["cases"]
    else:
        if prepared.exists():
            parser.error("Prepared inputs already exist; pass --prepared-inputs to reuse them")
        cases = prepare(args.peak_dir, args.synthetic_per_system, args.seed)
        prepared.write_text(json.dumps({"seed": args.seed, "cases": cases}, indent=2) + "\n")
    if args.prepare_only:
        print(f"Saved {len(cases)} cases with frozen common initializations to {prepared}")
        return
    records = []
    for case in cases:
        for device in args.devices.split(","):
            for backend in ("serial", "batched"):
                for _ in range(args.warmup):
                    evaluate(case, backend, device, args.quick)
                runs = [evaluate(case, backend, device, args.quick) for _ in range(args.repeats)]
                durations = [run["total_ms"] for run in runs]
                records.append(
                    {
                        "case": case["id"],
                        "device": device,
                        "backend": backend,
                        "median_ms": statistics.median(durations),
                        "min_ms": min(durations),
                        "max_ms": max(durations),
                        "runs": runs,
                    }
                )
                print(
                    case["id"],
                    device,
                    backend,
                    f"median={statistics.median(durations):.2f}ms",
                    flush=True,
                )
                report = {
                    "quick_smoke_test": args.quick,
                    "torch": torch.__version__,
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "cpu": platform.processor(),
                    "cpu_threads": args.threads,
                    "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                    "cuda": torch.version.cuda,
                    "warmup": args.warmup,
                    "repeats": args.repeats,
                    "prepared_inputs_sha256": hashlib.sha256(prepared.read_bytes()).hexdigest(),
                    "records": records,
                    "cuda_default_enabled": False,
                }
                (args.output_dir / "comparison.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )


if __name__ == "__main__":
    main()
