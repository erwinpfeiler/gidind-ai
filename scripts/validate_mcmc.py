#!/usr/bin/env python3
"""Checkpoint-free CPU/CUDA MCMC validation and orientation-batching benchmark.

Run with the installed gidind-ai package (or PYTHONPATH=src). Full runs report
empirical coverage; --quick validates execution only, not convergence/coverage.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from gidind_ai.batched_refinement import Geometry, peak_mask, simulate
from gidind_ai.mcmc import MCMCSettings, PeakPosterior, run_mcmc


def geometry(device, index_limit):
    laue = torch.tensor(
        list(itertools.product(range(-index_limit, index_limit + 1), repeat=3)),
        device=device,
        dtype=torch.float64,
    )
    return Geometry(
        torch.tensor([4.0] * 3 + [60.0] * 3, device=device, dtype=torch.float64),
        torch.tensor([26.0] * 3 + [60.0] * 3, device=device, dtype=torch.float64),
        laue[laue.abs().sum(-1) > 0],
        2.0,
        2.0,
        False,
        0.05,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--max-laue-index", type=int, default=6)
    args = parser.parse_args()
    if args.trials < 1 or args.max_laue_index < 1:
        parser.error("trials and max-laue-index must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error(
            "CUDA unavailable; validation remains pending (use --device cpu for CPU checks)"
        )
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    g = geometry(args.device, args.max_laue_index)
    cpu = geometry("cpu", args.max_laue_index)
    truth = torch.tensor(
        [[8.0, 9.0, 10.0, 82.0, 85.0, 87.0]], dtype=torch.float64, device=args.device
    )
    orientations = torch.tensor([[0, 0, 1], [1, 0, 0], [1, 0, 2]], device=args.device)
    initial = truth.repeat(3, 1)
    initial[:, :3] += 0.05
    initial[:, 3:] += 0.1
    points = simulate(truth[0], orientations[0], g.laue)
    points = torch.unique(points[peak_mask(points, g, simulated=True)], dim=0)
    # A fixed subset bounds memory and provides well/weakly constrained cases.
    points = points[:: max(1, len(points) // 24)][:24]
    sigma = 0.003
    config = MCMCSettings()
    if args.quick:
        config = replace(config, ensembles=2, walkers=8, burn_in=20, production=40, thin=2)
    rng = np.random.default_rng(349)
    runs = []
    coverage = {"well_constrained": [], "weakly_constrained": []}
    for case, case_coverage in coverage.items():
        for trial in range(1 if args.quick else args.trials):
            observed = points if case == "well_constrained" else points[:3]
            observed = observed + torch.as_tensor(
                rng.normal(0, sigma, observed.shape), device=args.device, dtype=torch.float64
            )
            x = ((initial - g.minimum) / g.span)[:, None, None].expand(3, 2, 4, 6)
            p_gpu = PeakPosterior(observed, orientations, g, sigma, 1.5)(x).cpu()
            p_cpu = PeakPosterior(observed.cpu(), orientations.cpu(), cpu, sigma, 1.5)(x.cpu())
            torch.testing.assert_close(p_gpu, p_cpu, rtol=1e-10, atol=1e-9)
            previous_samples = previous_scores = None
            for chunk in (1, 3):
                if args.device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                path = args.output_dir / f"{case}_{trial}_chunk{chunk}.npz"
                result = run_mcmc(
                    initial,
                    observed,
                    orientations,
                    g,
                    q_sigma=sigma,
                    settings=replace(
                        config, seed=config.seed + trial, orientation_chunk_size=chunk
                    ),
                    output_path=path,
                )
                with np.load(path) as archive:
                    samples, scores = archive["samples"], archive["log_likelihood"]
                    if previous_samples is not None:
                        np.testing.assert_allclose(
                            samples, previous_samples, rtol=1e-10, atol=1e-10
                        )
                        np.testing.assert_allclose(scores, previous_scores, rtol=1e-10, atol=1e-9)
                    previous_samples, previous_scores = samples.copy(), scores.copy()
                candidates = result["candidates"]
                seconds = result["timings"]["sampling_seconds"]
                ess = sum(c.get("diagnostics", {}).get("min_ess_bulk", 0) for c in candidates)
                result["performance"] = {
                    "estimated_min_bulk_ess_per_second": ess / seconds,
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
                    if args.device == "cuda"
                    else None,
                }
                result["case"], result["trial"], result["orientation_chunk_size"] = (
                    case,
                    trial,
                    chunk,
                )
                runs.append(result)
                if chunk == 1 and "credible_interval_95" in candidates[0]:
                    interval = np.asarray(candidates[0]["credible_interval_95"])
                    true = truth[0].cpu().numpy()
                    case_coverage.append(
                        {
                            "covered": (
                                (true >= interval[:, 0]) & (true <= interval[:, 1])
                            ).tolist(),
                            "is_converged": candidates[0]["is_converged"],
                            "absolute_error": abs(
                                np.asarray(candidates[0]["refined_cell_deg"]) - true
                            ).tolist(),
                        }
                    )
    report = {
        "device": args.device,
        "torch_version": torch.__version__,
        "device_name": torch.cuda.get_device_name() if args.device == "cuda" else "CPU",
        "cuda_parity_and_chunking_passed": args.device == "cuda",
        "cuda_validation_status": "parity_and_execution_passed"
        if args.device == "cuda"
        else "pending",
        "scientific_validation_status": "quick_execution_only"
        if args.quick
        else "inspect_coverage_and_diagnostics",
        "coverage_trials": coverage,
        "runs": runs,
    }
    (args.output_dir / "validation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(f"Wrote {args.output_dir / 'validation.json'}")


if __name__ == "__main__":
    main()
