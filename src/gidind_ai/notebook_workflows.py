"""Small, reproducible reviewer workflows shared by the four release notebooks."""

from __future__ import annotations
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np


def initialize(root):
    root = Path(root).resolve()
    os.environ.setdefault("GIDIND_AI_ASSETS", str(root / ".assets"))
    os.environ.setdefault("TORCH_HOME", str(root / ".assets/torch"))
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".assets/matplotlib"))
    import torch

    torch.set_num_threads(1)
    from .setup_assets import prepare_assets

    prepare_assets(root / ".assets", download=False)
    print("Assets verified. Device:", "cuda" if torch.cuda.is_available() else "cpu")
    return root


def _args(root, samples, seed, device, *, baseline=False, full_grid=False):
    import torch

    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    return argparse.Namespace(
        run_dir=Path(root),
        samples=samples,
        seed=seed,
        baseline_only=baseline,
        max_peaks=20,
        peak_step=1 if full_grid else 5,
        noise_steps=21 if full_grid else 5,
        max_noise=0.02,
        noise_mode="gaussian",
        device=device,
        require_a100=False,
        checkpoint=None,
        allow_download=False,
        batch_size=16,
        cpu_threads=1,
        workers=1,
        condition_index=None,
        cohort_dir=None,
        print_condition_count=False,
        progress_seconds=30,
        maxiter=500,
        outer_loops=20,
        q_cutoff=0.056,
    )


def _show(path):
    from IPython.display import SVG, display

    display(SVG(filename=str(path)))


def _save(fig, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf"):
        fig.savefig(directory / f"{name}.{ext}", bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    _show(directory / f"{name}.svg")


def run_convergence(root, samples=100, seed=42, device="auto"):
    from .figure3_benchmark import run, database

    name = "convergence" if (samples, seed) == (100, 42) else f"convergence_n{samples}_seed{seed}"
    directory = Path(root) / "reviewer_outputs" / name
    args = _args(directory, samples, seed, device, baseline=True)
    run(args)
    db = database(directory)
    records = [json.loads(r[0]) for r in db.execute("SELECT result FROM fits ORDER BY sample")]
    db.close()
    # Persist full predictions/errors; failed fits count in the fixed denominator.
    from .figure3_assets import SYSTEMS

    rows = []
    for i, r in enumerate(records):
        before = np.abs(np.asarray(r["initial"]) - r["truth"])
        after = np.abs(
            np.asarray(r["refined"] if r["refined"] is not None else [np.inf] * 6) - r["truth"]
        )
        rows.append(
            dict(
                sample=i,
                system=SYSTEMS[r["system"]],
                orientation=r["orientation_index"],
                nn_correct=r["nn_correct"],
                refined_correct=r["correct"],
                nn_length_mae=float(before[:3].mean()),
                nn_angle_mae=float(before[3:].mean()),
                refined_length_mae=float(after[:3].mean()),
                refined_angle_mae=float(after[3:].mean()),
                failure=r["error"],
            )
        )
    with (directory / "per_cell.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(
        samples=samples,
        nn_recovered=sum(r["nn_correct"] for r in rows),
        refined_recovered=sum(r["refined_correct"] for r in rows),
        convergence_rate=sum(r["refined_correct"] for r in rows) / samples,
        failures=sum(r["failure"] is not None for r in rows),
        criterion="all lengths <0.1 A AND all angles <0.1 degree",
        known_contact_plane=True,
    )
    (directory / "convergence.json").write_text(json.dumps(summary, indent=2) + "\n")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, parameter, unit in zip(axes, ("length", "angle"), ("Å", "°")):
        for j, system in enumerate(SYSTEMS):
            for arm, offset, color in [("nn", -0.17, "#c44e52"), ("refined", 0.17, "#1f77b4")]:
                vals = np.array(
                    [r[f"{arm}_{parameter}_mae"] for r in rows if r["system"] == system]
                )
                vals = vals[np.isfinite(vals)]
                if not len(vals):
                    continue
                vals = np.maximum(vals, 1e-10)  # display floor only; raw errors remain in CSV
                if len(vals) > 1 and np.ptp(vals) > 0:
                    violin = ax.violinplot(
                        vals, positions=[j + offset], widths=0.3, showmedians=True
                    )
                    for body in violin["bodies"]:
                        body.set_facecolor(color)
                        body.set_alpha(0.65)
                    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
                        violin[key].set_color(color)
                else:
                    ax.scatter([j + offset], vals[:1], color=color, s=10)
        ax.axhline(0.1, color="black", ls="--", lw=0.8)
        ax.set_yscale("log")
        ax.set_xticks(range(7), SYSTEMS, rotation=45, ha="right")
        ax.set_ylabel(f"Mean absolute {parameter} error ({unit})")
    from matplotlib.lines import Line2D

    fig.legend(
        handles=[
            Line2D([], [], color="#c44e52", lw=4, label="NN only"),
            Line2D([], [], color="#1f77b4", lw=4, label="NN + refinement"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(f"{summary['refined_recovered']}/{samples} recovered; known contact plane", y=1.06)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, directory, "convergence_violins")
    print(json.dumps(summary, indent=2))
    return summary


def run_figure3(root, samples=100, seed=42, device="auto", full_grid=False):
    name = (
        "figure3bc"
        if (samples, seed, full_grid) == (100, 42, False)
        else f"figure3bc_{'full' if full_grid else 'preview'}_n{samples}_seed{seed}"
    )
    args = _args(Path(root) / "reviewer_outputs" / name, samples, seed, device, full_grid=full_grid)
    # Run in an importable subprocess so CPU condition workers also work in Jupyter.
    workers = min(4, os.cpu_count() or 1) if args.device == "cpu" else 1
    command = [
        sys.executable,
        "-m",
        "gidind_ai.figure3_benchmark",
        "run",
        "--run-dir",
        str(args.run_dir),
        "--samples",
        str(samples),
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--peak-step",
        str(args.peak_step),
        "--noise-steps",
        str(args.noise_steps),
        "--workers",
        str(workers),
        "--outer-loops",
        "20",
        "--maxiter",
        "500",
        "--q-cutoff",
        "0.056",
    ]
    print(
        f"{(41 if full_grid else 9) + (20 if full_grid else 4)} settings × {samples} cells. Results saved after every fit."
    )
    subprocess.run(command, check=True)
    from .paper_plots import figure3

    figure3(args.run_dir)
    _show(args.run_dir / "figure3bc_95ci.svg")
    return str(args.run_dir)


def run_real_case(root, case_id, device="auto"):
    from .workflows import resolve_checkpoint, run_inference, _load_and_filter_peaks
    from .release_models import real_case_profile, release_model
    import torch
    from .batched_refinement import Geometry, simulate
    from gixd_data_generator import GixdDataGenerator

    root = Path(root)
    profile = real_case_profile(case_id)
    model = release_model("high_missingness")
    output = root / "reviewer_outputs/real_cases" / case_id
    info = resolve_checkpoint(
        checkpoint=root / ".assets/models" / model["filename"],
        model_repo=None,
        model_id=model["id"],
        sha256=model["sha256"],
    )
    result = run_inference(
        input_path=root / ".assets/real_cases" / f"{case_id}_without_specular.csv",
        output_dir=output,
        checkpoint_info=info,
        device_name=device,
        workers=1,
        q_scale=profile["q_scale"],
        figures=False,
        refinement=profile["refinement"],
        case_profile=case_id,
        refinement_backend="serial",
        refinement_algorithm="gradient_laplace",
    )

    # Ranking uses fit information only; literature references never select a candidate.
    def rank(c):
        loss = float(c["q_loss"]) if c["q_loss"] is not None else float("inf")
        loss = loss if np.isfinite(loss) else float("inf")
        return (not c["is_converged"], loss, c["orientation_index"])

    candidates = sorted(result["candidates"], key=rank)
    import pandas as pd
    from IPython.display import display

    table = pd.DataFrame(
        [
            dict(
                rank=i + 1,
                hkl=str(tuple(c["hkl"])),
                accepted=c["is_converged"],
                q_loss=float(c["q_loss"]),
                **dict(zip(["a", "b", "c", "alpha", "beta", "gamma"], c["refined_cell_deg"])),
            )
            for i, c in enumerate(candidates)
        ]
    )
    table.to_csv(output / "ranked_candidates.csv", index=False)
    display(table.head(10))
    print("Accepted:", sum(c["is_converged"] for c in candidates), "of", len(candidates))
    best = candidates[0]
    generator = GixdDataGenerator("uniform", "testing", load_unit_cell_generator=False)
    geometry = Geometry.from_generator(generator, "cpu")
    qxy, qz, scale = _load_and_filter_peaks(
        root / ".assets/real_cases" / f"{case_id}_without_specular.csv",
        generator.feature_config,
        profile["q_scale"],
    )
    observed = np.column_stack((qxy.numpy(), qz.numpy())) * scale
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    for ax, key, label in zip(
        axes, ["initial_cell_deg", "refined_cell_deg"], ["NN prediction", "Gradient refinement"]
    ):
        predicted = (
            simulate(
                torch.tensor(best[key], dtype=torch.float64),
                torch.tensor(best["hkl"]),
                geometry.laue,
            )
            .cpu()
            .numpy()
        )
        keep = (
            (predicted[:, 0] > 0)
            & (predicted[:, 0] < 2.5)
            & (predicted[:, 1] >= 0)
            & (predicted[:, 1] < 2.5)
        )
        ax.scatter(
            predicted[keep, 0],
            predicted[keep, 1],
            s=20,
            facecolors="none",
            edgecolors="#c44e52",
            label="Simulated",
        )
        ax.scatter(
            observed[:, 0], observed[:, 1], s=16, color="#1f77b4", marker="x", label="Measured"
        )
        ax.set(
            xlabel=r"$q_{xy}$ (Å$^{-1}$)",
            title=label,
            xlim=(0, max(observed[:, 0]) * 1.1),
            ylim=(0, max(observed[:, 1]) * 1.1),
        )
        ax.set_aspect("equal")
    axes[0].set_ylabel(r"$q_z$ (Å$^{-1}$)")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"{case_id}: hkl {tuple(best['hkl'])}; accepted={best['is_converged']}")
    fig.tight_layout()
    _save(fig, output, "reconstruction")
    print("Conditional local uncertainties (a,b,c,alpha,beta,gamma):", best["laplace_uncertainty"])
    return result


def train_demo(root, full_training=False):
    """Train in a separate process so config overrides never leak into other notebooks."""
    root = Path(root)
    import json5

    base = json5.loads(
        (root / "configs/training/high_missingness_no_specular_30pct.json5").read_text()
    )
    output = root / "reviewer_outputs" / ("training_full" if full_training else "training_demo")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Training output already exists: {output}. Use the documented --resume latest command or a new directory."
        )
    output.mkdir(parents=True, exist_ok=True)
    base["run"]["id"] = "reviewer-full" if full_training else "reviewer-demo"
    base["run"]["model_name"] = "reviewer_model"
    base["paths"]["root_path"] = str(output / "run")
    base["paths"]["unit_cell_generator_path"] = str(
        root / ".assets/paper/gmm/unit_cell_generators.pkl"
    )
    if not full_training:
        hp = base["ml_hyperparameters"]
        hp["num_epochs"] = 2
        hp["training"].update(batch_size=8, samples_per_epoch=128)
        hp["validation"].update(batch_size=8, samples_per_epoch=64)
        hp["simulation"]["batch_size"] = 32
    config = output / "config.json5"
    config.write_text(json.dumps(base, indent=2) + "\n")
    command = [sys.executable, "-m", "gidind_ai.training", "--config", str(config)]
    print(
        "CUDA when available, otherwise CPU. Legacy validation is developmental, not an independent final test."
    )
    subprocess.run(command, check=True, cwd=root)
    return plot_training_history(output)


def plot_training_history(output):
    """Replot epoch means from saved batch logs without running training again."""
    output = Path(output)
    import pandas as pd
    import matplotlib.pyplot as plt

    history = next((output / "run").rglob("metrics.csv"))
    data = pd.read_csv(history)
    epochs = json.loads((output / "config.json5").read_text())["ml_hyperparameters"]["num_epochs"]
    data = data[data["epoch"] < epochs]  # final SWA BN refresh has no optimizer updates
    fig, ax = plt.subplots(figsize=(6, 4))
    curves = {}
    for key, label in [("train_loss", "Training"), ("val_loss", "Validation")]:
        if key in data:
            values = data.dropna(subset=["epoch", key]).groupby("epoch")[key].mean()
            curves[key] = values
            ax.plot(values.index + 1, values, marker="o", label=label)
    pd.DataFrame(curves).rename_axis("epoch_zero_based").to_csv(output / "epoch_losses.csv")
    ax.set(xlabel="Epoch", ylabel="Normalized cell MSE")
    ax.set_xticks(sorted(data["epoch"].dropna().unique() + 1))
    ax.legend()
    fig.tight_layout()
    _save(fig, output, "training_history")
    return str(output)
