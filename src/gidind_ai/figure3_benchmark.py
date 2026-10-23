"""Reproduce Fig. 3b/c with a paired, known-orientation synthetic cohort.

Run via scripts/benchmark_figure3.py. Plotting is independent of Torch/CUDA.
All failed/empty fits remain in the denominator. No ground-truth fit selection.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import importlib.metadata
import json
import os
import platform
import sqlite3
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from .figure3_assets import assets, atomic_json, sampler, sha256


class Progress:
    """Flush a heartbeat even while one refinement or preparation step is busy."""

    def __init__(self, interval=60, on_advance=None):
        self.interval = interval
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.started = time.monotonic()
        self.work_started = self.started
        self.done = self.saved = self.total = 0
        self.stage = "Starting benchmark"
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.on_advance = on_advance

    def __enter__(self):
        self.report()
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop.set()
        self.thread.join()
        if exc is not None:
            self.set_stage(f"STOPPED: {exc}")

    def _heartbeat(self):
        while not self.stop.wait(self.interval):
            self.report()

    def configure(self, total, saved):
        with self.lock:
            self.total, self.done, self.saved = total, saved, saved
            self.work_started = time.monotonic()

    def set_stage(self, stage, report=True):
        with self.lock:
            self.stage = stage
        if report:
            self.report()

    def advance(self):
        with self.lock:
            self.done += 1
        if self.on_advance is not None:
            self.on_advance(1)

    def report(self):
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.started
            new = self.done - self.saved
            rate = new / max(now - self.work_started, 1e-9)
            eta = f"{(self.total - self.done) / rate / 3600:.2f}h" if rate else "warming up"
            counts = (
                f"overall={self.done}/{self.total} ({100 * self.done / self.total:.1f}%) "
                f"resumed={self.saved} rate={rate * 60:.2f} eval/min ETA≈{eta} | "
                if self.total
                else ""
            )
            stamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            print(f"[{stamp}] elapsed={elapsed / 60:.1f}min | {counts}{self.stage}", flush=True)


def settings(max_peaks=20, noise_steps=21, max_noise=0.02, peak_step=1):
    return [("peaks", float(i)) for i in range(-max_peaks, max_peaks + 1, peak_step)] + [
        ("noise", float(x)) for x in np.linspace(0, max_noise, noise_steps)[1:]
    ]


def condition_grid(args):
    grid = (
        [("peaks", 0.0)]
        if args.baseline_only
        else settings(args.max_peaks, args.noise_steps, args.max_noise, args.peak_step)
    )
    return [grid[args.condition_index]] if args.condition_index is not None else grid


def perturb(points, arm, value, seed, sample_id, domain, noise_mode="gaussian"):
    """Nested, paired perturbations; bounds discard peaks, never clip them.

    Gaussian noise has per-coordinate sigma=value (unbounded). Bounded noise
    is uniform in a disk of radius=value, hence ||delta q|| <= value.
    """
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, sample_id, {"peaks": 11, "noise": 23}[arm]])
    )
    p = np.array(points, dtype=np.float64, copy=True)
    removed = added = 0
    if arm == "peaks" and value < 0:
        order = rng.permutation(len(p))
        removed = min(-int(value), len(p))
        p = p[order[removed:]]
    elif arm == "peaks" and value > 0:
        # Draw all 20 at once so a smaller addition is a prefix of a larger one.
        extra = rng.uniform([domain[0], 0], [domain[1], domain[2]], (int(value), 2))
        p = np.concatenate([p, extra])
        added = int(value)
    elif arm == "noise":
        if noise_mode == "gaussian":
            p += value * rng.normal(size=p.shape)
        else:
            angle = rng.uniform(0, 2 * np.pi, len(p))
            radius = value * np.sqrt(rng.uniform(size=len(p)))
            p += np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
    mask = (p[:, 0] > domain[0]) & (p[:, 0] < domain[1]) & (p[:, 1] >= 0) & (p[:, 1] < domain[2])
    return p[mask], removed, added, int((~mask).sum())


def correct(truth, predicted):
    return bool(
        np.isfinite(predicted).all() and np.all(np.abs(np.asarray(predicted) - truth) < 0.1)
    )


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def database(root):
    db = sqlite3.connect(root / "results.sqlite")
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute(
        "CREATE TABLE IF NOT EXISTS fits(arm TEXT,value REAL,sample INTEGER,result TEXT,PRIMARY KEY(arm,value,sample))"
    )
    return db


def cohort(root, bundle, generator, cfg):
    """Generate once from the immutable, sanitized paper test split."""
    import torch

    from gixd_data_generator import GixdDataGenerator

    path = root / "cohort.npz"
    if path.exists():
        if sha256(path) != json.loads((root / "cohort.sha256.json").read_text())["sha256"]:
            raise ValueError("Saved cohort changed")
        return np.load(path, allow_pickle=False)
    generator.unit_cell_generator = sampler(bundle, cfg["seed"])
    generator.rng = np.random.default_rng(cfg["seed"])
    generator.feature_config.testing_gaussian_noise_sigma = 0.0
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    # Draw exactly N samples once; this also fixes the system mixture and orientation.
    target, _, systems, qxy, qz, top = generator.generate_data_batch(cfg["samples"])
    truth = np.asarray(target).copy()
    truth[:, 3:] = np.rad2deg(truth[:, 3:])
    points = []
    for x, z in zip(qxy.cpu(), qz.cpu()):
        keep = (
            (x > generator.feature_config.specular_cutoff)
            & (z >= 0)
            & (x < generator.q_xy_max)
            & (z < generator.q_z_max)
        )
        x, z = GixdDataGenerator.filter_q_points(x[keep], z[keep], tolerance=0.01)
        points.append(torch.stack([x, z], -1).numpy())
    counts = np.array([len(p) for p in points])
    padded = np.zeros((len(points), max(counts, default=0), 2))
    for i, p in enumerate(points):
        padded[i, : len(p)] = p
    tmp = root / "cohort.tmp.npz"
    np.savez_compressed(
        tmp,
        truth=truth,
        systems=systems.cpu().numpy(),
        top=np.asarray(top),
        points=padded,
        counts=counts,
    )
    tmp.replace(path)
    atomic_json(root / "cohort.sha256.json", {"sha256": sha256(path)})
    return np.load(path, allow_pickle=False)


def provenance(args, bundle):
    from config_loader import get_config_path

    source = Path(__file__).resolve().parents[1]
    names = [
        "gidind_ai/figure3_benchmark.py",
        "gidind_ai/figure3_assets.py",
        "gidind_ai/figure3_cpu.py",
        "gixd_data_generator.py",
        "gixd_regressor.py",
        "polishing_fit.py",
        "neural_network.py",
        "niggli.py",
        "unit_cell_generator.py",
        "config_loader.py",
        "gidind_ai/workflows.py",
        "gidind_ai/batched_refinement.py",
    ]
    return {
        "schema": 1,
        "samples": args.samples,
        "seed": args.seed,
        "settings": condition_grid(args),
        "noise_mode": args.noise_mode,
        "deduplication_A_inverse": 0.01,
        "model": bundle["model"],
        "split_revision": bundle["source_revision"],
        "split_hashes": bundle["hashes"],
        "configuration_sha256": sha256(get_config_path()),
        "source_hashes": {name: sha256(source / name) for name in names},
        "success": "all 3 lengths <0.1 Angstrom AND all 3 angles <0.1 degree; failures count as incorrect",
        "orientation": "true contact plane passed to NN and refinement",
        "cohort": "same N cells and orientations at every setting; uniform mixture of 7 lattice systems",
        "missing": "uniform removal without replacement; remove min(requested,available), record actual count; empty patterns fail",
        "extra": "uniform coordinates in accessible domain; pixel collisions allowed and recorded in images",
        "refinement": {
            "q_cutoff": args.q_cutoff,
            "maxiter": args.maxiter,
            "outer_loops": args.outer_loops,
            "matching_method": "hybrid",
            "method": "cg",
            "refinement_backend": "serial",
        },
        "device": args.device,
        "batch_size": args.batch_size,
    }


def run(args, progress_callback=None):
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    with Progress(args.progress_seconds, on_advance=progress_callback) as progress:
        if args.workers > 1:
            from .figure3_cpu import run_parallel

            run_parallel(args, progress)
        else:
            _run(args, progress)


def _run(args, progress):
    import torch

    from polishing_fit import direct_inversion_pytorch

    from .workflows import _build_regressor

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable. Submit on a GPU node; CPU smoke tests require --device cpu."
        )
    if args.require_a100 and (args.device != "cuda" or "A100" not in torch.cuda.get_device_name()):
        raise RuntimeError("This production job requires an NVIDIA A100")
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    progress.set_stage("Verifying checkpoint and frozen test assets")
    bundle = assets(args.allow_download)
    if args.checkpoint:
        if sha256(args.checkpoint) != bundle["model"]["sha256"]:
            raise ValueError("--checkpoint must match the pinned real_cases_model.ckpt SHA-256")
        bundle["checkpoint"] = str(args.checkpoint.resolve())
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg = provenance(args, bundle)
        # JSON roundtrip makes tuples compare consistently on resume.
        cfg = json.loads(json.dumps(cfg))
        manifest = root / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()) != cfg:
            raise ValueError("Run inputs/settings/code changed; use a new run directory")
        atomic_json(manifest, cfg)
        atomic_json(
            root / f"environment_{time.time_ns()}.json",
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
                "cuda": torch.version.cuda,
                "slurm_job_id": os.getenv("SLURM_JOB_ID"),
                "versions": {
                    p: importlib.metadata.version(p)
                    for p in (
                        "torch",
                        "torchvision",
                        "lightning",
                        "numpy",
                        "scipy",
                        "matplotlib",
                        "pymatgen",
                    )
                },
                "git_head": subprocess.run(
                    ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
                ).stdout.strip(),
            },
        )
        progress.set_stage(f"Loading model on {args.device}")
        reg, gen = _build_regressor(Path(bundle["checkpoint"]), torch.device(args.device))
        if gen.feature_config.use_specular_scan:
            raise ValueError("real_cases model requires the no-specular configuration")
        progress.set_stage("Preparing/verifying the fixed synthetic cohort")
        data = cohort(args.cohort_dir or root, bundle, gen, cfg)
        if len(data["truth"]) != args.samples:
            raise ValueError("Cohort count mismatch")
        domain = (gen.feature_config.specular_cutoff, gen.q_xy_max, gen.q_z_max)
        db = database(root)
        progress.configure(
            len(cfg["settings"]) * args.samples,
            db.execute("SELECT count(*) FROM fits").fetchone()[0],
        )
        for setting_index, (arm, value) in enumerate(cfg["settings"], 1):
            completed = {
                r[0]
                for r in db.execute("SELECT sample FROM fits WHERE arm=? AND value=?", (arm, value))
            }
            ids = [i for i in range(args.samples) if i not in completed]
            label = f"setting={setting_index}/{len(cfg['settings'])} {arm}={value:g}"
            progress.set_stage(f"{label} completed={len(completed)}/{args.samples}")
            for begin in range(0, len(ids), args.batch_size):
                progress.set_stage(
                    f"{label} NN batch starting at pending sample {begin + 1}", report=False
                )
                batch_ids = ids[begin : begin + args.batch_size]
                observations = [
                    perturb(
                        data["points"][i, : data["counts"][i]],
                        arm,
                        value,
                        args.seed,
                        i,
                        domain,
                        args.noise_mode,
                    )
                    for i in batch_ids
                ]
                # Rasterize separately so padding never creates a direct-beam pixel.
                images = []
                for p, *_ in observations:
                    t = torch.as_tensor(p, dtype=torch.float64, device=args.device)
                    images.append(gen.calc_images_from_q_points(t[:, 0], t[:, 1]))
                with torch.inference_mode():
                    predictions = reg._predict_cells_from_images(
                        torch.cat(images), data["top"][batch_ids]
                    )
                for i, initial, (p, removed, added, dropped) in zip(
                    batch_ids, predictions, observations
                ):
                    progress.set_stage(
                        f"{label} refining sample={i + 1}/{args.samples}", report=False
                    )
                    tick = time.perf_counter()
                    refined = np.full(6, np.nan)
                    error = None
                    if len(p) < 3:
                        error = "fewer than 3 observed reflections"
                    else:
                        obs = torch.as_tensor(p, dtype=torch.float64, device=args.device)
                        try:
                            result = direct_inversion_pytorch(
                                torch.tensor(initial, dtype=torch.float64, device=args.device),
                                obs[:, 0],
                                obs[:, 1],
                                gen.feature_config.miller_indices[int(data["top"][i])],
                                gen,
                                return_uncertainties=False,
                                **cfg["refinement"],
                            )
                            refined = result["refined_predicted_cell_deg"].detach().cpu().numpy()
                            if not np.isfinite(refined).all():
                                error = "nonfinite refinement"
                        except (ArithmeticError, ValueError, RuntimeError) as exc:
                            error = repr(exc)
                    truth = data["truth"][i]
                    record = {
                        "truth": truth.tolist(),
                        "initial": initial.tolist(),
                        "refined": refined.tolist() if np.isfinite(refined).all() else None,
                        "orientation_index": int(data["top"][i]),
                        "system": int(data["systems"][i]),
                        "observed": p.tolist(),
                        "n_clean": int(data["counts"][i]),
                        "n_observed": len(p),
                        "removed": removed,
                        "added": added,
                        "out_of_bounds": dropped,
                        "nn_correct": correct(truth, initial),
                        "correct": correct(truth, refined),
                        "error": error,
                        "seconds": time.perf_counter() - tick,
                    }
                    # A completed fit is durable before starting the next one.
                    with db:
                        db.execute(
                            "INSERT INTO fits VALUES(?,?,?,?)",
                            (arm, value, i, json.dumps(record, allow_nan=False)),
                        )
                    progress.advance()
                progress.set_stage(
                    f"{label} completed={len(completed) + min(begin + args.batch_size, len(ids))}/{args.samples}"
                )
            if args.condition_index is None:
                progress.set_stage(f"{label} exporting progress figures")
                plot(root, allow_partial=True)
        db.close()
        data.close()
    if args.condition_index is None:
        progress.set_stage("Exporting final PDF/SVG figures")
        plot(root)
    progress.set_stage(f"COMPLETE; results and figures: {root}")


def plot(root, allow_partial=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    root = Path(root)
    cfg = json.loads((root / "manifest.json").read_text())
    db = database(root)
    rows = []
    for arm, value in cfg["settings"]:
        records = [
            json.loads(r[0])
            for r in db.execute(
                "SELECT result FROM fits WHERE arm=? AND value=? ORDER BY sample", (arm, value)
            )
        ]
        if len(records) != cfg["samples"]:
            if allow_partial:
                continue
            raise ValueError(
                f"Incomplete setting {arm}={value}: {len(records)}/{cfg['samples']}; no final figure written"
            )
        k = sum(r["correct"] for r in records)
        lo, hi = wilson(k, len(records))
        rows.append(
            {
                "arm": arm,
                "value": value,
                "n": len(records),
                "correct": k,
                "fraction": k / len(records),
                "ci_low": lo,
                "ci_high": hi,
                "nn_fraction": sum(r["nn_correct"] for r in records) / len(records),
                "failures": sum(r["error"] is not None for r in records),
                "removal_shortfalls": sum(r["removed"] < -value for r in records)
                if arm == "peaks" and value < 0
                else 0,
            }
        )
    db.close()
    if not rows:
        return
    complete = len(rows) == len(cfg["settings"])
    suffix = "" if complete else "_partial"
    with (root / f"summary{suffix}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    baseline = next((r for r in rows if r["arm"] == "peaks" and r["value"] == 0), None)
    peak_rows = sorted((r for r in rows if r["arm"] == "peaks"), key=lambda r: r["value"])
    noise_rows = sorted(
        ([{**baseline, "arm": "noise"}] if baseline else [])
        + [r for r in rows if r["arm"] == "noise"],
        key=lambda r: r["value"],
    )
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none", "pdf.fonttype": 42}
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)

    def draw(ax, selected, panel):
        if not selected:
            return
        x = [r["value"] for r in selected]
        y = [r["fraction"] for r in selected]
        ax.plot(x, y, "o-", color="#24788e", ms=3, lw=1.4)
        ax.fill_between(
            x,
            [r["ci_low"] for r in selected],
            [r["ci_high"] for r in selected],
            color="#24788e",
            alpha=0.18,
            lw=0,
        )
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.2)
        ax.set_title(panel, loc="left", fontweight="bold")
        if not complete:
            ax.text(0.5, 0.5, "INCOMPLETE RUN", transform=ax.transAxes, ha="center", alpha=0.4)

    draw(axes[0], peak_rows, "b")
    draw(axes[1], noise_rows, "c")
    axes[0].axvline(0, color=".5", ls=":", lw=0.8)
    axes[0].set_xlabel("Peak count change (− missing, + added)")
    label = (
        r"Per-coordinate Gaussian $\sigma$ (Å$^{-1}$)"
        if cfg["noise_mode"] == "gaussian"
        else r"Maximum displacement $|\Delta q|$ (Å$^{-1}$)"
    )
    axes[1].set_xlabel(label)
    axes[0].xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    axes[1].xaxis.set_major_locator(MaxNLocator(nbins=4))
    axes[0].set_ylabel("Fraction of correctly reconstructed cells")
    if cfg["samples"] < 1000:
        fig.suptitle(f"PRELIMINARY — {cfg['samples']} evaluations per setting", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "svg", "png"):
        fig.savefig(root / f"figure3bc{suffix}.{ext}", dpi=180)
    # Also export each panel independently for manuscript assembly.
    for ax, name in zip(axes, ("figure3b", "figure3c")):
        fig.canvas.draw()
        bbox = (
            ax.get_tightbbox(fig.canvas.get_renderer())
            .transformed(fig.dpi_scale_trans.inverted())
            .expanded(1.03, 1.05)
        )
        for ext in ("pdf", "svg"):
            fig.savefig(root / f"{name}{suffix}.{ext}", bbox_inches=bbox)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("cache", help="Download/verify immutable assets before the offline Slurm job")
    p = sub.add_parser("run")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--baseline-only",
        action="store_true",
        help="Evaluate only the noiseless baseline for a preliminary check",
    )
    p.add_argument("--max-peaks", type=int, default=20)
    p.add_argument(
        "--peak-step",
        type=int,
        default=1,
        help="Spacing of peak-count conditions; must divide max-peaks",
    )
    p.add_argument("--noise-steps", type=int, default=21)
    p.add_argument("--max-noise", type=float, default=0.02)
    p.add_argument("--noise-mode", choices=("gaussian", "bounded"), default="gaussian")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--require-a100", action="store_true")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--allow-download", action="store_true")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--cpu-threads", type=int, default=1)
    p.add_argument("--workers", type=int, default=1, help="Independent CPU condition workers")
    p.add_argument("--condition-index", type=int, help="Run one zero-based condition index")
    p.add_argument("--cohort-dir", type=Path, help="Read a verified precomputed cohort")
    p.add_argument(
        "--print-condition-count", action="store_true", help="Print resource count and exit"
    )
    p.add_argument(
        "--progress-seconds",
        type=float,
        default=60,
        help="Flush a stdout heartbeat at this interval, including during long fits",
    )
    p.add_argument("--maxiter", type=int, default=500)
    p.add_argument("--outer-loops", type=int, default=20)
    p.add_argument("--q-cutoff", type=float, default=0.056)
    p = sub.add_parser("plot", help="Export final figures from a completed run; no GPU needed")
    p.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "cache":
        print(json.dumps(assets(True), indent=2))
    elif args.command == "plot":
        plot(args.run_dir)
    else:
        if (
            min(args.samples, args.batch_size, args.cpu_threads, args.maxiter, args.outer_loops) < 1
            or args.max_peaks < 0
            or args.peak_step < 1
            or args.max_peaks % args.peak_step != 0
            or args.noise_steps < 2
            or args.max_noise <= 0
            or args.q_cutoff <= 0
            or not np.isfinite(args.progress_seconds)
            or args.progress_seconds <= 0
            or args.workers < 1
        ):
            parser.error(
                "Counts/iterations/cutoff/noise must be positive; noise-steps >=2 and max-peaks >=0"
            )
        n = (
            1
            if args.baseline_only
            else len(settings(args.max_peaks, args.noise_steps, args.max_noise, args.peak_step))
        )
        if args.condition_index is not None and not 0 <= args.condition_index < n:
            parser.error("condition-index is outside the configured condition grid")
        if args.workers > 1 and (
            args.device != "cpu" or args.cpu_threads != 1 or args.condition_index is not None
        ):
            parser.error(
                "Parallel conditions require --device cpu --cpu-threads 1 and no condition-index"
            )
        if args.print_condition_count:
            print(len(condition_grid(args)))
        else:
            run(args)


if __name__ == "__main__":
    main()
