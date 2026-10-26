"""One single-threaded CPU process per robustness condition, with durable shards."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import multiprocessing as mp
import os
import queue
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

from .figure3_assets import assets, atomic_json

_PROGRESS_QUEUE = None


def _init_worker(progress_queue, cpu_ids, next_cpu):
    global _PROGRESS_QUEUE
    _PROGRESS_QUEUE = progress_queue
    if cpu_ids:
        with next_cpu.get_lock():
            cpu = cpu_ids[next_cpu.value]
            next_cpu.value += 1
        os.sched_setaffinity(0, {cpu})
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _worker(payload):
    from .figure3_benchmark import run

    values, index, root = payload
    args = argparse.Namespace(**values)
    args.workers = 1
    args.condition_index = index
    args.cohort_dir = Path(root)
    args.run_dir = Path(root) / "conditions" / f"{index:03d}"
    log = Path(root) / "condition_logs" / f"condition_{index:03d}.out"
    with (
        log.open("a", buffering=1) as handle,
        contextlib.redirect_stdout(handle),
        contextlib.redirect_stderr(handle),
    ):
        try:
            run(args, progress_callback=_PROGRESS_QUEUE.put)
        except BaseException:
            traceback.print_exc()
            raise
    return index


def merge_condition(db, shard, arm, value, samples):
    """Read a stopped/completed shard; only its assigned condition is accepted."""
    from .figure3_benchmark import database

    child = database(shard)
    rows = child.execute("SELECT arm,value,sample,result FROM fits ORDER BY sample").fetchall()
    child.close()
    if any(
        row["arm"] != arm or row["value"] != value or not 0 <= row["sample"] < samples
        for row in rows
    ):
        raise ValueError(f"Unexpected evaluation in condition shard {shard}")
    with db:
        db.executemany("INSERT OR REPLACE INTO fits VALUES(?,?,?,?)", [tuple(row) for row in rows])
    return len(rows)


def run_parallel(args, progress):
    import torch

    from gixd_data_generator import GixdDataGenerator

    from .figure3_benchmark import cohort, database, plot, provenance

    # Enforce one compute thread per condition before spawning fresh interpreters.
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "condition_logs").mkdir(exist_ok=True)
    with (root / "run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        progress.set_stage("Verifying pinned assets for CPU condition workers")
        bundle = assets(args.allow_download)
        if args.checkpoint:
            from .figure3_assets import sha256

            if sha256(args.checkpoint) != bundle["model"]["sha256"]:
                raise ValueError("Checkpoint does not match pinned real_cases_model.ckpt")
        cfg = json.loads(json.dumps(provenance(args, bundle)))
        manifest = root / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()) != cfg:
            raise ValueError("Run inputs/settings/code changed; use a new run directory")
        atomic_json(manifest, cfg)
        progress.set_stage("Preparing/verifying one shared CPU cohort for all conditions")
        gen = GixdDataGenerator("uniform", "testing", load_unit_cell_generator=False)
        gen.device = torch.device("cpu")
        gen.laue_indices_tensor = gen.laue_indices_tensor.cpu()
        if gen.feature_config.use_specular_scan:
            raise ValueError("real_cases model requires the no-specular configuration")
        data = cohort(root, bundle, gen, cfg)
        if len(data["truth"]) != args.samples:
            raise ValueError("Shared cohort count mismatch")
        data.close()
        del gen
        db = database(root)
        pending = []
        total_saved = 0
        for index, (arm, value) in enumerate(cfg["settings"]):
            shard = root / "conditions" / f"{index:03d}"
            shard.mkdir(parents=True, exist_ok=True)
            expected = {**cfg, "settings": [[arm, value]]}
            child_manifest = shard / "manifest.json"
            if child_manifest.exists() and json.loads(child_manifest.read_text()) != expected:
                raise ValueError(f"Changed condition shard manifest: {shard}")
            atomic_json(child_manifest, expected)
            # Seed shards from any parent records, e.g. an interrupted serial run.
            child = database(shard)
            rows = db.execute("SELECT * FROM fits WHERE arm=? AND value=?", (arm, value)).fetchall()
            with child:
                child.executemany(
                    "INSERT OR IGNORE INTO fits VALUES(?,?,?,?)", [tuple(r) for r in rows]
                )
            child.close()
            saved = merge_condition(db, shard, arm, value, args.samples)
            total_saved += saved
            if saved < args.samples:
                pending.append((vars(args).copy(), index, str(root)))
        progress.configure(len(cfg["settings"]) * args.samples, total_saved)
        atomic_json(
            root / f"cpu_execution_{time.time_ns()}.json",
            {
                "workers_requested": args.workers,
                "workers_started": min(args.workers, len(pending)),
                "cpu_threads_per_worker": 1,
                "device": "cpu",
                "start_method": "spawn",
                "slurm_job_id": os.getenv("SLURM_JOB_ID"),
                "slurm_cpus_per_task": os.getenv("SLURM_CPUS_PER_TASK"),
                "conditions": len(cfg["settings"]),
                "samples_per_condition": args.samples,
            },
        )
        ctx = mp.get_context("spawn")
        messages = ctx.Queue()
        cpu_ids = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
        active_workers = min(args.workers, len(pending))
        if active_workers > (len(cpu_ids) if cpu_ids else os.cpu_count() or 1):
            raise RuntimeError("More condition workers requested than allocated CPUs")
        next_cpu = ctx.Value("i", 0)
        errors = []
        finished = len(cfg["settings"]) - len(pending)

        def drain():
            while True:
                try:
                    count = messages.get_nowait()
                except queue.Empty:
                    return
                for _ in range(count):
                    progress.advance()

        if pending:
            progress.set_stage(
                f"Starting {min(args.workers, len(pending))} CPU workers; one condition per worker; {finished}/{len(cfg['settings'])} conditions complete"
            )
            with ProcessPoolExecutor(
                max_workers=min(args.workers, len(pending)),
                mp_context=ctx,
                initializer=_init_worker,
                initargs=(messages, cpu_ids, next_cpu),
            ) as pool:
                futures = {pool.submit(_worker, task): task[1] for task in pending}
                while futures:
                    ready, _ = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
                    drain()
                    for future in ready:
                        index = futures.pop(future)
                        arm, value = cfg["settings"][index]
                        try:
                            future.result()
                        except Exception as exc:  # noqa: BLE001 - Finish other durable condition shards.
                            errors.append(f"condition {index} ({arm}={value:g}): {exc}")
                        count = merge_condition(
                            db, root / "conditions" / f"{index:03d}", arm, value, args.samples
                        )
                        if count == args.samples:
                            finished += 1
                        progress.set_stage(
                            f"CPU conditions complete={finished}/{len(cfg['settings'])}; last={arm}:{value:g} ({count}/{args.samples}); errors={len(errors)}"
                        )
                        plot(root, allow_partial=True)
            drain()
        messages.close()
        messages.join_thread()
        db.close()
        if errors:
            raise RuntimeError(
                "CPU workers failed; completed fits preserved. See condition_logs/: "
                + "; ".join(errors)
            )
        progress.set_stage("All CPU conditions complete; exporting final PDF/SVG figures")
        plot(root)
        progress.set_stage(f"COMPLETE; results and figures: {root}")
