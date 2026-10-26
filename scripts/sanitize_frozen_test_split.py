#!/usr/bin/env python3
"""Remove Niggli-equivalent training cells from the legacy test split.

This is a one-time migration for the paper assets.  It never changes training
cells or refits the GMM; it removes matching rows only from the test partition
and serializes an updated sampler containing the same fitted GMM parameters.
The original sampler is retained under a clearly legacy-only filename.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from audit_frozen_split import LATTICE_SYSTEMS, load_cells, niggli_reduce


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def keys_for(cells: np.ndarray, precision: int) -> list[tuple[float, ...] | None]:
    keys: list[tuple[float, ...] | None] = []
    for cell in cells:
        if np.all(np.isfinite(cell)):
            keys.append(tuple(np.round(cell, precision)))
        else:
            keys.append(None)
    return keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, default=Path("training_unit_cells"))
    parser.add_argument("--test-dir", type=Path, default=Path("test_unit_cells"))
    parser.add_argument(
        "--generator",
        type=Path,
        default=Path("paper_1e-3_0050_00122_20/unit_cell_generators.pkl"),
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=1_000)
    parser.add_argument("--precision", type=int, default=4)
    parser.add_argument(
        "--backup-generator",
        type=Path,
        help="Private archive location for the sampler state before this migration.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the migration. Without this flag, only calculate and report the proposed removal.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.chunk_size < 1 or args.precision < 0:
        raise ValueError(
            "--workers and --chunk-size must be positive; --precision must be non-negative."
        )
    if not args.generator.is_file():
        raise FileNotFoundError(f"Legacy sampler does not exist: {args.generator}")

    train_keys: set[tuple[float, ...]] = set()
    test_data: dict[str, np.ndarray] = {}
    removed_indices: dict[str, list[int]] = {}
    report: dict[str, Any] = {
        "schema_version": 1,
        "operation": "remove Niggli-equivalent training cells from frozen legacy test split",
        "applied": args.apply,
        "niggli_key_precision_decimal_places": args.precision,
        "training_cells_changed": False,
        "gmm_refit": False,
        "systems": {},
    }

    for lattice_system in LATTICE_SYSTEMS:
        path = args.training_dir / f"{lattice_system}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Missing training split file: {path}")
        reduced, failures = niggli_reduce(load_cells(path), args.workers, args.chunk_size)
        if failures:
            raise ValueError(
                f"Niggli reduction failed for {failures} training cells in {lattice_system}."
            )
        train_keys.update(key for key in keys_for(reduced, args.precision) if key is not None)

    for lattice_system in LATTICE_SYSTEMS:
        path = args.test_dir / f"{lattice_system}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Missing test split file: {path}")
        cells = load_cells(path)
        reduced, failures = niggli_reduce(cells, args.workers, args.chunk_size)
        if failures:
            raise ValueError(
                f"Niggli reduction failed for {failures} test cells in {lattice_system}."
            )
        indices = [
            index
            for index, key in enumerate(keys_for(reduced, args.precision))
            if key is not None and key in train_keys
        ]
        test_data[lattice_system] = cells
        removed_indices[lattice_system] = indices
        report["systems"][lattice_system] = {
            "test_rows_before": len(cells),
            "removed_rows": len(indices),
            "test_rows_after": len(cells) - len(indices),
            "removed_row_indices_zero_based": indices,
        }

    report["total_removed_test_rows"] = sum(len(indices) for indices in removed_indices.values())
    report["generator_before_sha256"] = sha256(args.generator)
    backup = args.backup_generator or args.generator.with_name(
        f"{args.generator.stem}_pre_niggli_{args.precision}dp_cleanup{args.generator.suffix}"
    )
    report["legacy_generator_backup"] = str(backup)

    if args.apply:
        if backup.exists():
            raise FileExistsError(
                f"Refusing to overwrite sampler archive {backup}; choose --backup-generator explicitly."
            )
        with tempfile.TemporaryDirectory(prefix="gidind-test-split-") as temporary:
            temporary_root = Path(temporary)
            replacement_paths: list[tuple[Path, Path]] = []
            for lattice_system, cells in test_data.items():
                keep = np.ones(len(cells), dtype=bool)
                keep[removed_indices[lattice_system]] = False
                replacement = temporary_root / f"{lattice_system}.npy"
                np.savetxt(replacement, cells[keep])
                replacement_paths.append((replacement, args.test_dir / replacement.name))

            with args.generator.open("rb") as handle:
                sampler = pickle.load(handle)
            for lattice_system, cells in test_data.items():
                keep = np.ones(len(cells), dtype=bool)
                keep[removed_indices[lattice_system]] = False
                sampler.test_crystals[lattice_system] = cells[keep].astype(np.float32, copy=False)
            replacement_generator = temporary_root / args.generator.name
            with replacement_generator.open("wb") as handle:
                pickle.dump(sampler, handle, protocol=pickle.HIGHEST_PROTOCOL)

            shutil.copy2(args.generator, backup)
            for source, destination in replacement_paths:
                shutil.copy2(source, destination)
            shutil.copy2(replacement_generator, args.generator)
        report["generator_after_sha256"] = sha256(args.generator)
        report["test_file_sha256"] = {
            lattice_system: sha256(args.test_dir / f"{lattice_system}.npy")
            for lattice_system in LATTICE_SYSTEMS
        }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps({"applied": args.apply, "removed": report["total_removed_test_rows"]}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
