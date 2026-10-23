#!/usr/bin/env python3
"""Audit a frozen unit-cell split without modifying it.

The legacy cell files have a ``.npy`` suffix but are whitespace-delimited text
tables written with :func:`numpy.savetxt`.  This tool reports both raw-row and
Niggli-reduced overlap, so that equivalent conventional cells cannot be
mistaken for independent train and test examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

LATTICE_SYSTEMS = (
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "rhombohedral",
    "hexagonal",
    "cubic",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cells(path: Path) -> np.ndarray:
    cells = np.loadtxt(path, dtype=np.float64)
    return np.atleast_2d(cells)


def raw_keys(cells: np.ndarray) -> set[bytes]:
    return {row.tobytes() for row in np.ascontiguousarray(cells)}


def reduce_chunk(cells: np.ndarray) -> tuple[np.ndarray, int]:
    """Niggli-reduce one batch in an isolated worker process."""
    from niggli import cell_to_niggli

    reduced = np.empty_like(cells)
    failures = 0
    for index, cell in enumerate(cells):
        try:
            result = np.asarray(cell_to_niggli(cell), dtype=np.float64)
            if result.shape != (6,) or not np.all(np.isfinite(result)):
                raise ValueError("invalid Niggli reduction")
            reduced[index] = result
        except (ArithmeticError, RuntimeError, TypeError, ValueError):
            # The audit records malformed cells rather than changing the split.
            reduced[index] = np.nan
            failures += 1
    return reduced, failures


def niggli_reduce(cells: np.ndarray, workers: int, chunk_size: int) -> tuple[np.ndarray, int]:
    chunks = [cells[index : index + chunk_size] for index in range(0, len(cells), chunk_size)]
    if workers == 1:
        results = [reduce_chunk(chunk) for chunk in chunks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(reduce_chunk, chunks))
    return np.concatenate([result for result, _ in results]), sum(failures for _, failures in results)


def niggli_keys(cells: np.ndarray, precision: int) -> set[tuple[float, ...]]:
    valid = cells[np.all(np.isfinite(cells), axis=1)]
    return {tuple(row) for row in np.round(valid, precision)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, default=Path("training_unit_cells"))
    parser.add_argument("--test-dir", type=Path, default=Path("test_unit_cells"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=2_000)
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Decimal places used to identify numerically equivalent reduced cells.",
    )
    parser.add_argument(
        "--compare-precisions",
        type=int,
        nargs="*",
        default=(),
        help="Additional decimal precisions to report from the same Niggli reductions.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (
        args.workers < 1
        or args.chunk_size < 1
        or args.precision < 0
        or any(precision < 0 for precision in args.compare_precisions)
    ):
        raise ValueError("--workers and --chunk-size must be positive; --precision must be non-negative.")
    precisions = sorted({args.precision, *args.compare_precisions})

    summary: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "read-only audit of the frozen legacy train/test split",
        "reducer": "niggli.cell_to_niggli",
        "niggli_key_precision_decimal_places": args.precision,
        "systems": {},
    }
    global_train_raw: set[bytes] = set()
    global_test_raw: set[bytes] = set()
    global_train_niggli: set[tuple[float, ...]] = set()
    global_test_niggli: set[tuple[float, ...]] = set()
    precision_counts = {
        precision: {"training": set(), "test": set()}
        for precision in precisions
    }

    for lattice_system in LATTICE_SYSTEMS:
        train_path = args.training_dir / f"{lattice_system}.npy"
        test_path = args.test_dir / f"{lattice_system}.npy"
        if not train_path.is_file() or not test_path.is_file():
            raise FileNotFoundError(f"Missing frozen split file for {lattice_system}.")
        train = load_cells(train_path)
        test = load_cells(test_path)
        train_reduced, train_failures = niggli_reduce(train, args.workers, args.chunk_size)
        test_reduced, test_failures = niggli_reduce(test, args.workers, args.chunk_size)
        train_raw = raw_keys(train)
        test_raw = raw_keys(test)
        train_niggli = niggli_keys(train_reduced, args.precision)
        test_niggli = niggli_keys(test_reduced, args.precision)
        summary["systems"][lattice_system] = {
            "training": {
                "path": str(train_path),
                "sha256": sha256(train_path),
                "rows": len(train),
                "unique_raw_rows": len(train_raw),
                "unique_niggli_keys": len(train_niggli),
                "niggli_failures": train_failures,
            },
            "test": {
                "path": str(test_path),
                "sha256": sha256(test_path),
                "rows": len(test),
                "unique_raw_rows": len(test_raw),
                "unique_niggli_keys": len(test_niggli),
                "niggli_failures": test_failures,
            },
            "raw_cross_split_overlap": len(train_raw & test_raw),
            "niggli_cross_split_overlap_within_system": len(train_niggli & test_niggli),
        }
        global_train_raw.update(train_raw)
        global_test_raw.update(test_raw)
        global_train_niggli.update(train_niggli)
        global_test_niggli.update(test_niggli)
        for precision, key_sets in precision_counts.items():
            key_sets["training"].update(niggli_keys(train_reduced, precision))
            key_sets["test"].update(niggli_keys(test_reduced, precision))

    summary["global"] = {
        "training_unique_raw_rows": len(global_train_raw),
        "test_unique_raw_rows": len(global_test_raw),
        "raw_cross_split_overlap": len(global_train_raw & global_test_raw),
        "training_unique_niggli_keys": len(global_train_niggli),
        "test_unique_niggli_keys": len(global_test_niggli),
        "niggli_cross_split_overlap": len(global_train_niggli & global_test_niggli),
    }
    summary["precision_comparison"] = {
        str(precision): {
            "training_unique_niggli_keys": len(key_sets["training"]),
            "test_unique_niggli_keys": len(key_sets["test"]),
            "niggli_cross_split_overlap": len(key_sets["training"] & key_sets["test"]),
        }
        for precision, key_sets in precision_counts.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary["global"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
