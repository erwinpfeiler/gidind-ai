#!/usr/bin/env python3
"""Build the checksummed Hugging Face Dataset bundle for paper benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

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


def copy_asset(source: Path, destination: Path, output: Path, records: dict[str, Any]) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Required paper asset does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    records[destination.relative_to(output).as_posix()] = {
        "sha256": sha256(destination),
        "size_bytes": destination.stat().st_size,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--release-id", default="paper-v1-niggli-1e-3")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    output.mkdir(parents=True)
    files: dict[str, Any] = {}
    counts: dict[str, dict[str, int]] = {}
    for split in ("training_unit_cells", "test_unit_cells"):
        for lattice_system in LATTICE_SYSTEMS:
            original = source / split / f"{lattice_system}.npy"
            destination = output / "splits" / split / original.name
            copy_asset(original, destination, output, files)
            counts.setdefault(lattice_system, {})[split] = sum(
                1 for _ in original.open(encoding="utf-8")
            )
    sampler = source / "paper_1e-3_0050_00122_20" / "unit_cell_generators.pkl"
    copy_asset(sampler, output / "gmm" / sampler.name, output, files)
    for document in (
        "frozen_split_audit.json",
        "frozen_split_sanitization.json",
        "frozen_split_sanitization_1e-3.json",
        "frozen_split_clean_audit.json",
    ):
        copy_asset(
            source / "docs" / "reproduction" / document,
            output / "provenance" / document,
            output,
            files,
        )

    manifest = {
        "schema_version": 1,
        "release_id": args.release_id,
        "purpose": "immutable assets for the gidind-ai paper synthetic benchmark",
        "paper_equivalence": "Use this release, not a live COD download, for paper benchmarks.",
        "cod": {
            "provider": "Crystallography Open Database",
            "license": "CC0",
            "source_limitation": "Historic parsed tables do not retain per-entry COD identifiers or an upstream archive checksum.",
        },
        "split": {
            "legacy_split_seed": 42,
            "training_rows": 276731,
            "test_rows": 69154,
            "niggli_reduction": "niggli.cell_to_niggli",
            "deduplication": "Test rows matching a training Niggli key rounded independently to 0.001 A/degree were removed.",
            "global_niggli_overlap_after_sanitization": 0,
            "systems": counts,
        },
        "sampler": {
            "format": "trusted project pickle",
            "scikit_learn_version": "1.9.0",
            "gmm_refit": False,
            "warning": "The checksum must be verified before unpickling; use only this pinned asset release.",
        },
        "files": files,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "README.md").write_text(
        "---\n"
        "license: cc0-1.0\n"
        "pretty_name: gidind-ai paper benchmark assets\n"
        "---\n\n"
        "# gidind-ai paper assets\n\n"
        "This release contains the frozen synthetic benchmark split and pre-fitted sampler "
        "needed to reproduce the paper benchmark. It does not redistribute a full COD archive. "
        "Use the immutable revision recorded by gidind-ai; see `manifest.json` and `provenance/`.\n"
    )
    print(f"Built {args.release_id} at {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
