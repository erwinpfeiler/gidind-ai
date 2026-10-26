"""Deterministic preparation of a pinned, user-supplied COD CIF snapshot.

This module deliberately does not download a mutable live COD database.  A caller
first obtains a snapshot, records its upstream identity, and then this module
turns that immutable local directory into auditable derived tables and splits.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .workflows import WorkflowError

LATTICE_SYSTEMS = (
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "rhombohedral",
    "hexagonal",
    "cubic",
)
CELL_COLUMNS = ("a", "b", "c", "alpha", "beta", "gamma")
_FLOAT_PREFIX = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
_COD_ID = re.compile(r"(?<!\d)(\d{7})(?!\d)")


@dataclass(frozen=True)
class PreparedCodAssets:
    """Locations and summary of one deterministic derived COD release."""

    output_dir: Path
    manifest_path: Path
    accepted_entries: int
    rejected_entries: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_fingerprint(cif_dir: Path, files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(cif_dir).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _parse_float(value: Any, field: str) -> float:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"{field} has {len(value)} values")
        value = value[0]
    match = _FLOAT_PREFIX.match(str(value))
    if not match:
        raise ValueError(f"{field} is not numeric: {value!r}")
    number = float(match.group(1))
    if not math.isfinite(number):
        raise ValueError(f"{field} is not finite")
    return number


def _value(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data and data[key] not in (None, "?", "."):
            return data[key]
    raise ValueError(f"Missing CIF field {keys[0]}")


def _lattice_system(space_group_number: int) -> str:
    if 1 <= space_group_number <= 2:
        return "triclinic"
    if 3 <= space_group_number <= 15:
        return "monoclinic"
    if 16 <= space_group_number <= 74:
        return "orthorhombic"
    if 75 <= space_group_number <= 142:
        return "tetragonal"
    if space_group_number in {146, 148, 155, 160, 161, 166, 167}:
        return "rhombohedral"
    if 143 <= space_group_number <= 194:
        return "hexagonal"
    if 195 <= space_group_number <= 230:
        return "cubic"
    raise ValueError(f"Unsupported space-group number {space_group_number}")


def _cod_id(path: Path) -> str:
    match = _COD_ID.search(path.stem)
    return match.group(1) if match else path.stem


def _read_cif(path: Path, root: Path) -> dict[str, Any]:
    try:
        from pymatgen.io.cif import CifFile
    except ImportError as exc:  # pragma: no cover - pymatgen is a runtime dependency
        raise WorkflowError("COD preparation requires pymatgen.") from exc

    cif = CifFile.from_file(path)
    if not cif.data:
        raise ValueError("CIF contains no data block")
    data = next(iter(cif.data.values())).data
    sg = int(
        _parse_float(
            _value(data, "_space_group_IT_number", "_symmetry_Int_Tables_number"), "space group"
        )
    )
    record: dict[str, Any] = {
        "cod_id": _cod_id(path),
        "source_relative_path": path.relative_to(root).as_posix(),
        "cif_sha256": _sha256(path),
        "space_group_number": sg,
        "lattice_system": _lattice_system(sg),
        "a": _parse_float(_value(data, "_cell_length_a"), "_cell_length_a"),
        "b": _parse_float(_value(data, "_cell_length_b"), "_cell_length_b"),
        "c": _parse_float(_value(data, "_cell_length_c"), "_cell_length_c"),
        "alpha": _parse_float(_value(data, "_cell_angle_alpha"), "_cell_angle_alpha"),
        "beta": _parse_float(_value(data, "_cell_angle_beta"), "_cell_angle_beta"),
        "gamma": _parse_float(_value(data, "_cell_angle_gamma"), "_cell_angle_gamma"),
    }
    return record


def _convert_rhombohedral(record: dict[str, Any]) -> dict[str, Any]:
    """Apply the legacy conventional-hexagonal to rhombohedral conversion."""
    if record["lattice_system"] != "rhombohedral":
        return record
    alpha, beta, gamma = (record[key] for key in ("alpha", "beta", "gamma"))
    if gamma > 119.0 and alpha < 91.0 and beta < 91.0:
        a_hex, c_hex = np.mean([record["a"], record["b"]]), record["c"]
    elif beta > 119.0 and gamma < 91.0 and alpha < 91.0:
        a_hex, c_hex = np.mean([record["c"], record["a"]]), record["b"]
    elif alpha > 119.0 and beta < 91.0 and gamma < 91.0:
        a_hex, c_hex = np.mean([record["b"], record["c"]]), record["a"]
    else:
        return record
    a_rhom = float(np.sqrt(3 * a_hex**2 + c_hex**2) / 3)
    angle = float(
        np.degrees(np.arccos(0.5 * (2 * c_hex**2 - 3 * a_hex**2) / (c_hex**2 + 3 * a_hex**2)))
    )
    converted = dict(record)
    converted.update(
        {"a": a_rhom, "b": a_rhom, "c": a_rhom, "alpha": angle, "beta": angle, "gamma": angle}
    )
    return converted


def _niggli_reduce(record: dict[str, Any]) -> dict[str, Any]:
    """Replace cell parameters with their canonical Niggli-reduced representation."""
    try:
        from niggli import cell_to_niggli
    except ImportError as exc:  # pragma: no cover - installed as a packaged legacy module
        raise WorkflowError("COD preparation could not import the bundled Niggli reducer.") from exc
    from read_crystal_data_2 import checked_niggli_cell
    reduced = checked_niggli_cell([record[column] for column in CELL_COLUMNS])
    if reduced.shape != (len(CELL_COLUMNS),) or not np.all(np.isfinite(reduced)):
        raise ValueError("Niggli reduction returned an invalid cell")
    if np.any(reduced[:3] >= 999.0):
        raise ValueError("Niggli reduction rejected the cell")
    result = dict(record)
    result.update({"conventional_" + key: record[key] for key in CELL_COLUMNS})
    result.update(dict(zip(CELL_COLUMNS, reduced.tolist(), strict=True)))
    return result


def _filter_and_deduplicate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    for record in records:
        if record["lattice_system"] == "hexagonal" and not 119.9 <= record.get("conventional_gamma", record["gamma"]) <= 120.1:
            continue
        if not all(4.0 <= record.get("conventional_"+key, record[key]) <= 30.0 for key in ("a", "b", "c")):
            continue
        if not all(60.0 <= record.get("conventional_"+key, record[key]) <= 120.0 for key in ("alpha", "beta", "gamma")):
            continue
        accepted.append(record)

    # Deduplicate globally, not within a lattice-system stratum: the same Niggli
    # cell can originate from different conventional settings, and otherwise one
    # copy could end up in training and another in testing.
    unique: list[dict[str, Any]] = []
    seen: set[tuple[float, ...]] = set()
    for record in sorted(
        accepted,
        key=lambda item: (
            item["cod_id"],
            item["source_relative_path"],
            item["cif_sha256"],
        ),
    ):
        key = tuple(np.round([record[column] for column in CELL_COLUMNS], 3))
        if key not in seen:
            seen.add(key)
            unique.append(record)
    return unique


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare_cod_snapshot(
    *,
    cif_dir: str | Path,
    output_dir: str | Path,
    source_url: str,
    source_revision: str,
    seed: int = 42,
    test_fraction: float = 0.2,
) -> PreparedCodAssets:
    """Create auditable train/test arrays from a pinned local COD CIF directory."""
    root = Path(cif_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not root.is_dir():
        raise WorkflowError(f"COD CIF directory does not exist: {root}")
    if output.exists() and any(output.iterdir()):
        raise WorkflowError(f"COD output directory must be empty: {output}")
    if not source_url.strip() or not source_revision.strip():
        raise WorkflowError("--source-url and --source-revision must both be non-empty.")
    if not 0.0 < test_fraction < 1.0:
        raise WorkflowError("--test-fraction must be strictly between zero and one.")

    files = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".cif"
    )
    if not files:
        raise WorkflowError(f"No CIF files found below {root}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "splits").mkdir(exist_ok=True)

    parsed: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for path in files:
        try:
            parsed.append(_niggli_reduce(_convert_rhombohedral(_read_cif(path, root))))
        except Exception as exc:  # noqa: BLE001 - malformed third-party CIFs are recorded verbatim
            rejected.append(
                {
                    "cod_id": _cod_id(path),
                    "source_relative_path": path.relative_to(root).as_posix(),
                    "cif_sha256": _sha256(path),
                    "reason": str(exc),
                }
            )
    accepted = _filter_and_deduplicate(parsed)
    if not accepted:
        raise WorkflowError("No usable COD entries remain after documented filters.")

    entry_columns = [
        "cod_id",
        "source_relative_path",
        "cif_sha256",
        "space_group_number",
        "lattice_system",
        *CELL_COLUMNS,
        *("conventional_" + key for key in CELL_COLUMNS),
    ]
    accepted.sort(
        key=lambda item: (item["lattice_system"], item["cod_id"], item["source_relative_path"])
    )
    _write_csv(output / "accepted_entries.csv", accepted, entry_columns)
    _write_csv(
        output / "rejected_entries.csv",
        rejected,
        ["cod_id", "source_relative_path", "cif_sha256", "reason"],
    )

    rng = np.random.default_rng(seed)
    split_summary: dict[str, dict[str, int]] = {}
    files_manifest: dict[str, dict[str, Any]] = {}
    for lattice_system in LATTICE_SYSTEMS:
        subset = [item for item in accepted if item["lattice_system"] == lattice_system]
        subset.sort(
            key=lambda item: (item["cod_id"], item["source_relative_path"], item["cif_sha256"])
        )
        n_test = math.ceil(len(subset) * test_fraction)
        permutation = rng.permutation(len(subset))
        test_indices = set(permutation[:n_test].tolist())
        train = [item for index, item in enumerate(subset) if index not in test_indices]
        test = [item for index, item in enumerate(subset) if index in test_indices]
        for split_name, rows in (("training", train), ("test", test)):
            table_path = output / "splits" / f"{split_name}_{lattice_system}.csv"
            array_path = output / "splits" / f"{split_name}_{lattice_system}.npy"
            _write_csv(table_path, rows, entry_columns)
            array = np.asarray(
                [[row[column] for column in CELL_COLUMNS] for row in rows], dtype=np.float64
            )
            if array.size == 0:
                array = array.reshape(0, len(CELL_COLUMNS))
            np.save(array_path, array)
            for asset_path in (table_path, array_path):
                relative = asset_path.relative_to(output).as_posix()
                files_manifest[relative] = {
                    "sha256": _sha256(asset_path),
                    "bytes": asset_path.stat().st_size,
                }
        split_summary[lattice_system] = {
            "accepted": len(subset),
            "training": len(train),
            "test": len(test),
        }

    for asset_path in (output / "accepted_entries.csv", output / "rejected_entries.csv"):
        relative = asset_path.relative_to(output).as_posix()
        files_manifest[relative] = {
            "sha256": _sha256(asset_path),
            "bytes": asset_path.stat().st_size,
        }
    manifest = {
        "schema_version": 1,
        "release_kind": "reproducible_cod_snapshot",
        "paper_equivalence": "not claimed; use the separately frozen paper assets for paper benchmarks",
        "source": {
            "url": source_url,
            "revision": source_revision,
            "cif_file_count": len(files),
            "cif_directory_sha256": _directory_fingerprint(root, files),
        },
        "selection": {
            "distance_range_angstrom": [4.0, 30.0],
            "angle_range_degree": [60.0, 120.0],
            "hexagonal_gamma_range_degree": [119.9, 120.1],
            "niggli_reduction": "performed before global duplicate removal and train/test splitting",
            "deduplication": "first stable entry globally per rounded-three-decimal Niggli-cell key",
            "rhombohedral_conversion": "legacy conventional-hexagonal to rhombohedral conversion",
        },
        "split": {
            "algorithm": "numpy.random.default_rng(seed).permutation over stable per-system order",
            "seed": seed,
            "test_fraction": test_fraction,
            "systems": split_summary,
        },
        "parsing": {
            "accepted_entries": len(accepted),
            "rejected_entries": len(rejected),
            "rejected_entries_file": "rejected_entries.csv",
        },
        "files": files_manifest,
    }
    manifest_path = output / "manifest.json"
    _write_manifest(manifest_path, manifest)
    return PreparedCodAssets(output, manifest_path, len(accepted), len(rejected))
