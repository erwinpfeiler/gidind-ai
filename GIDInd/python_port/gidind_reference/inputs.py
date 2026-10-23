"""Strict CSV/XLSX input; never use the AI unit/scale inference or q filtering."""
import csv
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import numpy as np


@dataclass
class Peaks:
    coordinates: np.ndarray
    source_rows: np.ndarray
    provenance: dict


def read_peaks(path, mode, settings):
    path = Path(path).resolve()
    if mode not in ("specular", "without-specular"):
        raise ValueError("Unknown indexing mode")
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as stream:
            records = list(csv.reader(stream))
    elif path.suffix.lower() == ".xlsx":
        from openpyxl import load_workbook
        wb = load_workbook(path, read_only=True, data_only=True)
        try:
            records = list(wb.worksheets[0].values)
        finally:
            wb.close()
    else:
        raise ValueError("Input must be .csv or .xlsx (legacy .xls is not supported)")
    if not records:
        raise ValueError("Empty input")
    labels = [str(x).strip().lower() for x in records[0]]
    if "q_xy" in labels and "q_z" in labels:
        if labels.count("q_xy") != 1 or labels.count("q_z") != 1:
            raise ValueError("Duplicate coordinate headers")
        columns = [labels.index("q_xy"), labels.index("q_z")]
        start = 1
    else:
        columns, start = [0, 1], 0
        if len(records[0]) != 2:
            raise ValueError("Headerless inputs require exactly two columns: q_xy,q_z")
    coords, rows, removed = [], [], []
    for row_number, record in enumerate(records[start:], start + 1):
        if start == 0 and len(record) != 2:
            raise ValueError(f"Row {row_number}: expected exactly two columns")
        try:
            raw = [record[i] for i in columns]
            if any(isinstance(x, bool) or x is None or str(x).strip() == "" for x in raw):
                raise ValueError()
            xy, z = map(float, raw)
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError(f"Row {row_number}: missing or nonnumeric q coordinates") from exc
        if not all(math.isfinite(x) and x >= 0 for x in (xy, z)):
            raise ValueError(f"Row {row_number}: coordinates must be finite and nonnegative")
        if xy == 0 and z == 0:
            removed.append({"source_row": row_number, "reason": "zero_placeholder"})
        elif mode == "without-specular" and xy <= settings.specular_filter_cutoff:
            removed.append({"source_row": row_number, "reason": "specular_filter"})
        else:
            coords.append([xy, z]); rows.append(row_number)
    if len(coords) < (4 if mode == "specular" else 6):
        raise ValueError("Not enough retained peaks (need 4 with specular, 6 without)")
    arr = np.asarray(coords, dtype=float)
    spec = np.flatnonzero((arr[:, 0] == 0) & (arr[:, 1] > 0))
    if mode == "specular" and len(spec) == 0:
        raise ValueError("Specular mode requires a positive-q_z peak with q_xy=0")
    provenance = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "units": "angstrom^-1", "scaling_applied": False,
                  "num_peaks": len(arr), "removed_rows": removed,
                  "selected_specular_source_row": int(rows[spec[0]]) if mode == "specular" else None,
                  "selected_specular_q": float(arr[spec[0], 1]) if mode == "specular" else None,
                  "specular_selection_policy": "first genuine specular row in original input order",
                  "prepared_sha256": hashlib.sha256(arr.astype('<f8').tobytes()).hexdigest()}
    return Peaks(arr, np.asarray(rows, dtype=int), provenance)


def seed_coordinates(peaks, mode, settings):
    coords = peaks.coordinates
    if settings.seed_peak_rows is not None:
        lookup = {int(row): i for i, row in enumerate(peaks.source_rows)}
        if any(row not in lookup for row in settings.seed_peak_rows):
            raise ValueError("Selected seed row was filtered or is absent")
        indices = np.array([lookup[row] for row in settings.seed_peak_rows])
    else:
        if mode == 'without-specular':
            indices = np.lexsort((coords[:, 1], coords[:, 0], np.linalg.norm(coords, axis=1)))
        else:
            indices = np.argsort(np.linalg.norm(coords, axis=1), kind="stable")
    if mode == "specular":
        # INITIALIZE_GUI: uniquetol(|q|, .0005), exclude either coordinate zero.
        valid = []
        last = -np.inf
        for i in indices:
            norm = float(np.linalg.norm(coords[i]))
            if norm - last <= .0005:
                continue
            last = norm
            if np.all(coords[i] != 0):
                valid.append(i)
        indices = np.asarray(valid[:settings.specular_seed_peaks], dtype=int)
        need = 3
    else:
        indices = indices[:settings.without_specular_seed_peaks]
        need = 6
    if len(indices) < need:
        raise ValueError(f"Need at least {need} eligible seed peaks")
    return coords[indices], peaks.source_rows[indices]
