import json
from pathlib import Path

import numpy as np

from gidind_ai.cli import main
from gidind_ai.cod_prepare import prepare_cod_snapshot


def _cif(cell: float, identifier: int, space_group: int = 221) -> str:
    return f"""data_cod_{identifier}
_cell_length_a {cell}
_cell_length_b {cell}
_cell_length_c {cell}
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number {space_group}
"""


def _write_snapshot(root: Path) -> None:
    root.mkdir()
    for index, cell in enumerate((5.0, 5.1, 5.2, 5.3, 5.4), start=1_000_001):
        (root / f"{index}.cif").write_text(_cif(cell, index))
    # A second conventional representation of the first cubic cell. Niggli
    # reduction makes it identical, so it must be removed before splitting.
    (root / "1000006.cif").write_text(_cif(5.0, 1_000_006))
    (root / "1000007.cif").write_text(_cif(5.0, 1_000_007, space_group=123))
    (root / "broken_1000010.cif").write_text("data_broken\n_cell_length_a nope\n")


def test_cod_prepare_records_provenance_and_stable_split(tmp_path, capsys):
    snapshot = tmp_path / "snapshot"
    _write_snapshot(snapshot)
    first = prepare_cod_snapshot(
        cif_dir=snapshot,
        output_dir=tmp_path / "first",
        source_url="https://example.invalid/cod-snapshot.tar.gz",
        source_revision="2026-10-12 sha256:example",
        seed=42,
    )
    assert (
        main(
            [
                "cod",
                "prepare",
                "--cif-dir",
                str(snapshot),
                "--output-dir",
                str(tmp_path / "second"),
                "--source-url",
                "https://example.invalid/cod-snapshot.tar.gz",
                "--source-revision",
                "2026-10-12 sha256:example",
                "--seed",
                "42",
            ]
        )
        == 0
    )
    assert "Prepared 5 COD entries (1 rejected)" in capsys.readouterr().out
    second_manifest = json.loads((tmp_path / "second" / "manifest.json").read_text())

    first_manifest = json.loads(first.manifest_path.read_text())
    assert first.accepted_entries == 5
    assert first.rejected_entries == 1
    assert (
        first_manifest["source"]["cif_directory_sha256"]
        == second_manifest["source"]["cif_directory_sha256"]
    )
    assert first_manifest["split"]["systems"]["cubic"] == {"accepted": 5, "training": 4, "test": 1}
    assert first_manifest["split"]["systems"]["tetragonal"] == {
        "accepted": 0,
        "training": 0,
        "test": 0,
    }
    assert (
        first_manifest["selection"]["niggli_reduction"]
        == "performed before global duplicate removal and train/test splitting"
    )
    np.testing.assert_array_equal(
        np.load(first.output_dir / "splits" / "test_cubic.npy"),
        np.load(tmp_path / "second" / "splits" / "test_cubic.npy"),
    )
    rejected = (first.output_dir / "rejected_entries.csv").read_text()
    assert "broken_1000010.cif" in rejected


def test_global_three_decimal_keys_preserve_unrounded_cells_before_splitting(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    # Equal keys across two COD symmetry labels must never cross the split.
    for identifier, length, group in [
        (1000001, 5.0001, 221),
        (1000002, 5.0004, 123),
        (1000003, 5.0008, 221),
    ]:
        (snapshot / f"{identifier}.cif").write_text(_cif(length, identifier, group))
    result = prepare_cod_snapshot(
        cif_dir=snapshot,
        output_dir=tmp_path / "prepared",
        source_url="https://example.invalid/snapshot",
        source_revision="fixed-test-snapshot",
        seed=42,
    )
    assert result.accepted_entries == 2
    arrays = [np.load(p) for p in (result.output_dir / "splits").glob("*.npy")]
    nonempty = [a for a in arrays if len(a)]
    assert all(a.dtype == np.float64 for a in arrays)
    cells = np.concatenate(nonempty)
    np.testing.assert_allclose(sorted(cells[:, 0]), [5.0001, 5.0008], rtol=0, atol=1e-10)
    keys = np.round(cells, 3)
    assert len(np.unique(keys, axis=0)) == len(cells)
