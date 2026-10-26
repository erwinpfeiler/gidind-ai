"""Scientific invariants for the reproducible robustness benchmark."""

import json

import numpy as np
import pytest

from gidind_ai.figure3_benchmark import correct, database, perturb, plot, settings, wilson

DOMAIN = (0.05, 2.0, 2.0)
POINTS = np.column_stack([np.linspace(0.1, 1.9, 30), np.linspace(0.2, 1.8, 30)])


def test_exact_sweep_and_shared_baseline():
    sweep = settings()
    assert len(sweep) == 61
    assert ("peaks", -20.0) in sweep and ("peaks", 20.0) in sweep
    assert sweep[-1] == ("noise", 0.02)
    assert ("noise", 0.0) not in sweep


def test_exact_nested_removals_and_saturation_without_resampling():
    p5, n5, _, _ = perturb(POINTS, "peaks", -5, 42, 0, DOMAIN)
    p20, n20, _, _ = perturb(POINTS, "peaks", -20, 42, 0, DOMAIN)
    assert n5 == 5 and n20 == 20
    assert len(p5) == 25 and len(p20) == 10
    assert set(map(tuple, p20)) <= set(map(tuple, p5))
    empty, removed, _, _ = perturb(POINTS[:5], "peaks", -20, 42, 0, DOMAIN)
    assert len(empty) == 0 and removed == 5


def test_added_points_are_exact_and_nested():
    p5, _, n5, _ = perturb(POINTS, "peaks", 5, 42, 0, DOMAIN)
    p20, _, n20, _ = perturb(POINTS, "peaks", 20, 42, 0, DOMAIN)
    assert n5 == 5 and n20 == 20
    assert np.array_equal(p5, p20[:35])
    assert len(p20) == 50


def test_noise_is_paired_bounded_and_does_not_clip():
    for mode in ("gaussian", "bounded"):
        small, *_ = perturb(POINTS, "noise", 0.01, 42, 0, DOMAIN, mode)
        large, *_ = perturb(POINTS, "noise", 0.02, 42, 0, DOMAIN, mode)
        assert np.allclose(large - POINTS, 2 * (small - POINTS))
        if mode == "bounded":
            assert np.max(np.linalg.norm(large - POINTS, axis=1)) <= 0.02
    out, _, _, dropped = perturb(POINTS, "noise", 10, 42, 0, DOMAIN)
    assert dropped > 0 and len(out) + dropped == len(POINTS)
    assert np.all(out[:, 0] < 2) and np.all(out[:, 1] < 2)


def test_success_requires_all_six_and_finite_values():
    truth = np.array([10, 11, 12, 90, 95, 100.0])
    assert correct(truth, truth + 0.09)
    for i in range(6):
        bad = truth.copy()
        bad[i] += 0.11
        assert not correct(truth, bad)
    assert not correct(truth, np.full(6, np.nan))
    assert wilson(0, 1000)[1] > 0
    assert wilson(1000, 1000)[0] < 1


def test_plot_uses_all_evaluations_and_refuses_incomplete_final(tmp_path):
    cfg = {"samples": 2, "settings": [["peaks", 0.0], ["noise", 0.02]], "noise_mode": "gaussian"}
    (tmp_path / "manifest.json").write_text(json.dumps(cfg))
    db = database(tmp_path)
    good = {"correct": True, "nn_correct": False, "error": None, "removed": 0}
    failed = {**good, "correct": False, "error": "empty"}
    for i, record in enumerate((good, failed)):
        db.execute("INSERT INTO fits VALUES(?,?,?,?)", ("peaks", 0, i, json.dumps(record)))
    db.commit()
    with pytest.raises(ValueError, match="Incomplete"):
        plot(tmp_path)
    plot(tmp_path, allow_partial=True)
    assert (tmp_path / "figure3bc_partial.pdf").exists()
    assert not (tmp_path / "figure3bc.pdf").exists()
    for i, record in enumerate((good, failed)):
        db.execute("INSERT INTO fits VALUES(?,?,?,?)", ("noise", 0.02, i, json.dumps(record)))
    db.commit()
    db.close()
    plot(tmp_path)
    import csv

    with (tmp_path / "summary.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert all(float(row["fraction"]) == 0.5 and int(row["n"]) == 2 for row in rows)
    for name in ("figure3b", "figure3c", "figure3bc"):
        for ext in ("pdf", "svg"):
            assert (tmp_path / f"{name}.{ext}").stat().st_size > 1000
