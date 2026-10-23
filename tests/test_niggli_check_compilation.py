"""Compile the final reduction check without changing checks or cell geometry."""

import numpy as np
import pytest

import niggli


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_compiled_checker_matches_original_scalar_conditions(dtype):
    rng = np.random.default_rng(42)
    cases = [
        [1, 1, 1, 0, 0, 0, 1e-5],
        [1, 2, 3, 0.1, 0.2, 0.3, 1e-5],
        [1, 2, 3, -0.1, -0.2, -0.3, 1e-5],
        [0, 1, 1, 0, 0, 0, 1e-5],
        [1, 2, 3, np.nan, 0, 0, 1e-5],
    ]
    for _ in range(200):
        abc = np.sort(rng.uniform(1, 100, 3))
        off_diagonal = rng.uniform(-abc[0], abc[0], 3)
        cases.append([*abc, *off_diagonal, 1e-5])
    # Exercise equality/tolerance boundaries of the individual reduction rules.
    for offset in (-2e-5, -1e-5, 0.0, 1e-5, 2e-5):
        cases.extend([
            [1, 1 + offset, 2, 0, 0, 0, 1e-5],
            [1, 2, 2 + offset, 0, 0, 0, 1e-5],
            [1, 2, 3, 2 + offset, 0, 0, 1e-5],
            [1, 2, 3, 0, 1 + offset, 0, 1e-5],
            [1, 2, 3, 0, 0, 1 + offset, 1e-5],
        ])
    for case in cases:
        arguments = tuple(dtype(value) for value in case)
        assert niggli.niggli_check(*arguments) == niggli.niggli_check.py_func(*arguments)


@pytest.mark.parametrize("cell", [
    [7.1, 8.2, 10.3, 79, 83, 102],
    [7, 9, 11, 90, 104, 90],
    [7, 8, 9, 90, 90, 90],
    [7, 7, 9, 90, 90, 90],
    [7, 7, 9, 90, 90, 120],
    [7, 7, 7, 80, 80, 80],
    [7, 7, 7, 90, 90, 90],
])
def test_cell_reduction_is_bit_identical_with_compiled_check(monkeypatch, cell):
    cell = np.array(cell, dtype=np.float64)
    compiled = niggli.niggli_check
    actual = np.array(niggli.cell_to_niggli(cell))
    monkeypatch.setattr(niggli, "niggli_check", compiled.py_func)
    expected = np.array(niggli.cell_to_niggli(cell))
    assert np.array_equal(actual, expected)
