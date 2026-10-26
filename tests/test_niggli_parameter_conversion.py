"""Compare compiled geometry helpers with the previous NumPy implementation."""

import numpy as np
import pytest

import niggli


def _numpy_reference(cell):
    # Preserve the previous formulas, scalar dtypes, sentinel, and exception
    # handling independently of the compiled parameter-conversion helpers.
    a, b, c = cell[:3]
    alpha, beta, gamma = (np.deg2rad(value) for value in cell[3:])
    v1 = np.array([a, 0, 0])
    v2 = np.array([b*np.cos(gamma), b*np.sin(gamma), 0.0])
    cosalphastar = (np.cos(beta)*np.cos(gamma) - np.cos(alpha))/(np.sin(beta)*np.sin(gamma))
    if 1 - cosalphastar**2 < 0.0:
        return [1000, 1000, 1000, 90, 90, 90]
    sinalphastar = np.sqrt(1 - cosalphastar**2)
    try:
        v3 = np.array([c*np.cos(beta), -c*np.sin(beta)*cosalphastar, c*np.sin(beta)*sinalphastar])
        reduced, _, _, _ = niggli.fast_niggli_reduction(np.transpose([v1, v2, v3]))
        v1, v2, v3 = reduced[:, 0], reduced[:, 1], reduced[:, 2]
        a, b, c = np.linalg.norm(v1), np.linalg.norm(v2), np.linalg.norm(v3)
        return [
            a, b, c,
            np.rad2deg(np.arccos(np.dot(v2, v3)/(b*c))),
            np.rad2deg(np.arccos(np.dot(v1, v3)/(a*c))),
            np.rad2deg(np.arccos(np.dot(v1, v2)/(a*b))),
        ]
    except RuntimeError:
        return [1000, 1000, 1000, 90, 90, 90]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_parameter_conversion_matches_numpy_at_symmetry_boundaries(dtype):
    cases = [
        [7.1, 8.2, 10.3, 79, 83, 102], [7, 9, 11, 90, 104, 90],
        [7, 8, 9, 90, 90, 90], [7, 7, 9, 90, 90, 90],
        [7, 7, 9, 90, 90, 120], [7, 7, 7, 80, 80, 80],
        [7, 7, 7, 90, 90, 90],
    ]
    for offset in (-1e-4, -1e-5, -1e-8, 0, 1e-8, 1e-5, 1e-4):
        cases.extend([
            [7, 7 + offset, 9, 90, 90, 90],
            [7, 9, 9 + offset, 90, 90, 90],
            [7, 7, 7, 90 + offset, 90 + offset, 90 + offset],
            [7, 7, 9, 90, 90, 120 + offset],
            [7, 7, 7, 60 + offset, 60 + offset, 60 + offset],
        ])
    rng = np.random.default_rng(42)
    cases.extend(np.column_stack((
        rng.uniform(4, 30, (128, 3)), rng.uniform(75, 105, (128, 3))
    )))
    for case in cases:
        cell = np.array(case, dtype=dtype)
        before = cell.copy()
        expected = np.array(_numpy_reference(cell))
        actual = np.array(niggli.cell_to_niggli(cell))
        # float32 transcendental functions can differ by a few ulps between
        # NumPy and Numba. Float64 retains much tighter equivalence.
        atol = 3e-5 if dtype == np.float32 else 1e-11
        np.testing.assert_allclose(actual, expected, rtol=0, atol=atol)
        assert np.array_equal(cell, before)
        assert isinstance(niggli.cell_to_niggli(cell), list)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("cell", [
    [4, 5, 6, 60, 60, 140],  # impossible angular geometry
    [0, 5, 6, 90, 90, 90],  # zero-volume cell
    [4, 5, 6, 90, 90, 0],   # singular gamma
    [4, 5, 6, 60, 60, 120], # coplanar boundary
    [4, 5, 6, 60, 60, 120 - 1e-5],
    [4, 5, 6, 60, 60, 120 + 1e-5],
])
def test_degenerate_cells_keep_numpy_failure_behavior(cell, dtype):
    cell = np.array(cell, dtype=dtype)
    with np.errstate(all="ignore"):
        try:
            expected = _numpy_reference(cell)
        except Exception as error:
            with pytest.raises(type(error)):
                niggli.cell_to_niggli(cell)
        else:
            np.testing.assert_allclose(
                niggli.cell_to_niggli(cell), expected, rtol=0,
                atol=3e-5 if dtype == np.float32 else 1e-11, equal_nan=True
            )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_compiled_vector_construction_retains_float64_basis(dtype):
    cell = np.array([7, 8, 9, 80, 90, 100], dtype=dtype)
    basis, valid = niggli._cell_vectors(
        cell[0], cell[1], cell[2], *(np.deg2rad(value) for value in cell[3:])
    )
    assert valid
    assert basis.dtype == np.float64
    assert basis.flags.f_contiguous


def test_reduction_runtime_error_keeps_existing_sentinel(monkeypatch):
    def fail(original):
        raise RuntimeError("reduction failed")

    monkeypatch.setattr(niggli, "fast_niggli_reduction", fail)
    assert niggli.cell_to_niggli([7, 8, 9, 90, 90, 90]) == [1000, 1000, 1000, 90, 90, 90]
