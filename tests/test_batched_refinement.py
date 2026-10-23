"""Numerical invariants for the independent orientation refinement backend."""

import itertools
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment

from gidind_ai.batched_refinement import (
    Geometry,
    hungarian,
    match,
    reciprocal_vectors,
    refine_batch,
    residuals,
    resolve_backend,
    safe_process_count,
    simulate,
)


def geometry(device="cpu"):
    laue = torch.tensor(
        list(itertools.product(range(-2, 3), repeat=3)), dtype=torch.float64, device=device
    )
    laue = laue[laue.abs().sum(-1) > 0]
    return Geometry(
        torch.tensor([4.0] * 3 + [60.0] * 3, dtype=torch.float64, device=device),
        torch.tensor([26.0] * 3 + [60.0] * 3, dtype=torch.float64, device=device),
        laue,
        2.0,
        2.0,
        False,
        0.05,
    )


@pytest.mark.parametrize("rows,cols", [(1, 1), (3, 7), (7, 3), (8, 8), (0, 4), (4, 0)])
def test_hungarian_matches_scipy(rows, cols):
    cost = torch.rand(
        (9, rows, cols), generator=torch.Generator().manual_seed(4), dtype=torch.float64
    )
    ri, ci = hungarian(cost)
    for i in range(len(cost)):
        r, c = linear_sum_assignment(cost[i].numpy())
        assert len(set(ri[i].tolist())) == min(rows, cols)
        assert len(set(ci[i].tolist())) == min(rows, cols)
        np.testing.assert_allclose(cost[i, ri[i], ci[i]].sum(), cost[i, r, c].sum(), atol=1e-12)


def test_hungarian_ties_and_permutation():
    cost = torch.ones((4, 5, 9), dtype=torch.float64)
    a, b = hungarian(cost)
    c, d = hungarian(cost.flip(0))
    torch.testing.assert_close(a, c.flip(0))
    torch.testing.assert_close(b, d.flip(0))


@pytest.mark.parametrize("method", ["one2one", "observed", "simulated", "simulated_fractional"])
def test_matching_weights_and_cutoffs_match_legacy(method):
    from gixd_data_generator import GixdDataGenerator
    from polishing_fit import match_laue_indices

    generator = GixdDataGenerator("uniform", "testing", load_unit_cell_generator=False)
    generator.device = torch.device("cpu")
    g = Geometry.from_generator(generator, "cpu")
    cells = torch.tensor([[8.0, 9.0, 10.0, 91.0, 100.0, 110.0]], dtype=torch.float64)
    orientations = torch.tensor([[1, 0, 2]])
    obs = simulate(cells[0], orientations[0], g.laue)[::17].clone()
    obs[:, 0] += 0.001
    obs = obs[(obs[:, 0] > 0.05) & (obs < 2).all(-1)]
    pack = match(cells, orientations, obs[None], g, 0.07, method)
    rec = reciprocal_vectors(cells, orientations)[0]
    im, _, _ = match_laue_indices(
        g.laue,
        rec[:, 0],
        rec[:, 1],
        rec[:, 2],
        obs[:, 0],
        obs[:, 1],
        generator,
        0.07,
        method=method,
    )
    keep = pack.mask[0]
    # Stable comparison by (observed index, Miller indices); tied optimal
    # one-to-one assignments may use a different degenerate Miller index.
    actual = pack.observed[0, keep]
    assert len(actual) == len(im)
    if method != "one2one":
        torch.testing.assert_close(pack.weights[0, keep].sort().values, im[:, 6].sort().values)
    torch.testing.assert_close(
        residuals((cells - g.minimum) / g.span, orientations, pack, g).square().sum(),
        (
            (simulate(cells[0], orientations[0], im[:, :3]) - im[:, 4:6]).square().sum(-1)
            * im[:, 6]
        ).sum()
        / im[:, 6].sum().clamp_min(1e-6),
        rtol=1e-7,
        atol=1e-12,
    )


def test_rectangular_assignment_ignores_invalid_padding_and_applies_cutoff_afterwards():
    g = geometry()
    cells = torch.tensor([[8.0, 9.0, 10.0, 91.0, 100.0, 110.0]] * 2, dtype=torch.float64)
    orientations = torch.tensor([[0, 0, 1], [1, 0, 2]])
    observed = torch.tensor(
        [[[0.4, 0.7], [1.2, 0.9], [0.0, 0.0]], [[0.4, 0.7], [0.0, 0.0], [0.0, 0.0]]],
        dtype=torch.float64,
    )
    pack = match(cells, orientations, observed, g, 0.15, "one2one")
    for i, count in enumerate([2, 1]):
        sim = pack.simulated[i, pack.simulated_mask[i]].numpy()
        cost = np.linalg.norm(observed[i, :count].numpy()[:, None] - sim[None], axis=-1)
        r, c = linear_sum_assignment(cost)
        assert pack.mask[i].sum() == (cost[r, c] <= 0.15).sum()


def test_geometry_and_fixed_assignment_objective_match_legacy():
    from gixd_data_generator import GixdDataGenerator
    from gixd_utility import GixdUtility
    from polishing_fit import direct_inversion_loss

    g = geometry()
    cells = torch.tensor([[8.0, 9.0, 10.0, 91.0, 100.0, 110.0]] * 4, dtype=torch.float64)
    orientations = torch.tensor([[0, 0, 1], [1, 0, 0], [1, 0, 2], [1, -1, 1]])
    actual = reciprocal_vectors(cells, orientations)
    for i in range(4):
        cell = torch.cat((cells[i, :3], torch.deg2rad(cells[i, 3:])))
        vectors = GixdDataGenerator.calc_real_vectors_from_unit_cell_rad(cell)
        rotation = GixdDataGenerator.calc_rotation_matrix(*vectors, orientations[i : i + 1])
        rotated = GixdDataGenerator.rotate_cell_by_matrix(*vectors, rotation)
        expected = torch.stack(GixdDataGenerator.calc_reciprocal_cell(*rotated), -1)[0]
        torch.testing.assert_close(actual[i], expected, rtol=1e-11, atol=1e-13)
    observed = simulate(cells[0], orientations[0], g.laue)
    pack = match(
        cells, orientations, observed[None].expand(4, -1, -1), g, 0.3, "simulated_fractional"
    )
    x = ((cells - g.minimum) / g.span).requires_grad_()
    losses = residuals(x, orientations, pack, g).square().sum(-1)
    dummy = SimpleNamespace(
        calc_real_vectors_from_unit_cell_rad=GixdDataGenerator.calc_real_vectors_from_unit_cell_rad,
        calc_rotation_matrix=GixdDataGenerator.calc_rotation_matrix,
        rotate_cell_by_matrix=GixdDataGenerator.rotate_cell_by_matrix,
        calc_reciprocal_cell=GixdDataGenerator.calc_reciprocal_cell,
    )
    for i in range(4):
        mask = pack.mask[i]
        im = torch.cat(
            (
                pack.hkl[i, mask],
                pack.observed_indices[i, mask, None],
                pack.observed[i, mask],
                pack.weights[i, mask, None],
            ),
            -1,
        )
        row = x[i].detach().requires_grad_()
        legacy = direct_inversion_loss(row, im, dummy, tuple(orientations[i].tolist()))
        torch.testing.assert_close(losses[i], legacy, atol=1e-12, rtol=1e-10)
        old_grad = torch.autograd.grad(legacy, row)[0]
        new_grad = torch.autograd.grad(losses[i], x, retain_graph=True)[0][i]
        torch.testing.assert_close(new_grad, old_grad, atol=1e-10, rtol=1e-8)
        torch.testing.assert_close(GixdUtility.denormalize_cell_deg(row), cells[i])


@pytest.mark.parametrize(
    "method", ["one2one", "observed", "simulated", "simulated_fractional", "hybrid"]
)
def test_fits_are_independent_of_batch_permutation_and_failed_candidate(method):
    g = geometry()
    truth = torch.tensor([8.0, 9.0, 10.0, 91.0, 100.0, 110.0], dtype=torch.float64)
    orientations = torch.tensor([[0, 0, 1], [1, 0, 2]])
    observations = simulate(truth, orientations[0], g.laue)
    observations = observations[(observations[:, 0] > 0.05) & (observations < 2).all(-1)]
    initial = truth.repeat(2, 1)
    initial[:, :3] += 0.03
    options = {"q_cutoff": 0.2, "matching_method": method, "maxiter": 10, "outer_loops": 2}
    batch = refine_batch(initial, observations, orientations, g, **options)
    for i in range(2):
        single = refine_batch(
            initial[i : i + 1], observations, orientations[i : i + 1], g, **options
        )
        torch.testing.assert_close(batch["cells"][i], single["cells"][0], atol=1e-8, rtol=1e-9)
        torch.testing.assert_close(
            batch["uncertainties"][i],
            single["uncertainties"][0],
            atol=1e-8,
            rtol=1e-7,
            equal_nan=True,
        )
    invalid = initial[:1].clone()
    invalid[:, 0] = float("nan")
    extended = refine_batch(
        torch.cat((initial.flip(0), invalid)),
        observations,
        torch.cat((orientations.flip(0), orientations[:1])),
        g,
        **options,
    )
    torch.testing.assert_close(batch["cells"], extended["cells"][:2].flip(0), atol=1e-8, rtol=1e-9)
    assert not extended["valid"][-1]
    assert torch.isnan(extended["uncertainties"][-1]).all()
    assert batch["loss_history"].shape[0] == 2


def test_empty_observations_and_exact_solution():
    g = geometry()
    cells = torch.tensor([[8.0, 9.0, 10.0, 91.0, 100.0, 110.0]], dtype=torch.float64)
    orientations = torch.tensor([[0, 0, 1]])
    empty = refine_batch(cells, torch.empty((0, 2)), orientations, g)
    assert not empty["valid"].any()
    assert torch.isinf(empty["losses"]).all()
    exact = refine_batch(cells, simulate(cells[0], orientations[0], g.laue), orientations, g)
    torch.testing.assert_close(exact["cells"], cells)
    assert exact["losses"].max() < 1e-20


def test_backend_release_gate_and_process_safety():
    assert resolve_backend("auto", "cuda") == "serial"
    assert resolve_backend("batched", "cpu") == "batched"
    assert safe_process_count("cuda", 20) == 1
    assert safe_process_count("cpu", 1) == 1
    with pytest.raises(ValueError, match="multiphase"):
        safe_process_count("cuda", 2, supported=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware required")
@pytest.mark.parametrize('use_specular', [False, True])
@pytest.mark.parametrize('method', ['one2one', 'simulated_fractional', 'hybrid'])
def test_cuda_assignment_and_fit_match_cpu(use_specular, method):
    from dataclasses import replace
    cost = torch.rand((49, 15, 80), dtype=torch.float64)
    ri, ci = hungarian(cost)
    gr, gc = hungarian(cost.cuda())
    torch.testing.assert_close(
        cost[torch.arange(49)[:, None], ri, ci].sum(-1),
        cost.cuda()[torch.arange(49, device="cuda")[:, None], gr, gc].sum(-1).cpu(),
    )

    truth = torch.tensor([8.0, 9.0, 10.0, 91.0, 100.0, 110.0], dtype=torch.float64)
    orientations = torch.tensor([[0, 0, 1], [1, 0, 2]])
    observed = simulate(truth, orientations[0], geometry().laue)
    keep = (observed < 2).all(-1)
    if not use_specular: keep = keep & (observed[:, 0] > 0.05)
    observed = observed[keep]
    initial = truth.repeat(2, 1)
    initial[:, :3] += 0.03
    kwargs = {"q_cutoff": 0.2, "maxiter": 10, "outer_loops": 2, "matching_method": method}
    cpu = refine_batch(initial, observed, orientations, replace(geometry(),use_specular=use_specular), **kwargs)
    gpu = refine_batch(
        initial.cuda(), observed.cuda(), orientations.cuda(), replace(geometry("cuda"),use_specular=use_specular), **kwargs
    )
    for key in ("cells", "losses", "uncertainties"):
        torch.testing.assert_close(cpu[key], gpu[key].cpu(), atol=1e-7, rtol=1e-5, equal_nan=True)


def test_uncertainty_uses_legacy_hessian_and_unweighted_variance():
    from gixd_data_generator import GixdDataGenerator
    from polishing_fit import direct_inversion_loss

    g = geometry()
    truth = torch.tensor([[8.0, 9.0, 10.0, 91.0, 100.0, 110.0]], dtype=torch.float64)
    orientations = torch.tensor([[1, 0, 2]])
    observed = simulate(truth[0], orientations[0], g.laue)
    observed = observed[(observed[:, 0] > 0.05) & (observed < 2).all(-1)]
    observed = observed + 0.001
    result = refine_batch(
        truth,
        observed,
        orientations,
        g,
        matching_method="simulated_fractional",
        maxiter=1,
        outer_loops=1,
    )
    pack = result["final_matches"]
    keep = pack.mask[0]
    im = torch.cat(
        (
            pack.hkl[0, keep],
            pack.observed_indices[0, keep, None],
            pack.observed[0, keep],
            pack.weights[0, keep, None],
        ),
        -1,
    )
    dummy = SimpleNamespace(
        calc_real_vectors_from_unit_cell_rad=GixdDataGenerator.calc_real_vectors_from_unit_cell_rad,
        calc_rotation_matrix=GixdDataGenerator.calc_rotation_matrix,
        rotate_cell_by_matrix=GixdDataGenerator.rotate_cell_by_matrix,
        calc_reciprocal_cell=GixdDataGenerator.calc_reciprocal_cell,
    )
    x = (result["cells"][0] - g.minimum) / g.span
    h = torch.autograd.functional.hessian(
        lambda row: direct_inversion_loss(row, im, dummy, (1, 0, 2)), x
    )
    inv = torch.linalg.pinv((h + h.T) / 2 + 1e-6 * torch.eye(6))
    residual = (
        simulate(result["cells"][0], orientations[0], pack.hkl[0, keep]) - pack.observed[0, keep]
    )
    variance = residual.square().sum() / max(1, 2 * int(keep.sum()) - 6)
    expected = (inv.diagonal() * variance).sqrt() * g.span
    torch.testing.assert_close(
        result["uncertainties"][0], expected, rtol=1e-6, atol=1e-8, equal_nan=True
    )


def test_cutoff_includes_equality_and_excludes_next_lower_cutoff():
    g = geometry()
    cell = torch.tensor([[8.0, 9.0, 10.0, 91.0, 100.0, 110.0]], dtype=torch.float64)
    orientation = torch.tensor([[0, 0, 1]])
    simulated = simulate(cell, orientation, g.laue)
    point = simulated[0, (simulated[0, :, 0] > 0.1) & (simulated[0] < 1.8).all(-1)][0]
    observed = (point + point.new_tensor([0.001, 0.002]))[None, None]
    pack = match(cell, orientation, observed, g, 1.0, "one2one")
    distance = torch.linalg.vector_norm(
        pack.observed[0, 0] - pack.simulated[0, pack.simulated_indices[0, 0]]
    )
    equal = match(cell, orientation, observed, g, float(distance), "one2one")
    below = match(
        cell,
        orientation,
        observed,
        g,
        float(torch.nextafter(distance, distance.new_tensor(0.0))),
        "one2one",
    )
    assert equal.mask.sum() == 1
    assert below.mask.sum() == 0
