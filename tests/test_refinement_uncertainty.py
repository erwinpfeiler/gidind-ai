"""Unresolved cell parameters cannot acquire precision from Hessian damping."""

import pytest
import torch

from gidind_ai.uncertainty import guard_uncertainties


def test_full_rank_preserves_existing_uncertainties_and_is_scale_invariant():
    uncertainty = torch.arange(1, 7, dtype=torch.float64) / 100
    jac = torch.diag(torch.arange(1, 7, dtype=torch.float64))
    for scale in [1.0, 1e-15, 1e15]:
        guarded, diagnostics = guard_uncertainties(uncertainty, jac * scale)
        assert torch.equal(guarded, uncertainty)
        assert diagnostics["rank"] == 6
        assert not diagnostics["unconstrained"].any()


def test_mixed_null_direction_marks_all_affected_parameters():
    jac = torch.eye(6, dtype=torch.float64)
    jac[0] = 0
    jac[1, :2] = torch.tensor([1.0, -1.0])
    guarded, diagnostics = guard_uncertainties(torch.ones(6), jac)
    assert diagnostics["rank"] == 5
    assert torch.isinf(guarded[:2]).all()
    assert torch.equal(guarded[2:], torch.ones(4))


def test_short_empty_and_nonfinite_jacobians_are_isolated():
    short = torch.eye(6, dtype=torch.float64)[:2]
    guarded, diagnostics = guard_uncertainties(torch.zeros(6), short)
    assert diagnostics["rank"] == 2
    assert torch.equal(guarded[:2], torch.zeros(2))
    assert torch.isinf(guarded[2:]).all()
    guarded, diagnostics = guard_uncertainties(torch.zeros(6), short[:0])
    assert diagnostics["rank"] == 0 and torch.isinf(guarded).all()
    jac = torch.eye(6, dtype=torch.float64).repeat(3, 1, 1)
    jac[1, 0, 0] = float("nan")
    jac[2, 3] = 0
    guarded, diagnostics = guard_uncertainties(torch.ones((3, 6)), jac)
    assert torch.equal(guarded[0], torch.ones(6))
    assert torch.isnan(guarded[1]).all()
    assert torch.isinf(guarded[2, 3])
    assert diagnostics["rank"].tolist() == [6, 0, 5]
    assert diagnostics["finite"].tolist() == [True, False, True]


@pytest.fixture
def generator():
    from gixd_data_generator import GixdDataGenerator

    g = GixdDataGenerator("uniform", "testing", load_unit_cell_generator=False)
    g.device = torch.device("cpu")
    return g


def test_empty_match_fallback_returns_degrees_and_no_finite_precision(generator):
    from polishing_fit import direct_inversion_pytorch

    initial = torch.tensor([8.0, 9.0, 10.0, 91.0, 100.0, 110.0], dtype=torch.float64)
    result = direct_inversion_pytorch(
        initial,
        torch.tensor([100.0]),
        torch.tensor([100.0]),
        (0, 0, 1),
        generator,
        method="cg",
        refinement_backend="serial",
        q_cutoff=1e-8,
        outer_loops=2,
        maxiter=1,
        return_uncertainties=True,
    )
    torch.testing.assert_close(result["refined_predicted_cell_deg"], initial, atol=1e-12, rtol=0)
    assert result["refined_num_matches"] == 0
    assert torch.isinf(result["refined_uncertainty_cell_deg"]).all()
    assert result["identifiability"]["rank"] == 0


@pytest.mark.parametrize("backend", ["serial", "batched"])
def test_case601_flat_b_direction_is_not_reported_as_precise(generator, backend):
    from polishing_fit import direct_inversion_pytorch

    # Recorded fitted cell and four continuous observations from case 601.
    initial = torch.tensor(
        [
            4.474852742872176,
            4.668567180633545,
            11.452000860300489,
            89.98086701465867,
            90.00331619657044,
            99.98298785136458,
        ],
        dtype=torch.float64,
    )
    observed = torch.tensor(
        [
            [1.4256960407318406, 6.23191583845977e-8],
            [1.4256960407318406, 0.5486539274956425],
            [1.4256960407318406, 1.0973079173104434],
            [1.4256960407318406, 1.645961907125244],
        ],
        dtype=torch.float64,
    )
    result = direct_inversion_pytorch(
        initial,
        observed[:, 0],
        observed[:, 1],
        (0, 0, 1),
        generator,
        method="cg",
        refinement_backend=backend,
        q_cutoff=0.056,
        matching_method="hybrid",
        maxiter=1,
        outer_loops=2,
        return_uncertainties=True,
    )
    assert torch.isinf(result["refined_uncertainty_cell_deg"][1])
    assert "b" in result["identifiability"]["unconstrained_parameters"]
    assert result["identifiability"]["rank"] < 6
    torch.testing.assert_close(
        result["refined_predicted_cell_deg"][1], initial[1], atol=1e-10, rtol=0
    )


@pytest.mark.parametrize("backend", ["serial", "batched"])
def test_informative_fit_retains_finite_uncertainty(generator, backend):
    from gidind_ai.batched_refinement import Geometry, peak_mask, simulate
    from polishing_fit import direct_inversion_pytorch

    geo = Geometry.from_generator(generator, "cpu")
    truth = torch.tensor([8.0, 9.0, 10.0, 91.0, 100.0, 110.0], dtype=torch.float64)
    observed = simulate(truth, torch.tensor([1, 0, 2]), geo.laue)
    observed = torch.unique(observed[peak_mask(observed, geo, simulated=True)], dim=0)
    result = direct_inversion_pytorch(
        truth,
        observed[:, 0],
        observed[:, 1],
        (1, 0, 2),
        generator,
        method="cg",
        refinement_backend=backend,
        q_cutoff=0.056,
        matching_method="hybrid",
        maxiter=1,
        outer_loops=2,
        return_uncertainties=True,
    )
    assert result["identifiability"]["rank"] == 6
    assert torch.isfinite(result["refined_uncertainty_cell_deg"]).all()
