"""Statistical and integration checks for the fresh DE ensemble implementation."""

import itertools
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from gidind_ai.batched_refinement import Geometry, peak_mask, simulate
from gidind_ai.mcmc import (
    MCMCSettings,
    PeakPosterior,
    cell_support,
    diagnostics,
    initialize,
    make_streams,
    run_mcmc,
    sample_ensemble,
    validate_sigma,
)


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def geometry(device="cpu"):
    return Geometry(
        torch.tensor([4.0] * 3 + [60.0] * 3, dtype=torch.float64, device=device),
        torch.tensor([26.0] * 3 + [60.0] * 3, dtype=torch.float64, device=device),
        torch.tensor(
            list(itertools.product(range(-2, 3), repeat=3)), dtype=torch.float64, device=device
        ),
        2.0,
        2.0,
        False,
        0.05,
    )


def settings(**kwargs):
    return MCMCSettings(
        ensembles=2, walkers=4, burn_in=2, production=8, thin=2, reflection_chunk_size=29, **kwargs
    )


def example(g):
    cell = torch.tensor(
        [[8.0, 9.0, 10.0, 82.0, 85.0, 87.0]], dtype=torch.float64, device=g.minimum.device
    )
    orientation = torch.tensor([[0, 0, 1]], device=g.minimum.device)
    points = simulate(cell[0], orientation[0], g.laue)
    obs = points[peak_mask(points, g, simulated=True)][::7].clone()
    obs += 0.001
    return cell, orientation, obs


@pytest.mark.parametrize(
    "kwargs",
    [
        {"walkers": 3},
        {"ensembles": 1},
        {"thin": 0},
        {"burn_in": -1},
        {"production": 1},
        {"jitter": 0},
        {"gamma": float("nan")},
    ],
)
def test_settings_reject_invalid_values(kwargs):
    with pytest.raises(ValueError):
        MCMCSettings(**kwargs)


@pytest.mark.parametrize("sigma", [None, 0, -1, float("inf"), float("nan")])
def test_sigma_required(sigma):
    with pytest.raises(ValueError, match="q_sigma"):
        validate_sigma(sigma)


def test_gaussian_likelihood_matches_hand_calculation_and_chunking():
    g = geometry()
    cell, hkl, obs = example(g)
    # Duplicate the observation to verify many-to-one assignment and summed loss.
    obs = torch.cat((obs[:1], obs[:1], torch.tensor([[1.95, 1.95]], dtype=torch.float64)))
    x = ((cell - g.minimum) / g.span)[:, None, None, :].expand(1, 2, 4, 6)
    points = simulate(cell[0], hkl[0], g.laue)
    points = points[peak_mask(points, g, simulated=True)]
    sse = ((obs[:, None] - points[None]).square().sum(-1).min(-1).values).sum()
    expected = -sse / (2 * 0.01**2) - len(obs) * np.log(2 * np.pi * 0.01**2)
    for chunk in (1, 17, 1000):
        posterior = PeakPosterior(obs, hkl, g, 0.01, 1.5, chunk)
        torch.testing.assert_close(posterior(x), expected.expand(1, 2, 4))


def test_invalid_support_empty_simulations_and_empty_observations():
    g = geometry()
    cell, hkl, obs = example(g)
    posterior = PeakPosterior(obs, hkl, g, 0.01, 1.5)
    x = ((cell - g.minimum) / g.span)[:, None, None, :].expand(1, 2, 4, 6).clone()
    x[0, 0, 0, 0] = -1
    x[0, 0, 1, 3:] = 1  # 120,120,120 has zero Gram determinant
    x[0, 0, 2, 0] = float("nan")
    assert torch.isneginf(posterior(x)[0, 0, :3]).all()
    assert not cell_support(cell, g, 30).any()
    empty = replace(g, laue=torch.zeros((0, 3), dtype=torch.float64))
    assert torch.isneginf(PeakPosterior(obs, hkl, empty, 0.01, 1.5)(x)).all()
    with pytest.raises(ValueError, match="eligible observed"):
        PeakPosterior(torch.zeros((2, 2), dtype=torch.float64), hkl, g, 0.01, 1.5)


def test_reassignment_changes_with_cell_and_no_derivatives():
    g = geometry()
    cell, hkl, obs = example(g)
    posterior = PeakPosterior(obs, hkl, g, 0.01, 1.5)
    x = ((cell - g.minimum) / g.span)[:, None, None, :].requires_grad_()
    first = posterior(x)
    shifted = x.detach().clone()
    shifted[..., 0] += 0.02
    assert not first.requires_grad
    assert not torch.equal(first, posterior(shifted))


def test_initialization_reduces_only_starting_cells(monkeypatch):
    import niggli

    calls = []

    def reduction(cell):
        calls.append(cell.copy())
        return cell

    monkeypatch.setattr(niggli, "cell_to_niggli", reduction)
    g = geometry()
    cell, hkl, obs = example(g)
    config = settings()
    posterior = PeakPosterior(obs, hkl, g, 0.02, 1.5)
    initial, failures = initialize(cell, posterior, config, [0])
    assert not failures
    assert len(calls) == 8
    result = sample_ensemble(posterior, initial, config, make_streams(config, [0], "cpu"))
    assert len(calls) == 8  # no reduction inside proposals
    assert result["samples"].shape == (1, 2, 4, 4, 6)


def test_orientation_chunking_and_reproducibility_and_artifact(tmp_path):
    g = geometry()
    cell, hkl, obs = example(g)
    cells = cell.repeat(2, 1)
    hkls = torch.cat((hkl, torch.tensor([[1, 0, 2]])))
    path = tmp_path / "posterior.npz"
    a = run_mcmc(cells, obs, hkls, g, q_sigma=0.02, settings=settings(), output_path=path)
    with np.load(path) as z:
        draws, scores = z["samples"].copy(), z["log_likelihood"].copy()
        assert z["sample_axes"].tolist() == [
            "orientation",
            "ensemble",
            "draw",
            "walker",
            "parameter",
        ]
    b = run_mcmc(
        cells,
        obs,
        hkls,
        g,
        q_sigma=0.02,
        settings=settings(orientation_chunk_size=2),
        output_path=path,
    )
    with np.load(path) as z:
        np.testing.assert_array_equal(z["samples"], draws)
        np.testing.assert_array_equal(z["log_likelihood"], scores)
    assert a["candidates"] == b["candidates"]
    assert not a["candidates"][0]["is_converged"]
    json.dumps(a, allow_nan=False)


def test_physical_q_scaling_and_covariance(tmp_path):
    g = geometry()
    cell, hkl, obs = example(g)
    path = tmp_path / "samples.npz"
    unscaled = run_mcmc(cell, obs, hkl, g, q_sigma=0.02, settings=settings(), output_path=path)
    with np.load(path) as z:
        scores = z["log_likelihood"].copy()
        draws = z["samples"].copy()
    scaled = run_mcmc(
        cell, obs, hkl, g, q_sigma=0.04, q_scale=2, settings=settings(), output_path=path
    )
    with np.load(path) as z:
        expected = draws.copy()
        expected[..., :3] /= 2
        np.testing.assert_array_equal(z["samples"], expected)
        np.testing.assert_allclose(z["log_likelihood"], scores - 2 * len(obs) * np.log(2))
    a, b = unscaled["candidates"][0], scaled["candidates"][0]
    factor = np.array([0.5] * 3 + [1.0] * 3)
    np.testing.assert_allclose(b["posterior_std"], np.array(a["posterior_std"]) * factor)
    np.testing.assert_allclose(
        b["posterior_covariance"],
        np.array(a["posterior_covariance"]) * factor[:, None] * factor[None],
    )


def test_initialization_failure_is_explicit_and_json_safe(tmp_path):
    g = geometry()
    cell, hkl, obs = example(g)
    result = run_mcmc(
        cell,
        obs,
        hkl,
        g,
        q_sigma=0.02,
        min_height=100,
        settings=settings(initialization_attempts=2),
        output_path=tmp_path / "failed.npz",
    )
    assert result["candidates"][0]["status"] == "initialization_failed"
    assert len(result["initialization_failures"]) == 8
    assert not result["candidates"][0]["is_converged"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("bounded", [False, True])
def test_de_recovers_known_gaussian_and_uniform_moments(bounded):
    config = MCMCSettings(ensembles=4, walkers=16, burn_in=700, production=2000, thin=2)
    generator = torch.Generator().manual_seed(124)
    initial = torch.randn((1, 4, 16, 2), dtype=torch.float64, generator=generator) * 0.2

    def logp(x):
        if bounded:
            return torch.where((x.abs() < 1).all(-1), 0.0, -torch.inf)
        # Correlated non-isotropic normal with known covariance.
        z0 = x[..., 0]
        z1 = (x[..., 1] - 0.7 * z0) / 0.5
        return -(z0.square() + z1.square()) / 2

    result = sample_ensemble(logp, initial, config, make_streams(config, [0], "cpu"))
    flat = result["samples"].reshape(-1, 2)
    covariance = np.eye(2) / 3 if bounded else np.array([[1, 0.7], [0.7, 0.74]])
    # Conservative errors based on independent ensemble averages and a fixed floor.
    ensemble_means = result["samples"][0].mean(axis=(1, 2))
    mcse = ensemble_means.std(axis=0, ddof=1) / np.sqrt(config.ensembles)
    assert np.all(abs(flat.mean(0)) < 5 * mcse + 0.015)
    np.testing.assert_allclose(np.cov(flat, rowvar=False), covariance, atol=0.055, rtol=0.08)
    assert np.all((result["acceptance_rate"] > 0.05) & (result["acceptance_rate"] < 0.95))
    if bounded:
        assert (abs(flat) < 1).all()


def test_diagnostics_detect_independent_chains_stuck_and_shifted_chains():
    rng = np.random.default_rng(42)
    iid = rng.normal(size=(4, 2000, 4, 6))
    good = diagnostics(iid)
    assert good["is_converged"]
    shifted = iid.copy()
    shifted[0] += 4
    assert not diagnostics(shifted)["is_converged"]
    stuck = diagnostics(np.ones_like(iid))
    assert not stuck["is_converged"]
    assert stuck["max_rhat"] is None
    assert stuck["min_ess_bulk"] == 0


def test_cli_requires_sigma_before_checkpoint_resolution(capsys):
    from gidind_ai.cli import main

    assert (
        main(
            [
                "infer",
                "--input",
                "missing.csv",
                "--output-dir",
                "unused",
                "--refinement-algorithm",
                "mcmc",
            ]
        )
        == 2
    )
    assert "q_sigma" in capsys.readouterr().err


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_matches_cpu_likelihood_and_geometry():
    cpu, gpu = geometry(), geometry("cuda")
    cell, hkl, obs = example(cpu)
    x = ((cell - cpu.minimum) / cpu.span)[:, None, None, :].expand(1, 2, 4, 6)
    a = PeakPosterior(obs, hkl, cpu, 0.02, 1.5, 17)(x)
    b = PeakPosterior(obs.cuda(), hkl.cuda(), gpu, 0.02, 1.5, 17)(x.cuda())
    torch.testing.assert_close(a, b.cpu(), rtol=1e-10, atol=1e-10)


def test_conditional_diffraction_posterior_matches_numerical_integration():
    g = replace(geometry(), laue=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64))
    cell, hkl, _ = example(geometry())
    cell[:] = torch.tensor([8.0, 9.0, 10.0, 90.0, 90.0, 90.0])
    observed = simulate(cell[0], hkl[0], g.laue) + 0.002
    posterior = PeakPosterior(observed, hkl, g, 0.01, 1.5)
    base = ((cell - g.minimum) / g.span)[0]
    lower, upper = (7 - 4) / 26, (9 - 4) / 26

    def logp(x):
        full = base.expand(*x.shape[:-1], 6).clone()
        full[..., 0] = x[..., 0]
        value = posterior(full)
        return torch.where((x[..., 0] > lower) & (x[..., 0] < upper), value, -torch.inf)

    grid = torch.linspace(lower, upper, 5001, dtype=torch.float64)
    density = logp(grid.reshape(1, 1, -1, 1)).flatten().exp().numpy()
    density /= density.sum()
    grid_a = 4 + 26 * grid.numpy()
    expected_mean = (density * grid_a).sum()
    expected_std = np.sqrt((density * (grid_a - expected_mean) ** 2).sum())
    config = MCMCSettings(ensembles=4, walkers=8, burn_in=300, production=1200, thin=2)
    initial = torch.full((1, 4, 8, 1), float(base[0]), dtype=torch.float64)
    initial += torch.randn(initial.shape, generator=torch.Generator().manual_seed(4)) * 0.005
    result = sample_ensemble(logp, initial, config, make_streams(config, [0], "cpu"))
    physical = result["samples"].flatten() * 26 + 4
    assert abs(physical.mean() - expected_mean) < 0.015
    assert abs(physical.std() - expected_std) < 0.012
    # Same single reflection and fixed assignment: local Laplace width agrees
    # with the small-noise conditional posterior to first order.
    fitted_a = 2 * np.pi / float(observed[0, 0])
    laplace_std = 0.01 * fitted_a**2 / (2 * np.pi)
    assert abs(expected_std - laplace_std) / laplace_std < 0.02
    # A fixed-seed noisy diffraction case recovers the generating lattice length.
    assert abs(physical.mean() - 8) < 0.1
    np.testing.assert_allclose(
        np.quantile(physical, [0.025, 0.975]),
        np.interp([0.025, 0.975], density.cumsum(), grid_a),
        atol=0.025,
    )


def test_inference_mcmc_writes_reports_figures_and_samples(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from gidind_ai import workflows

    g = geometry()
    cell, hkl, obs = example(g)
    target = SimpleNamespace(
        min_distance=4.0, max_distance=30.0, min_angle=60.0, max_angle=120.0, min_cell_height=1.5
    )
    features = SimpleNamespace(
        miller_indices=hkl.tolist(),
        max_q_xy=2.0,
        max_q_z=2.0,
        use_specular_scan=False,
        specular_cutoff=0.05,
    )
    generator = SimpleNamespace(
        target_config=target,
        feature_config=features,
        laue_indices_tensor=g.laue,
        calc_images_from_q_points=lambda *args: torch.zeros((1, 1, 4, 4)),
    )
    normalized = (cell - g.minimum) / g.span
    regressor = SimpleNamespace(feature_config=features, gixd_model=lambda _: normalized)
    monkeypatch.setattr(workflows, "_build_regressor", lambda *args: (regressor, generator))
    monkeypatch.setattr(
        workflows, "_load_and_filter_peaks", lambda *args: (obs[:, 0], obs[:, 1], 1.0)
    )
    source = tmp_path / "input.csv"
    source.write_text("q_xy,q_z\n1,1\n")
    result = workflows.run_inference(
        input_path=source,
        output_dir=tmp_path,
        checkpoint_info=workflows.CheckpointInfo(tmp_path / "fake.ckpt", "local"),
        device_name="cpu",
        refinement_algorithm="mcmc",
        refinement={"algorithm": "mcmc"},
        q_sigma=0.02,
        mcmc_settings=settings(),
    )
    assert result["runtime"]["algorithm"] == "mcmc"
    assert "posterior_std" in result["candidates"][0]
    assert "laplace_uncertainty" not in result["candidates"][0]
    assert (tmp_path / "posterior_samples.npz").is_file()
    assert (tmp_path / "posterior_orientation_0.png").is_file()
    assert "Posterior standard deviation" in (tmp_path / "report.txt").read_text()
    assert json.loads((tmp_path / "results.json").read_text())["candidates"] == result["candidates"]


def test_cli_forwards_mcmc_settings(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from gidind_ai import cli

    calls = []
    monkeypatch.setattr(cli, "resolve_checkpoint", lambda **kwargs: SimpleNamespace(path="fake"))
    monkeypatch.setattr(cli, "run_inference", lambda **kwargs: calls.append(kwargs))
    assert (
        cli.main(
            [
                "infer",
                "--input",
                "fake.csv",
                "--output-dir",
                str(tmp_path),
                "--refinement-algorithm",
                "mcmc",
                "--q-sigma",
                ".003",
                "--mcmc-orientation-chunk-size",
                "4",
                "--mcmc-gamma",
                ".5",
                "--mcmc-seed",
                "19",
            ]
        )
        == 0
    )
    assert calls[0]["refinement_algorithm"] == "mcmc"
    assert calls[0]["q_sigma"] == 0.003
    config = calls[0]["mcmc_settings"]
    assert (config.orientation_chunk_size, config.gamma, config.seed) == (4, 0.5, 19)


@pytest.mark.parametrize("failed", [False, True])
def test_mcmc_benchmark_noise_coverage_and_failures(monkeypatch, tmp_path, failed):
    import sys
    from types import ModuleType, SimpleNamespace

    from gidind_ai import workflows

    g = geometry()
    cell, _, obs = example(g)
    target = SimpleNamespace(
        min_distance=4.0,
        max_distance=30.0,
        min_angle=60.0,
        max_angle=120.0,
        min_cell_height=100.0 if failed else 1.5,
    )
    features = SimpleNamespace(
        miller_indices=[[0, 0, 1]],
        max_q_xy=2.0,
        max_q_z=2.0,
        use_specular_scan=False,
        specular_cutoff=0.05,
    )
    radians_cell = cell.numpy().copy()
    radians_cell[:, 3:] = np.deg2rad(radians_cell[:, 3:])
    rasterized = []

    def rasterize(x, z):
        rasterized.append((x.clone(), z.clone()))
        return torch.zeros((1, 1, 4, 4))

    generator = SimpleNamespace(
        target_config=target,
        feature_config=features,
        laue_indices_tensor=g.laue,
        device="cpu",
        generate_data_batch=lambda _: (
            radians_cell,
            None,
            None,
            obs[:, 0][None],
            obs[:, 1][None],
            torch.tensor([0]),
        ),
        calc_images_from_q_points=rasterize,
    )
    generator_module = ModuleType("gixd_data_generator")
    generator_module.create_data_generators = lambda *args, **kwargs: (
        {},
        {},
        {"uniform": generator},
    )
    regressor_module = ModuleType("gixd_regressor")
    regressor_module.GixdRegressor = lambda *args: SimpleNamespace(
        feature_config=features,
        load_checkpoint=lambda _: None,
        _predict_cells_from_images=lambda *args: cell.numpy(),
    )
    monkeypatch.setitem(sys.modules, "gixd_data_generator", generator_module)
    monkeypatch.setitem(sys.modules, "gixd_regressor", regressor_module)
    assets = SimpleNamespace(
        generator_path=tmp_path / "unused.pkl",
        source="local",
        repository=None,
        revision=None,
        release_id=None,
        manifest_path=tmp_path,
    )
    result = workflows.run_benchmark(
        output_dir=tmp_path,
        checkpoint_info=workflows.CheckpointInfo(tmp_path / "fake", "local"),
        samples=1,
        paper_assets=assets,
        refinement_algorithm="mcmc",
        q_sigma=0.02,
        mcmc_settings=settings(initialization_attempts=2),
    )
    assert result["algorithm"] == "mcmc"
    assert len(rasterized) == 1
    assert not torch.equal(rasterized[0][0][0], obs[:, 0])
    assert result["summary"]["successful_initializations"] == (0 if failed else 1)
    assert (tmp_path / "posterior_samples_0.npz").is_file()
    assert (tmp_path / "benchmark.json").is_file()
    json.dumps(workflows._json_value(result), allow_nan=False)
    if failed:
        assert result["summary"]["refined_mae"] is None
        assert result["summary"]["marginal_95_coverage"] is None
    else:
        assert len(result["summary"]["marginal_95_coverage"]) == 6
