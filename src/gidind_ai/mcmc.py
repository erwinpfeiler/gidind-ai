"""Fresh DE ensemble MCMC from the mathematical DE-MC transition.

Reference: ter Braak (2006), Statistics and Computing 16, 239–249.
No historical MCMC implementation was consulted. Current project geometry and
MIT-attributed pyniggli reduction are reused; see THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .batched_refinement import Geometry, peak_mask, reciprocal_vectors, valid_cells


@dataclass(frozen=True)
class MCMCSettings:
    ensembles: int = 4
    walkers: int = 32
    burn_in: int = 2000
    production: int = 4000
    thin: int = 2
    gamma: float = 2.38 / math.sqrt(12)
    jitter: float = 1e-6
    initial_sigma: float = 0.005
    initialization_attempts: int = 1000
    seed: int = 42
    orientation_chunk_size: int = 1
    reflection_chunk_size: int = 256

    def __post_init__(self):
        for name in (
            "ensembles",
            "walkers",
            "production",
            "thin",
            "initialization_attempts",
            "orientation_chunk_size",
            "reflection_chunk_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.ensembles < 2 or self.walkers < 4 or self.walkers % 2:
            raise ValueError("MCMC needs at least two ensembles and an even walker count >= 4")
        if not isinstance(self.burn_in, int) or self.burn_in < 0:
            raise ValueError("burn_in must be a nonnegative integer")
        if self.production < self.thin:
            raise ValueError("production must be at least thin")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        for name in ("gamma", "jitter", "initial_sigma"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")


def validate_sigma(q_sigma: float | None, q_scale: float = 1.0) -> float:
    if q_sigma is None or not math.isfinite(q_sigma) or q_sigma <= 0:
        raise ValueError("MCMC requires finite positive q_sigma in physical inverse angstroms")
    if not math.isfinite(q_scale) or q_scale <= 0:
        raise ValueError("q_scale must be finite and positive")
    return q_sigma / q_scale


def cell_support(cells: Tensor, geometry: Geometry, min_height: float) -> Tensor:
    valid = valid_cells(cells)
    valid &= ((cells >= geometry.minimum) & (cells <= geometry.minimum + geometry.span)).all(-1)
    # Use safe inputs before matrix operations; invalid cells must not poison a batch.
    safe = torch.where(valid[..., None], cells, cells.new_tensor([10, 10, 10, 90, 90, 90]))
    orientation = safe.new_tensor([0, 0, 1]).expand(*safe.shape[:-1], 3)
    reciprocal = reciprocal_vectors(safe, orientation)
    heights = 1 / reciprocal.square().sum(-2).sqrt()
    return valid & torch.isfinite(heights).all(-1) & (heights >= min_height).all(-1)


class PeakPosterior:
    """Nearest-peak Gaussian residual likelihood, with a bounded uniform prior.

    This is the manuscript's reassigned Gaussian residual objective, not a
    marginal likelihood over latent peak assignments or an evidence estimator.
    Inputs have shape [orientation, ensemble, walker, 6], in normalized units.
    """

    def __init__(
        self,
        observed: Tensor,
        orientations: Tensor,
        geometry: Geometry,
        sigma: float,
        min_height: float,
        reflection_chunk_size: int = 256,
    ):
        self.geometry = geometry
        self.orientations = orientations
        self.sigma = validate_sigma(sigma)
        self.min_height = min_height
        self.chunk = reflection_chunk_size
        if not math.isfinite(min_height) or min_height < 0 or self.chunk < 1:
            raise ValueError("Invalid minimum cell height or reflection chunk size")
        if observed.ndim != 2 or observed.shape[-1] != 2:
            raise ValueError("observed must have shape [peak, 2]")
        # Padded zero peaks are excluded; all accepted observations remain in the loss.
        mask = peak_mask(observed, geometry, simulated=True)
        self.observed = observed[mask]
        if len(self.observed) == 0:
            raise ValueError("MCMC needs at least one eligible observed reflection")
        self.normalization = len(self.observed) * (math.log(2 * math.pi) + 2 * math.log(self.sigma))

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        cells = self.geometry.minimum + self.geometry.span * x
        valid = cell_support(cells, self.geometry, self.min_height)
        safe = torch.where(valid[..., None], cells, cells.new_tensor([10, 10, 10, 90, 90, 90]))
        nearest = cells.new_full((*cells.shape[:-1], len(self.observed)), float("inf"))
        orientations = self.orientations[:, None, None, :].expand(*cells.shape[:-1], 3)
        reciprocal = reciprocal_vectors(safe, orientations)
        for start in range(0, len(self.geometry.laue), self.chunk):
            hkl = self.geometry.laue[start : start + self.chunk]
            q = 2 * torch.pi * (hkl @ reciprocal.transpose(-1, -2))
            xy = q[..., :2].square().sum(-1).clamp_min(1e-32).sqrt()
            points = torch.stack((xy, q[..., 2].abs()), -1)
            eligible = peak_mask(points, self.geometry, simulated=True)
            # [O,E,W,observed,reflection]; chunked to avoid a full distance matrix.
            distances = (self.observed[:, None, :] - points[..., None, :, :]).square().sum(-1)
            distances.masked_fill_(~eligible[..., None, :], float("inf"))
            nearest = torch.minimum(nearest, distances.min(-1).values)
        logp = -(nearest.sqrt() / self.sigma).square().sum(-1) / 2 - self.normalization
        return torch.where(valid & torch.isfinite(logp), logp, -torch.inf)


def _stream_seed(seed: int, orientation: int, ensemble: int) -> int:
    # Stable across orientation chunks and unaffected by global RNG consumption.
    return int(
        np.random.SeedSequence([seed, orientation, ensemble]).generate_state(1, dtype=np.uint64)[0]
        % np.uint64(2**63 - 1)
    )


def make_streams(settings: MCMCSettings, orientation_ids: list[int], device) -> list:
    return [
        [
            torch.Generator(device=device).manual_seed(_stream_seed(settings.seed, o, e))
            for e in range(settings.ensembles)
        ]
        for o in orientation_ids
    ]


def initialize(
    initial: Tensor, posterior: PeakPosterior, settings: MCMCSettings, orientation_ids: list[int]
) -> tuple[Tensor, list[dict]]:
    from niggli import cell_to_niggli

    shape = (len(initial), settings.ensembles, settings.walkers, 6)
    states = initial.new_full(shape, float("nan"))
    failures = []
    minimum, span = posterior.geometry.minimum.cpu().numpy(), posterior.geometry.span.cpu().numpy()
    for o, orientation_id in enumerate(orientation_ids):
        for e in range(settings.ensembles):
            rng = np.random.default_rng(_stream_seed(settings.seed, orientation_id, e))
            center = (initial[o].cpu().numpy() - minimum) / span
            for w in range(settings.walkers):
                for _ in range(settings.initialization_attempts):
                    proposal = minimum + span * (center + rng.normal(0, settings.initial_sigma, 6))
                    if not np.isfinite(proposal).all() or np.any(proposal[:3] <= 0):
                        continue
                    # Reduction is initialization only; subsequent proposals stay in this basis.
                    try:
                        reduced = torch.as_tensor(
                            cell_to_niggli(proposal), device=initial.device, dtype=torch.float64
                        )
                    except (ValueError, RuntimeError, np.linalg.LinAlgError):
                        continue
                    if bool(cell_support(reduced, posterior.geometry, posterior.min_height)):
                        states[o, e, w] = (
                            reduced - posterior.geometry.minimum
                        ) / posterior.geometry.span
                        break
                else:
                    failures.append(
                        {
                            "orientation_index": orientation_id,
                            "ensemble": e,
                            "walker": w,
                            "attempts": settings.initialization_attempts,
                        }
                    )
    return states, failures


@torch.no_grad()
def sample_ensemble(
    log_probability: Callable[[Tensor], Tensor],
    initial: Tensor,
    settings: MCMCSettings,
    streams: list,
) -> dict:
    """Two complementary group updates preserve the ensemble product target.

    Donors are distinct ordered draws from the other (frozen) half, within the
    same orientation and ensemble. This makes each conditional proposal symmetric.
    """
    x = initial.clone()
    logp = log_probability(x)
    if not torch.isfinite(logp).all():
        raise ValueError("All initial walkers must have finite log probability")
    o_count, e_count, walkers, dimension = x.shape
    if (e_count, walkers) != (settings.ensembles, settings.walkers):
        raise ValueError("Initial state does not match ensemble/walker settings")
    half = walkers // 2
    retained = settings.production // settings.thin
    draws = np.empty((o_count, e_count, retained, walkers, dimension), dtype=np.float64)
    scores = np.empty((o_count, e_count, retained, walkers), dtype=np.float64)
    accepted = torch.zeros_like(logp)
    for sweep in range(settings.burn_in + settings.production):
        for group in range(2):
            active = slice(group * half, (group + 1) * half)
            complement = x[:, :, (1 - group) * half : (2 - group) * half]
            proposals, uniforms = [], []
            for o in range(o_count):
                p_row, u_row = [], []
                for e in range(e_count):
                    generator = streams[o][e]
                    first = torch.randint(half, (half,), device=x.device, generator=generator)
                    second = torch.randint(half - 1, (half,), device=x.device, generator=generator)
                    second = second + (second >= first)
                    jitter = (
                        torch.randn(
                            (half, dimension), device=x.device, dtype=x.dtype, generator=generator
                        )
                        * settings.jitter
                    )
                    p_row.append(
                        x[o, e, active]
                        + settings.gamma * (complement[o, e, first] - complement[o, e, second])
                        + jitter
                    )
                    u_row.append(
                        torch.rand(half, device=x.device, dtype=x.dtype, generator=generator)
                    )
                proposals.append(torch.stack(p_row))
                uniforms.append(torch.stack(u_row))
            proposed = torch.stack(proposals)
            proposed_logp = log_probability(proposed)
            take = torch.stack(uniforms).log() < proposed_logp - logp[:, :, active]
            x[:, :, active] = torch.where(take[..., None], proposed, x[:, :, active])
            logp[:, :, active] = torch.where(take, proposed_logp, logp[:, :, active])
            if sweep >= settings.burn_in:
                accepted[:, :, active] += take
        production_sweep = sweep - settings.burn_in + 1
        if production_sweep > 0 and production_sweep % settings.thin == 0:
            index = production_sweep // settings.thin - 1
            draws[:, :, index] = x.cpu().numpy()
            scores[:, :, index] = logp.cpu().numpy()
    return {
        "samples": draws,
        "log_likelihood": scores,
        "acceptance_rate": (accepted / settings.production).cpu().numpy(),
    }


def _split_rhat(chains: np.ndarray) -> float:
    n = chains.shape[1]
    within = np.var(chains, axis=1, ddof=1).mean()
    between = n * np.var(chains.mean(1), ddof=1)
    if within <= 0:
        return float("inf")
    return float(np.sqrt(((n - 1) * within / n + between / n) / within))


def _ess(chains: np.ndarray) -> float:
    """FFT autocovariances and Geyer initial positive/monotone paired sequence."""
    m, n = chains.shape
    centered = chains - chains.mean(1, keepdims=True)
    fft_size = 1 << (2 * n - 1).bit_length()
    transform = np.fft.rfft(centered, n=fft_size, axis=1)
    acov = np.fft.irfft(transform * transform.conj(), n=fft_size, axis=1)[:, :n] / n
    within = acov[:, 0].mean() * n / (n - 1)
    variance = (n - 1) / n * within + np.var(chains.mean(1), ddof=1)
    if variance <= 0 or within <= 0:
        return 0.0
    rho = 1 - (within - acov.mean(0)) / variance
    rho[0] = 1
    pairs = rho[: n // 2 * 2].reshape(-1, 2).sum(1)
    nonpositive = np.flatnonzero(pairs <= 0)
    if len(nonpositive):
        pairs = pairs[: nonpositive[0]]
    pairs = np.minimum.accumulate(pairs)
    tau = max(1.0, -1 + 2 * pairs.sum())
    return float(min(m * n, m * n / tau))


def diagnostics(samples: np.ndarray) -> dict:
    """[ensemble,draw,walker,parameter]; corresponding walkers across ensembles.

    Walkers within an ensemble interact and are never concatenated into chains.
    Return conservative extremes over corresponding-walker diagnostic sets.
    """
    from scipy.special import ndtri
    from scipy.stats import rankdata

    ensembles, draws, walkers, parameters = samples.shape
    rhat = np.full((walkers, parameters), np.inf)
    bulk = np.zeros_like(rhat)
    tail = np.zeros_like(rhat)
    if ensembles >= 2 and draws >= 8 and np.isfinite(samples).all():
        half = draws // 2
        split = np.concatenate((samples[:, :half], samples[:, -half:]), axis=0)
        for w in range(walkers):
            for p in range(parameters):
                chains = split[:, :, w, p]

                def rank_normalize(a):
                    ranks = rankdata(a.reshape(-1)).reshape(a.shape)
                    return ndtri((ranks - 0.375) / (a.size + 0.25))

                z = rank_normalize(chains)
                folded = rank_normalize(abs(chains - np.median(chains)))
                rhat[w, p] = max(_split_rhat(z), _split_rhat(folded))
                bulk[w, p] = _ess(z)
                lo, hi = np.quantile(chains, [0.05, 0.95])
                tail[w, p] = min(
                    _ess((chains <= lo).astype(float)), _ess((chains <= hi).astype(float))
                )
    converged = bool((rhat < 1.01).all() and (bulk >= 400).all() and (tail >= 400).all())
    return {
        "rhat": np.where(np.isfinite(rhat), rhat, None).tolist(),
        "ess_bulk": bulk.tolist(),
        "ess_tail": tail.tolist(),
        "max_rhat": float(rhat.max()) if np.isfinite(rhat).all() else None,
        "min_ess_bulk": float(bulk.min()),
        "min_ess_tail": float(tail.min()),
        "is_converged": converged,
        "method": "corresponding walkers across independent ensembles; split rank/folded Rhat",
    }


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def run_mcmc(
    initial_cells,
    observed,
    orientations,
    geometry: Geometry,
    *,
    q_sigma: float,
    q_scale: float = 1.0,
    min_height: float = 1.5,
    settings: MCMCSettings | None = None,
    output_path: str | Path | None = None,
) -> dict:
    settings = settings or MCMCSettings()
    sigma = validate_sigma(q_sigma, q_scale)
    device = geometry.minimum.device
    initial = torch.as_tensor(initial_cells, device=device, dtype=torch.float64)
    orientations = torch.as_tensor(orientations, device=device, dtype=torch.int64)
    observed = torch.as_tensor(observed, device=device, dtype=torch.float64)
    if initial.ndim != 2 or initial.shape[1] != 6 or orientations.shape != (len(initial), 3):
        raise ValueError("Expected initial_cells [orientation,6] and orientations [orientation,3]")
    if len(initial) == 0 or (orientations == 0).all(-1).any():
        raise ValueError("At least one nonzero orientation is required")
    if not torch.isfinite(initial).all():
        raise ValueError("Initial cell predictions must be finite")
    count, retained = len(initial), settings.production // settings.thin
    samples = np.full((count, settings.ensembles, retained, settings.walkers, 6), np.nan)
    scores = np.full(samples.shape[:-1], -np.inf)
    acceptance = np.zeros((count, settings.ensembles, settings.walkers))
    failures, candidates = [], []
    _sync(device)
    started = time.perf_counter()
    initialization_seconds = sampling_seconds = 0.0
    for start in range(0, count, settings.orientation_chunk_size):
        ids = list(range(start, min(count, start + settings.orientation_chunk_size)))
        posterior = PeakPosterior(
            observed, orientations[ids], geometry, sigma, min_height, settings.reflection_chunk_size
        )
        t = time.perf_counter()
        state, failed = initialize(initial[ids], posterior, settings, ids)
        failures.extend(failed)
        finite = torch.isfinite(posterior(state)).all(dim=(1, 2)).cpu().numpy()
        for local, ok in enumerate(finite):
            if not ok and not any(f["orientation_index"] == ids[local] for f in failed):
                failures.append(
                    {"orientation_index": ids[local], "reason": "no finite initial likelihood"}
                )
        good = [ids[i] for i, ok in enumerate(finite) if ok]
        _sync(device)
        initialization_seconds += time.perf_counter() - t
        if not good:
            continue
        posterior = PeakPosterior(
            observed,
            orientations[good],
            geometry,
            sigma,
            min_height,
            settings.reflection_chunk_size,
        )
        t = time.perf_counter()
        sampled = sample_ensemble(
            posterior,
            state[torch.as_tensor(finite, device=device)],
            settings,
            make_streams(settings, good, device),
        )
        _sync(device)
        sampling_seconds += time.perf_counter() - t
        physical = sampled["samples"] * geometry.span.cpu().numpy() + geometry.minimum.cpu().numpy()
        physical[..., :3] /= q_scale
        samples[good], scores[good] = physical, sampled["log_likelihood"]
        acceptance[good] = sampled["acceptance_rate"]
    # Convert likelihood density from scaled q coordinates to physical q coordinates.
    eligible_count = int(peak_mask(observed, geometry, simulated=True).sum())
    scores -= 2 * eligible_count * math.log(q_scale)
    for index in range(count):
        failure = [f for f in failures if f["orientation_index"] == index]
        candidate = {
            "orientation_index": index,
            "hkl": orientations[index].cpu().tolist(),
            "initial_cell_deg": (
                initial[index].cpu().numpy() / np.array([q_scale] * 3 + [1.0] * 3)
            ).tolist(),
            "status": "initialization_failed" if failure else "unconverged",
            "initialization_failures": failure,
        }
        if not failure:
            flat = samples[index].reshape(-1, 6)
            diag = diagnostics(samples[index])
            candidate.update(
                refined_cell_deg=flat.mean(0).tolist(),
                posterior_std=flat.std(0, ddof=1).tolist(),
                posterior_covariance=np.cov(flat, rowvar=False).tolist(),
                credible_interval_95=np.quantile(flat, [0.025, 0.975], axis=0).T.tolist(),
                goodness_of_fit_log_likelihood=float(scores[index].max()),
                acceptance_rate=acceptance[index].tolist(),
                diagnostics=diag,
                is_converged=diag["is_converged"],
                status="converged" if diag["is_converged"] else "unconverged",
            )
        else:
            candidate["is_converged"] = False
        candidates.append(candidate)
    ranked = sorted(
        (c for c in candidates if "goodness_of_fit_log_likelihood" in c),
        key=lambda c: c["goodness_of_fit_log_likelihood"],
        reverse=True,
    )
    for rank, candidate in enumerate(ranked, 1):
        candidate["goodness_of_fit_rank"] = rank
    elapsed = time.perf_counter() - started
    result = {
        "algorithm": "mcmc",
        "settings": asdict(settings),
        "q_sigma_angstrom_inverse": q_sigma,
        "q_scale": q_scale,
        "device": str(device),
        "candidates": candidates,
        "initialization_failures": failures,
        "timings": {
            "total_seconds": elapsed,
            "initialization_seconds": initialization_seconds,
            "sampling_seconds": sampling_seconds,
        },
        "cuda_validated": False,
        "score_interpretation": "maximum retained Gaussian residual log likelihood; not evidence",
    }
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        import json

        np.savez_compressed(
            output_path,
            samples=samples,
            log_likelihood=scores,
            orientations=orientations.cpu().numpy(),
            acceptance_rate=acceptance,
            settings_json=json.dumps(asdict(settings)),
            q_scale=q_scale,
            q_sigma=q_sigma,
            sample_axes=np.array(["orientation", "ensemble", "draw", "walker", "parameter"]),
            parameter_names=np.array(["a", "b", "c", "alpha", "beta", "gamma"]),
            parameter_units=np.array(["angstrom"] * 3 + ["degree"] * 3),
            metadata_json=json.dumps(result, allow_nan=False),
        )
        result["samples_path"] = str(output_path.resolve())
    return result
