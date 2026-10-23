"""Independent float64 orientation fits, in one process and on one Torch device.

Adapted from the padded-residual and LM structure in Scattailab's legacy
GIWAXS refinement. Geometry, parameter normalization and matching semantics
follow gidind-ai's original polishing_fit implementation, not its codec.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.func import jacfwd, jacrev, vmap

from .uncertainty import diagnostic_row, guard_uncertainties

# Remains false until the CUDA validation runner has established parity/speed.
CUDA_BATCHED_VALIDATED = False


def resolve_backend(requested: str, device: Any) -> str:
    if requested not in {"auto", "serial", "batched"}:
        raise ValueError("refinement_backend must be auto, serial or batched")
    if requested == "auto":
        return (
            "batched"
            if torch.device(device).type == "cuda" and CUDA_BATCHED_VALIDATED
            else "serial"
        )
    return requested


def safe_process_count(device: Any, workers: int | None, *, supported: bool = True) -> int:
    """CUDA tensors must never enter a process pool, including the serial backend."""
    count = 1 if workers is None else int(workers)
    if count < 1:
        raise ValueError("num_processes must be positive")
    if torch.device(device).type == "cuda":
        if not supported and count > 1:
            raise ValueError(
                "CUDA multiprocessing is unsupported for partial/multiphase refinement"
            )
        return 1
    return count


@dataclass(frozen=True)
class Geometry:
    minimum: Tensor
    span: Tensor
    laue: Tensor
    q_xy_max: float
    q_z_max: float
    use_specular: bool
    specular_cutoff: float

    @classmethod
    def from_generator(cls, generator: Any, device: Any) -> Geometry:
        tc, fc = generator.target_config, generator.feature_config
        minimum = torch.tensor(
            [tc.min_distance] * 3 + [tc.min_angle] * 3, dtype=torch.float64, device=device
        )
        maximum = torch.tensor(
            [tc.max_distance] * 3 + [tc.max_angle] * 3, dtype=torch.float64, device=device
        )
        return cls(
            minimum,
            maximum - minimum,
            generator.laue_indices_tensor.to(device=device, dtype=torch.float64),
            float(fc.max_q_xy),
            float(fc.max_q_z),
            bool(fc.use_specular_scan),
            float(fc.specular_cutoff),
        )


def valid_cells(cells: Tensor) -> Tensor:
    angles = torch.deg2rad(cells[..., 3:])
    ca, cb, cg = angles.cos().unbind(-1)
    gram = 1 + 2 * ca * cb * cg - ca.square() - cb.square() - cg.square()
    return (
        torch.isfinite(cells).all(-1)
        & (cells[..., :3] > 0).all(-1)
        & (cells[..., 3:] > 0).all(-1)
        & (cells[..., 3:] < 180).all(-1)
        & (gram > 1e-14)
    )


def reciprocal_vectors(cells: Tensor, orientations: Tensor) -> Tensor:
    """Same cell/rotation/reciprocal equations as GixdDataGenerator, batch-safe."""
    a, b, c = cells[..., :3].unbind(-1)
    alpha, beta, gamma = torch.deg2rad(cells[..., 3:]).unbind(-1)
    zero = torch.zeros_like(a)
    v1 = torch.stack((a, zero, zero), -1)
    v2 = torch.stack((b * gamma.cos(), b * gamma.sin(), zero), -1)
    v3y = c * (alpha.cos() - beta.cos() * gamma.cos()) / gamma.sin()
    v3 = torch.stack(
        (c * beta.cos(), v3y, (c.square() - (c * beta.cos()).square() - v3y.square()).sqrt()), -1
    )
    h, k, l = orientations.to(cells).unbind(-1)
    h, k, l = h[..., None], k[..., None], l[..., None]
    hk = (h == 0) & (k == 0)
    hl = (h == 0) & (l == 0)
    kl = (k == 0) & (l == 0)
    hz = (h == 0) & (k != 0) & (l != 0)
    kz = (k == 0) & (h != 0) & (l != 0)
    lz = (l == 0) & (h != 0) & (k != 0)
    nz = (h != 0) & (k != 0) & (l != 0)
    eps = 1e-16
    x1 = (
        torch.where(hk | hl | hz, v1, 0)
        + torch.where(kl | kz, v2, 0)
        + torch.where(lz, v3, 0)
        + torch.where(nz, v1 / (h + eps) - v2 / (k + eps), 0)
    )
    x2 = (
        torch.where(hk, v2, 0)
        + torch.where(hl | kl, v3, 0)
        + torch.where(hz, v2 / (k + eps) - v3 / (l + eps), 0)
        + torch.where(kz | nz, v1 / (h + eps) - v3 / (l + eps), 0)
        + torch.where(lz, v1 / (h + eps) - v2 / (k + eps), 0)
    )
    identity = ((h == 0) & (k == 0) & ((l == 1) | (l == 2))).squeeze(-1)
    normal = torch.linalg.cross(x1, x2)
    normal = normal / normal.square().sum(-1, keepdim=True).clamp_min(1e-32).sqrt()
    old = torch.tensor([0.0, 0.0, 1.0], dtype=cells.dtype, device=cells.device).expand_as(normal)
    axis = torch.linalg.cross(old, normal)
    # Avoid undefined derivatives in inactive identity branches and at exact poles.
    axis = torch.where(identity[..., None], torch.ones_like(axis), axis)
    sine = axis.square().sum(-1).clamp_min(1e-32).sqrt()
    axis = axis / sine[..., None]
    theta = -torch.atan2(sine, normal[..., 2])
    ct, st = theta.cos(), theta.sin()
    ux, uy, uz = axis.unbind(-1)
    rotation = torch.stack(
        (
            ct + ux * ux * (1 - ct),
            ux * uy * (1 - ct) - uz * st,
            ux * uz * (1 - ct) + uy * st,
            uy * ux * (1 - ct) + uz * st,
            ct + uy * uy * (1 - ct),
            uy * uz * (1 - ct) - ux * st,
            uz * ux * (1 - ct) - uy * st,
            uz * uy * (1 - ct) + ux * st,
            ct + uz * uz * (1 - ct),
        ),
        -1,
    )
    rotation = rotation.reshape(*cells.shape[:-1], 3, 3)
    eye = torch.eye(3, dtype=cells.dtype, device=cells.device)
    rotation = torch.where(identity[..., None, None], eye, rotation)
    v1, v2, v3 = [(rotation @ v[..., None]).squeeze(-1) for v in (v1, v2, v3)]
    volume = (torch.linalg.cross(v1, v2) * v3).sum(-1, keepdim=True)
    return torch.stack(
        (
            torch.linalg.cross(v2, v3) / volume,
            torch.linalg.cross(v3, v1) / volume,
            torch.linalg.cross(v1, v2) / volume,
        ),
        -1,
    )


def simulate(cells: Tensor, orientations: Tensor, hkl: Tensor) -> Tensor:
    reciprocal = reciprocal_vectors(cells, orientations)
    q = 2 * torch.pi * (hkl @ reciprocal.transpose(-1, -2))
    xy = q[..., :2].square().sum(-1).clamp_min(1e-32).sqrt()
    return torch.stack((xy, q[..., 2].abs()), -1)


def peak_mask(points: Tensor, geometry: Geometry, *, simulated: bool) -> Tensor:
    x, z = points.unbind(-1)
    if geometry.use_specular:
        keep = (x > 0) | (z > 0)
    else:
        keep = x > geometry.specular_cutoff
    if simulated:
        keep = keep & (x < geometry.q_xy_max) & (z < geometry.q_z_max) & (z >= 0)
    return keep & torch.isfinite(points).all(-1)


def hungarian(cost: Tensor) -> tuple[Tensor, Tensor]:
    """Exact batched rectangular assignment, with device-resident augmenting paths.

    No per-candidate or per-column Python loop and no per-element scalar reads.
    One batch-wide termination check per augmenting-path step is required.
    """
    batch, rows, cols = cost.shape
    transposed = rows > cols
    if transposed:
        cost = cost.transpose(1, 2)
        rows, cols = cols, rows
    device = cost.device
    if not rows:
        empty = torch.empty((batch, 0), dtype=torch.long, device=device)
        return empty, empty
    u = cost.new_zeros((batch, rows + 1))
    v = cost.new_zeros((batch, cols + 1))
    p = torch.zeros((batch, cols + 1), dtype=torch.long, device=device)
    way = torch.zeros_like(p)
    bid = torch.arange(batch, device=device)
    col = torch.arange(cols + 1, device=device)[None, :]
    for i in range(1, rows + 1):
        p[:, 0] = i
        minv = cost.new_full((batch, cols + 1), float("inf"))
        used = torch.zeros_like(p, dtype=torch.bool)
        j0 = torch.zeros(batch, dtype=torch.long, device=device)
        active = torch.ones(batch, dtype=torch.bool, device=device)
        while bool(active.any()):
            used[bid, j0] |= active
            i0 = p[bid, j0]
            cur = cost[bid, (i0 - 1).clamp_min(0)] - u[bid, i0, None] - v[:, 1:]
            improve = active[:, None] & ~used[:, 1:] & (cur < minv[:, 1:])
            minv[:, 1:] = torch.where(improve, cur, minv[:, 1:])
            way[:, 1:] = torch.where(improve, j0[:, None], way[:, 1:])
            choices = minv.masked_fill(used | (col == 0), float("inf"))
            delta, j1 = choices.min(-1)
            delta = torch.where(active, delta, 0)
            updates = used.to(cost.dtype) * delta[:, None]
            u.scatter_add_(1, p, updates)
            v -= updates
            minv = torch.where(used, minv, minv - delta[:, None])
            j0 = torch.where(active, j1, j0)
            active &= p[bid, j0] != 0
        active = j0 != 0
        while bool(active.any()):
            j1 = way[bid, j0]
            p[bid, j0] = torch.where(active, p[bid, j1], p[bid, j0])
            j0 = torch.where(active, j1, j0)
            active = j0 != 0
    row_to_col = torch.zeros((batch, rows + 1), dtype=torch.long, device=device)
    row_to_col.scatter_(1, p[:, 1:], col[:, 1:].expand(batch, -1))
    ri = torch.arange(rows, device=device).expand(batch, -1)
    ci = row_to_col[:, 1:] - 1
    return (ci, ri) if transposed else (ri, ci)


@dataclass
class MatchPack:
    hkl: Tensor
    observed: Tensor
    weights: Tensor
    mask: Tensor
    observed_indices: Tensor
    simulated_indices: Tensor
    simulated: Tensor
    simulated_mask: Tensor


def match(
    cells: Tensor,
    orientations: Tensor,
    observed: Tensor,
    geometry: Geometry,
    cutoff: float,
    method: str,
) -> MatchPack:
    with torch.no_grad():
        sim = simulate(cells, orientations, geometry.laue)
        sm = peak_mask(sim, geometry, simulated=True)
        om = peak_mask(observed, geometry, simulated=False)
        batch, n_sim = sm.shape
        n_obs = observed.shape[1]
        if method not in {"observed", "simulated", "simulated_fractional", "one2one"}:
            raise ValueError(f"Unsupported batched matching method: {method}")
        cost = torch.linalg.vector_norm(observed[:, :, None] - sim[:, None], dim=-1)
        valid = om[:, :, None] & sm[:, None]
        # A constant dummy cost larger than the sum of all valid costs ensures
        # padded assignment is equivalent to each unpadded rectangular problem.
        maximum = (
            torch.where(valid, cost, 0).amax((1, 2), keepdim=True)
            if n_obs and n_sim
            else cells.new_zeros((batch, 1, 1))
        )
        penalty = (maximum + 1) * (min(n_obs, n_sim) + 1)
        cost = torch.where(valid, cost, penalty)
        bid = torch.arange(batch, device=cells.device)[:, None]
        if not n_obs or not n_sim:
            oi = torch.empty((batch, 0), dtype=torch.long, device=cells.device)
            si = oi
        elif method == "one2one":
            oi, si = hungarian(cost)
        elif method == "observed":
            si = cost.argmin(-1)
            oi = torch.arange(n_obs, device=cells.device).expand(batch, -1)
        else:
            oi = cost.argmin(1)
            si = torch.arange(n_sim, device=cells.device).expand(batch, -1)
        keep = valid[bid, oi, si] & (cost[bid, oi, si] <= cutoff)
        weights = keep.to(cells.dtype)
        if method == "simulated_fractional" and n_obs:
            counts = cells.new_zeros((batch, n_obs)).scatter_add_(1, oi, weights)
            weights = weights / counts.gather(1, oi).clamp_min(1)
        hkls = geometry.laue[si]
        points = torch.where(keep[..., None], observed[bid, oi], 0)
        # Invalid padded rows must remain differentiable without sqrt(0) Hessians.
        hkls = torch.where(keep[..., None], hkls, cells.new_tensor([1.0, 0.0, 0.0]))
        return MatchPack(hkls, points, weights, keep, oi, si, sim, sm)


def residual_row(
    x: Tensor,
    orientation: Tensor,
    hkl: Tensor,
    observed: Tensor,
    weights: Tensor,
    minimum: Tensor,
    span: Tensor,
) -> Tensor:
    predicted = simulate(x * span + minimum, orientation, hkl)
    scale = (weights / weights.sum().clamp_min(1e-6)).sqrt()
    delta = (predicted - observed) * scale[:, None]
    return torch.cat((delta[:, 0], delta[:, 1]))


def residuals(x: Tensor, orientations: Tensor, pack: MatchPack, geometry: Geometry) -> Tensor:
    return vmap(residual_row, in_dims=(0, 0, 0, 0, 0, None, None))(
        x, orientations, pack.hkl, pack.observed, pack.weights, geometry.minimum, geometry.span
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def refine_batch(
    initial_cells: Tensor,
    observed: Tensor,
    orientations: Tensor,
    geometry: Geometry,
    *,
    matching_method: str = "one2one",
    q_cutoff: float = 0.1,
    outer_loops: int = 3,
    maxiter: int = 50,
    return_uncertainties: bool = True,
) -> dict[str, Any]:
    """Run independent damped least-squares fits with fixed assignments per round."""
    device = initial_cells.device
    cells = initial_cells.to(dtype=torch.float64)
    if cells.ndim == 1:
        cells = cells[None]
    orientations = torch.as_tensor(orientations, device=device, dtype=torch.long).reshape(-1, 3)
    if orientations.shape[0] != cells.shape[0] or cells.shape[1] != 6:
        raise ValueError("Expected one six-parameter cell per orientation")
    observed = torch.as_tensor(observed, device=device, dtype=torch.float64)
    if observed.ndim == 2:
        observed = observed[None].expand(cells.shape[0], -1, -1)
    if observed.shape[0] != cells.shape[0] or observed.shape[-1] != 2:
        raise ValueError("Observed peaks must have shape [N, 2] or [B, N, 2]")
    if outer_loops < 1 or maxiter < 1 or q_cutoff <= 0:
        raise ValueError("outer_loops, maxiter and q_cutoff must be positive")
    timings = dict.fromkeys(("matching_ms", "optimization_ms", "uncertainty_ms"), 0.0)
    _synchronize(device)
    start = time.perf_counter()
    valid = valid_cells(cells)
    safe = cells.new_tensor([8.0, 9.0, 10.0, 90.0, 100.0, 110.0])
    cells = torch.where(valid[:, None], cells, safe)
    x = (cells - geometry.minimum) / geometry.span
    iterations = torch.zeros(cells.shape[0], device=device, dtype=torch.long)
    history = []
    history_mask = []
    initial_pack = None
    # Six inputs versus potentially thousands of residuals: forward mode avoids
    # a residual-sized reverse-mode basis and keeps a 49-orientation batch small.
    jacobian_fn = vmap(jacfwd(residual_row), in_dims=(0, 0, 0, 0, 0, None, None))
    for outer in range(outer_loops):
        _synchronize(device)
        phase = time.perf_counter()
        method = matching_method
        if method == "hybrid":
            method = "one2one" if outer < outer_loops / 2 else "simulated_fractional"
        pack = match(
            x * geometry.span + geometry.minimum, orientations, observed, geometry, q_cutoff, method
        )
        if initial_pack is None:
            initial_pack = pack
        _synchronize(device)
        timings["matching_ms"] += (time.perf_counter() - phase) * 1000
        phase = time.perf_counter()
        active = valid & pack.mask.any(-1)
        damping = cells.new_full((cells.shape[0],), 1e-3)
        for _ in range(maxiter):
            if not bool(active.any()):
                break
            r = residuals(x, orientations, pack, geometry)
            jac = jacobian_fn(
                x,
                orientations,
                pack.hkl,
                pack.observed,
                pack.weights,
                geometry.minimum,
                geometry.span,
            )
            loss = r.square().sum(-1)
            finite = torch.isfinite(jac).all((1, 2)) & torch.isfinite(r).all(-1)
            active &= finite
            jac = torch.where(finite[:, None, None], jac, 0)
            r = torch.where(finite[:, None], r, 0)
            jtj = jac.transpose(1, 2) @ jac
            gradient = (jac.transpose(1, 2) @ r[..., None]).squeeze(-1)
            diagonal = jtj.diagonal(dim1=-2, dim2=-1).clamp_min(1e-8)
            system = jtj + torch.diag_embed(damping[:, None] * diagonal)
            delta, info = torch.linalg.solve_ex(system, -gradient[..., None], check_errors=False)
            delta = delta.squeeze(-1)
            trial = x + delta
            trial_valid = valid_cells(trial * geometry.span + geometry.minimum)
            usable = active & trial_valid & (info == 0) & torch.isfinite(delta).all(-1)
            candidate = torch.where(usable[:, None], trial, x)
            next_loss = residuals(candidate, orientations, pack, geometry).square().sum(-1)
            accepted = usable & torch.isfinite(next_loss) & (next_loss < loss)
            x = torch.where(accepted[:, None], candidate, x).detach()
            history_mask.append(active.clone())
            iterations += active.long()
            damping = torch.where(accepted, damping / 3, damping * 10).clamp(1e-12, 1e12)
            stopped = (
                (gradient.abs().amax(-1) < 1e-10)
                | (accepted & (delta.abs().amax(-1) < 1e-10))
                | (damping >= 1e12)
            )
            active &= ~stopped
            history.append(
                torch.where(
                    pack.mask.any(-1) & valid, torch.where(accepted, next_loss, loss), float("inf")
                ).detach()
            )
        _synchronize(device)
        timings["optimization_ms"] += (time.perf_counter() - phase) * 1000
    phase = time.perf_counter()
    # Preserve the original hybrid final-assignment convention.
    final = match(
        x * geometry.span + geometry.minimum,
        orientations,
        observed,
        geometry,
        q_cutoff,
        "one2one" if matching_method == "hybrid" else matching_method,
    )
    _synchronize(device)
    timings["matching_ms"] += (time.perf_counter() - phase) * 1000
    # Legacy q-loss is the final inner objective, even when hybrid's reporting
    # and covariance reassign with one-to-one matching afterward.
    loss = residuals(x, orientations, pack, geometry).square().sum(-1)
    usable = valid & final.mask.any(-1) & torch.isfinite(loss)
    uncertainty = torch.full_like(x, float("nan"))
    identifiability = None
    phase = time.perf_counter()
    if return_uncertainties and final.hkl.shape[1]:

        def objective(row, orientation, hkl, obs, weights):
            return (
                residual_row(row, orientation, hkl, obs, weights, geometry.minimum, geometry.span)
                .square()
                .sum()
            )

        hessians = vmap(jacrev(jacrev(objective)))(
            x, orientations, final.hkl, final.observed, final.weights
        )
        finite = torch.isfinite(hessians).all((1, 2))
        eye = torch.eye(6, device=device, dtype=x.dtype)
        hessians = torch.where(finite[:, None, None], hessians, eye)
        hessians = 0.5 * (hessians + hessians.transpose(1, 2)) + 1e-6 * eye
        inverse = torch.linalg.pinv(hessians)
        # Original ls_res uses unweighted residual SSE and 2 * matches - 6 DOF.
        raw = (
            simulate(x * geometry.span + geometry.minimum, orientations, final.hkl) - final.observed
        )
        sse = (raw.square().sum(-1) * final.mask).sum(-1)
        sigma2 = sse / (2 * final.mask.sum(-1) - 6).clamp_min(1)
        variance = inverse.diagonal(dim1=-2, dim2=-1) * sigma2[:, None]
        uncertainty = variance.sqrt() * geometry.span
        uncertainty = torch.where((usable & finite)[:, None], uncertainty, float("nan"))
        jacobian = vmap(jacrev(residual_row), in_dims=(0, 0, 0, 0, 0, None, None))(
            x,
            orientations,
            final.hkl,
            final.observed,
            final.weights,
            geometry.minimum,
            geometry.span,
        )
        uncertainty, identifiability = guard_uncertainties(uncertainty, jacobian)
        uncertainty = torch.where((usable & finite)[:, None], uncertainty, float("nan"))
    elif not return_uncertainties:
        uncertainty = torch.where(usable[:, None], torch.zeros_like(x), float("nan"))
    _synchronize(device)
    timings["uncertainty_ms"] += (time.perf_counter() - phase) * 1000
    timings["total_ms"] = (time.perf_counter() - start) * 1000
    refined = x * geometry.span + geometry.minimum
    refined = torch.where(valid[:, None], refined, initial_cells.reshape(-1, 6).to(refined))
    return {
        "cells": refined,
        "uncertainties": uncertainty,
        "identifiability": identifiability,
        "losses": torch.where(usable, loss, float("inf")),
        "valid": usable,
        "iterations": iterations,
        "loss_history": torch.stack(history, 1) if history else x.new_empty((x.shape[0], 0)),
        "loss_history_mask": torch.stack(history_mask, 1)
        if history_mask
        else torch.empty((x.shape[0], 0), dtype=torch.bool, device=device),
        "initial_matches": initial_pack,
        "final_matches": final,
        "timings": timings,
        "backend": "batched",
        "optimizer": "levenberg_marquardt",
    }


def result_rows(result: dict[str, Any], observed: Tensor) -> list[dict[str, Any]]:
    """Convert batched tensors to the legacy reporting schema only after fitting."""
    rows = []
    obs = torch.as_tensor(observed).detach().cpu().numpy()
    for index in range(result["cells"].shape[0]):
        observation = obs[index] if obs.ndim == 3 else obs
        plots = []
        for pack in (result["initial_matches"], result["final_matches"]):
            keep = pack.mask[index]
            matches = torch.cat(
                (
                    pack.hkl[index, keep],
                    pack.observed_indices[index, keep, None].to(pack.hkl.dtype),
                    pack.observed[index, keep],
                    pack.weights[index, keep, None],
                ),
                -1,
            )
            plots.append(
                {
                    "test_qxy": observation[..., 0],
                    "test_qz": observation[..., 1],
                    "sim_q": pack.simulated[index, pack.simulated_mask[index]].cpu().numpy(),
                    "indices_match": matches.cpu().numpy(),
                    "plot_title": "Batched refinement",
                }
            )
        rows.append(
            {
                "refined_predicted_cell_deg": result["cells"][index],
                "refined_uncertainty_cell_deg": result["uncertainties"][index],
                "refined_q_loss": float(result["losses"][index].cpu()),
                "convergence_status": bool(result["valid"][index].cpu()),
                "refined_num_matches": int(result["final_matches"].mask[index].sum().cpu()),
                "refined_num_sim_peaks": int(
                    result["final_matches"].simulated_mask[index].sum().cpu()
                ),
                "init_plot_dict": plots[0],
                "refined_plot_dict": plots[1],
                "refinement_backend": "batched",
                "optimizer": result["optimizer"],
                "iterations": int(result["iterations"][index].cpu()),
                "loss_history": result["loss_history"][index, result["loss_history_mask"][index]]
                .cpu()
                .tolist(),
                "identifiability": diagnostic_row(result["identifiability"], index)
                if result["identifiability"] is not None
                else None,
            }
        )
    return rows
