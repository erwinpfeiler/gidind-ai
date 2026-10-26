"""Native correction/reduction, followed by explicit integer-plane adaptation."""
import time
import numpy as np
from .geometry import (assign, direct_metric, first_order, hkl_grid, metric_to_cell,
                       observed3, primitive_planes, reciprocal_metric, reduce_basis,
                       within_bounds)
from .state import SearchLimit


def _native_finish(candidate, coords, grid, nospec, check):
    metric, h = candidate.metric, candidate.hkl
    cell = metric_to_cell(metric)
    native = dict(candidate.native_scores)
    if nospec:
        t = candidate.projection
        h, _, _ = assign(metric, coords, grid, projection=t, channels=(2,1))
        cell, corrected1 = first_order(cell, h, coords, channels=(2,))
        metric = reciprocal_metric(cell)
        t = np.linalg.lstsq(h, coords[:, 1], rcond=None)[0]
        continuous = np.linalg.solve(metric, t)
        h, _, _ = assign(metric, coords, grid, projection=t, channels=(2,1))
        cell, corrected2 = first_order(cell, h, coords, continuous=continuous, channels=(2,1))
        metric = reciprocal_metric(cell)
        t = np.linalg.lstsq(h, coords[:, 1], rcond=None)[0]
        metric, h, _, transform = reduce_basis(metric, h, projection=t, check=check)
        native['first_order_steps_applied'] = int(corrected1)+int(corrected2)
    else:
        plane = candidate.plane
        h, _, _ = assign(metric, coords, grid, plane=plane)
        cell, corrected = first_order(cell, h, coords, plane=plane)
        metric = reciprocal_metric(cell)
        h, _, _ = assign(metric, coords, grid, plane=plane)
        metric, h, _, transform = reduce_basis(metric, h, plane=plane, check=check)
        native['first_order_steps_applied'] = int(corrected)
    return metric, h, transform, native


def finalize(peaks, mode, settings, state):
    state.set_stage("correction_reduction_discrete_projection")
    planes = primitive_planes(settings.plane_limit, settings.planes)
    nospec = mode == "without-specular"
    limits = tuple(settings.without_specular_index_hkl) if nospec else (settings.specular_index_hk,)*2+(settings.specular_l,)
    grid = hkl_grid(limits)
    raw = state.pending
    corrected_window = nospec and settings.algorithm == 'corrected'
    ceiling = None
    if state.disk_candidates:
        if nospec and raw and not corrected_window:
            ceiling = max(raw.minimum()*(1+settings.without_specular_relative_window), 1e-12)
        # Corrected finalization historically sorted the pending list itself.
        # Preserve that order in small compatibility snapshots as well.
        raw.final_order = corrected_window
        raw = raw.view(ceiling)
    elif nospec and raw:
        best = min(c.native_scores["seed_linear_sumerr"] for c in raw)
        # Include exact synthetic zero scores; upstream strict < Min*1.2 loses them.
        if not corrected_window:
            ceiling = max(best*(1+settings.without_specular_relative_window), 1e-12)
            raw = [c for c in raw if c.native_scores["seed_linear_sumerr"] <= ceiling]
        raw.sort(key=lambda c: c.native_scores["seed_linear_sumerr"])
    elif raw:
        raw = sorted(raw, key=lambda c: c.native_scores["seed_qxyz_rmse"])
    state.counts["candidates_selected_for_finalization"] = len(raw)
    seen = {}
    target = observed3(peaks.coordinates)
    try:
        for candidate_index,candidate in enumerate(raw):
            state.check()
            if (corrected_window and ceiling is not None and
                    candidate.native_scores['seed_linear_sumerr'] > ceiling):
                state.counts['corrected_discrete_window_skipped'] += len(raw)-candidate_index
                break
            try:
                metric, h, transform, native = _native_finish(candidate, peaks.coordinates, grid, nospec, state.check)
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                state.counts['invalid_correction_or_reduction'] += 1
                continue
            state.counts["reduced_candidates"] += 1
            cell = metric_to_cell(metric)
            if not within_bounds(cell, settings, no_specular=nospec):
                state.counts["final_cell_bounds"] += 1; continue
            lengths = cell[:3].tolist()
            angles = np.rad2deg(cell[3:]).tolist()
            vol = float(np.sqrt(np.linalg.det(direct_metric(cell))))
            # Reduction changes Laue coordinates; include transformed native H as
            # candidates even if its components exceed the original search grid.
            expanded = np.unique(np.vstack([grid, h, -h]), axis=0)
            for plane in planes:
                state.check()
                order, specerror = None, None
                if not nospec:
                    qspec = peaks.provenance["selected_specular_q"]
                    fundamental = float(np.sqrt(plane@metric@plane))
                    orders = np.arange(1, max(settings.specular_uv, settings.specular_w)+1)
                    errors = np.abs(orders*fundamental-qspec)
                    take = int(np.argmin(errors))
                    if errors[take] > settings.specular_cutoff:
                        continue
                    order, specerror = int(orders[take]), float(errors[take])
                final_h, predicted, combined = assign(metric, peaks.coordinates, expanded, plane=plane)
                diff = predicted-target
                rmse = np.sqrt(np.mean(diff*diff, axis=0))
                if np.any(rmse > settings.rmse_cutoffs):
                    state.counts["discrete_residual_cutoff"] += 1; continue
                if corrected_window and ceiling is None:
                    # The best continuous orientation may have no allowed
                    # integer plane. Anchor the relative score window only
                    # after an integer-compatible candidate actually passes.
                    ceiling = max(candidate.native_scores['seed_linear_sumerr']*
                        (1+settings.without_specular_relative_window),1e-12)
                    state.counts['corrected_discrete_window_anchored'] += 1
                key = (tuple(plane), order, final_h.astype('<i4').tobytes())
                rank = (float(rmse[2]), float(rmse[1]), float(rmse[0]))
                if key in seen:
                    state.counts["duplicate_final_solution"] += 1
                    old = state.solutions[seen[key]]['rmse']
                    if rank >= (old['q_xyz'], old['q_z'], old['q_xy']):
                        continue
                assignments = []
                for i, row in enumerate(peaks.source_rows):
                    assignments.append({"source_row": int(row), "h": int(final_h[i,0]),
                                        "k": int(final_h[i,1]), "l": int(final_h[i,2]),
                                        "observed_q_xy": float(target[i,0]), "observed_q_z": float(target[i,1]),
                                        "observed_q_xyz": float(target[i,2]),
                                        "predicted_q_xy": float(predicted[i,0]), "predicted_q_z": float(predicted[i,1]),
                                        "predicted_q_xyz": float(predicted[i,2]),
                                        "residual_q_xy": float(diff[i,0]), "residual_q_z": float(diff[i,1]),
                                        "residual_q_xyz": float(diff[i,2])})
                assignments.sort(key=lambda a: a['source_row'])
                solution = {"cell": dict(zip(("a","b","c","alpha","beta","gamma"), lengths+angles)),
                                        "cell_units": "angstrom,degrees", "volume_angstrom3": vol,
                                        "contact_plane": plane.tolist(), "specular_reflection_order": order,
                                        "specular_error": specerror, "rmse": dict(zip(("q_xy","q_z","q_xyz"), rmse.tolist())),
                                        "max_combined_peak_error": float(combined.max()),
                                        "native_scores": native,
                                        "basis_transform": transform.tolist(),
                                        "parameter_uncertainty": None,
                                        "uncertainty_note": "Upstream GIDInd supplies residuals, not parameter uncertainties.",
                                        "assignments": assignments}
                if key in seen:
                    state.solutions[seen[key]] = solution
                    state.counts['improved_duplicate_fit'] += 1
                    state.record_solution(candidate,seen[key],solution,'replaced')
                else:
                    seen[key] = len(state.solutions)
                    state.solutions.append(solution)
                    if state.first_solution_wall_seconds is None:
                        state.first_solution_wall_seconds = time.perf_counter()-state.started
                    state.counts["accepted_solutions"] += 1
                    state.record_solution(candidate,seen[key],solution,'accepted')
                if len(state.solutions) >= settings.max_solutions:
                    state.partial("solution_limit")
                    return
            state.counts["finalized_native_candidates"] += 1
    except SearchLimit as exc:
        state.partial(str(exc))
    finally:
        state.solutions.sort(key=lambda s: (s['rmse']['q_xyz'], s['rmse']['q_z'], s['rmse']['q_xy'],
                                             tuple(s['contact_plane']), tuple(s['cell'].values())))
