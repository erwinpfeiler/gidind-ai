"""Triclinic updateplot translation from GIDInd_2021_04_05.mlapp.

Enumerate four seed HKLs, fit qz, choose the fifth/sixth HKLs, solve the six
reciprocal-metric coefficients, index all peaks, and refit. No random search,
neural candidates, or substitute global optimizer is used.
"""
from itertools import combinations
import numpy as np
from .geometry import (assign, cartesian_batches, hkl_grid, metric_design,
                       metric_to_cell, pair_reduced, six_to_metric)
from .state import Candidate, SearchLimit


def seed_systems(h4, seed, hkls, *, corrected=False, tail_candidates=8):
    """Vector form of updateplot's AA4 -> HKL15def/HKL16def -> BB -> qq6."""
    normal = np.einsum('nki,nkj->nij', h4, h4)
    good = np.abs(np.linalg.det(normal)) > .001
    if corrected:
        # Four independent metric equations are necessary: two additional rows
        # cannot complete six unknowns otherwise. Integer HKLs make this rank
        # gate well separated from floating-point uncertainty.
        h,k,l = h4.transpose(2,0,1)
        design4 = np.stack((h*h,k*k,l*l,2*h*k,2*h*l,2*k*l),axis=-1)
        good &= np.linalg.matrix_rank(design4) == 4
    original = np.flatnonzero(good)
    h4, normal = h4[good], normal[good]
    if len(h4) == 0:
        return original, np.empty((0, 6, 3)), np.empty((0, 6), dtype=int), np.empty((0, 3))
    four = np.asarray(list(combinations(range(len(seed)-2), 4)), dtype=int)
    if corrected:
        # An all-in-plane quartet fixes projection=0 and gives no usable
        # orientation for selecting off-plane tail peaks.
        four = four[np.any(seed[four,1] > 0,axis=1)]
        if not len(four):
            return np.empty(0,dtype=int),np.empty((0,6,3)),np.empty((0,6),dtype=int),np.empty((0,3))
    qz4 = seed[four, 1]
    rhs = np.einsum('nki,sk->nis', h4, qz4)
    fits = np.linalg.solve(normal, rhs)
    residual = np.einsum('nki,nis->nks', h4, fits) - qz4.T[None, :, :]
    labels = np.argmin(np.sum(residual*residual, axis=1), axis=1)
    q4 = fits[np.arange(len(h4)), :, labels]
    seed_indices = np.empty((len(h4), 6), dtype=int)
    h6 = np.empty((len(h4), 6, 3), dtype=int)
    h6[:, :4] = h4
    seed_indices[:, :4] = four[labels]
    predz = q4 @ hkls.T
    if corrected:
        width = min(tail_candidates, int(np.count_nonzero(np.any(hkls != 0,axis=1))))
        alternatives5 = np.empty((len(h4),width*width),dtype=int)
        alternatives6 = np.empty_like(alternatives5)
    # Tail pairs depend on which four-peak combination won. Keep native first min.
    for lab in np.unique(labels):
        selected = np.flatnonzero(labels == lab)
        pairs = np.asarray(list(combinations(range(four[lab, -1]+1, len(seed)), 2)), dtype=int)
        error5 = np.abs(predz[selected, :, None] - seed[pairs[:, 0], 1][None, None, :])
        error6 = np.abs(predz[selected, :, None] - seed[pairs[:, 1], 1][None, None, :])
        if corrected:
            zero = np.all(hkls == 0,axis=1)
            error5[:,zero,:] = np.inf
            error6[:,zero,:] = np.inf
        ix5 = np.argmin(error5, axis=1); ix6 = np.argmin(error6, axis=1)
        mins5 = np.min(error5, axis=1); mins6 = np.min(error6, axis=1)
        tail = np.argmin(mins5+mins6, axis=1)
        chosen5 = ix5[np.arange(len(selected)), tail]
        chosen6 = ix6[np.arange(len(selected)), tail]
        h6[selected, 4] = hkls[chosen5]
        h6[selected, 5] = hkls[chosen6]
        seed_indices[selected, 4:] = pairs[tail]
        if corrected:
            # Preserve the original best peak-pair selection, but retain a
            # bounded, stable set of alternatives instead of one greedy HKL.
            e5 = error5[np.arange(len(selected)),:,tail]
            e6 = error6[np.arange(len(selected)),:,tail]
            top5 = np.argsort(e5,axis=1,kind='stable')[:,:width]
            top6 = np.argsort(e6,axis=1,kind='stable')[:,:width]
            a5 = np.repeat(top5,width,axis=1)
            a6 = np.tile(top6,(1,width))
            cost = np.take_along_axis(e5,a5,axis=1)+np.take_along_axis(e6,a6,axis=1)
            order = np.argsort(cost,axis=1,kind='stable')
            alternatives5[selected] = np.take_along_axis(a5,order,axis=1)
            alternatives6[selected] = np.take_along_axis(a6,order,axis=1)
    if corrected:
        beam = width*width
        original = np.repeat(original,beam)
        h6 = np.repeat(h6,beam,axis=0)
        seed_indices = np.repeat(seed_indices,beam,axis=0)
        q4 = np.repeat(q4,beam,axis=0)
        h6[:,4] = hkls[alternatives5.ravel()]
        h6[:,5] = hkls[alternatives6.ravel()]
    return original, h6, seed_indices, q4


def refit(metric, projection, coords, grid, max_error):
    """All-peak indexing and linear refit from the middle of updateplot."""
    h, _, errors = assign(metric, coords, grid, projection=projection)
    if errors.max() >= max_error:
        return None, "max_peak_error"
    design = metric_design(h)
    if abs(np.linalg.det(design.T @ design)) <= .0001:
        return None, "singular_metric_refit"
    if abs(np.linalg.det(h.T @ h)) <= .0001:
        return None, "singular_orientation_refit"
    q2 = np.sum(coords*coords, axis=1)
    coeff = np.linalg.lstsq(design, q2, rcond=None)[0]
    newmetric = six_to_metric(coeff)
    cell = metric_to_cell(newmetric)
    if not pair_reduced(cell):
        return None, "pair_reduction_gate"
    t = np.linalg.lstsq(h, coords[:, 1], rcond=None)[0]
    # sumerr = norm(BBr*qqstar-QXYZ2)/n + norm(H*qqr-QZ)/n in MATLAB.
    score = (np.linalg.norm(design@coeff-q2) + np.linalg.norm(h@t-coords[:, 1]))/len(coords)
    return Candidate(newmetric, h, None, t, {"seed_linear_sumerr": float(score),
                                            "initial_max_peak_error": float(errors.max())}), None


def seed_batch(data, digits):
    """Native seed/refit calculations; global duplicate decisions stay in parent."""
    from collections import Counter
    settings, seed, hkls = data["settings"], data["seed"], data["seed_hkls"]
    peaks_coords, indexing = data["coordinates"], data["indexing"]
    counts = Counter()
    h4 = hkls[digits]
    original, h6, selected, _ = seed_systems(h4, seed, hkls,
        corrected=settings.algorithm == 'corrected',tail_candidates=settings.tail_hkl_candidates)
    counts["without_specular_hkl_quadruples"] += len(h4)
    if settings.algorithm == 'upstream':
        counts["singular_four_hkl"] += len(h4)-len(h6)
    else:
        counts['unusable_four_hkl'] += len(h4)-len(np.unique(original))
        counts['corrected_tail_systems'] += len(h6)
    if len(h6) == 0:
        return counts, []
    if settings.algorithm == 'corrected':
        # Integer products are exactly the same as metric_design row by row;
        # avoid a Python call for each of the expanded beam systems.
        h,k,l = h6.transpose(2,0,1)
        designs = np.stack((h*h,k*k,l*l,2*h*k,2*h*l,2*k*l),axis=-1)
    else:
        designs = np.stack([metric_design(h) for h in h6])
    det = np.linalg.det(designs)
    good = det*det > .0001
    counts["singular_six_hkl"] += int((~good).sum())
    if not np.any(good):
        return counts, []
    designs, h6, selected = designs[good], h6[good], selected[good]
    targets = np.sum(seed[selected]**2, axis=2)
    coeff = np.linalg.solve(designs, targets[..., None])[..., 0]
    metrics = six_to_metric(coeff)
    normal = np.einsum('nki,nkj->nij', h6, h6)
    rhs = np.einsum('nki,nk->ni', h6, seed[selected, 1])
    projections = np.linalg.solve(normal, rhs[..., None])[..., 0]
    good = np.linalg.eigvalsh(metrics)[:, 0] > 1e-12
    counts["nonpositive_seed_metric"] += int((~good).sum())
    outcomes = []
    for i in np.flatnonzero(good):
        data["check_cancel"]()
        key = tuple(np.round(np.r_[coeff[i], projections[i]], 10))
        try:
            candidate, rejection = refit(metrics[i], projections[i], peaks_coords,
                                         indexing, settings.without_specular_max_peak_error)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            candidate, rejection = None, "invalid_refit"
        outcomes.append((key, candidate, rejection))
    return counts, outcomes


def commit_seed_batch(counts, outcomes, raw_seen, state):
    """Apply native global cache and rejection decisions in traversal order."""
    state.counts.update(counts)
    for key, candidate, rejection in outcomes:
        state.check(state.search_deadline)
        if key in raw_seen:
            state.counts["duplicate_seed_geometry"] += 1
            continue
        raw_seen.add(key)
        if len(raw_seen) > 200000:
            raw_seen.clear()
        if candidate is None:
            state.counts[rejection] += 1
        else:
            state.add_pending(candidate)
            state.counts["native_no_specular_candidates"] += 1


def search(peaks, seed, settings, state):
    state.set_stage("without_specular_triclinic_seeds")
    hkls = hkl_grid(tuple(settings.without_specular_seed_hkl), sorted_norm=True)
    raw_seen = set()
    tasks = (digits for _, digits in cartesian_batches(range(len(hkls)), 4,
                                                       settings.batch_size, first_fast=False))
    try:
        for counts, outcomes in state.executor.ordered("without_specular", tasks,
                                                       lambda: state.check(state.search_deadline)):
            commit_seed_batch(counts, outcomes, raw_seen, state)
    except SearchLimit as exc:
        state.partial(str(exc))
