"""Translation of GIDInd's two-stage with-specular indexing pipeline."""
from itertools import combinations
import time
import numpy as np
from .geometry import (cartesian_batches, hkl_grid, matlab_round, metric_to_cell,
                       predict, reciprocal_metric, within_bounds)
from .state import Candidate, SearchLimit


def unique_tolerance_indices(values, tolerance):
    """Absolute-tolerance grouping used by MATLAB uniquetol (sorted representatives)."""
    order = np.argsort(values, kind="stable")
    labels = np.empty(len(values), dtype=int)
    reps, last, label = [], -np.inf, -1
    for i in order:
        if values[i] - last > tolerance:
            last = values[i]; reps.append(i); label += 1
        labels[i] = label
    return np.asarray(reps, dtype=int), labels


def stage1_matrices(hk, u, v, seed, qspec):
    """function_NEWMATRIXFILLER, vectorized over a bounded HK sextuple batch."""
    h = hk[:, ::2]; k = hk[:, 1::2]
    xy, z = seed.T
    x = h - u*z/qspec; y = k - v*z/qspec
    return np.stack([x*x/xy**2 + u*u/qspec**2,
                     y*y/xy**2 + v*v/qspec**2,
                     -2*(x*y/xy**2 + u*v/qspec**2)], axis=-1)


def abg_from_xyz(xyz, u, v, qspec, settings):
    """CALC_ABGfromXYZ, SUB2_CONDITION and ABGNIGGLI gates."""
    a0, b0, c0 = xyz.T
    with np.errstate(invalid="ignore", divide="ignore"):
        lam = (a0*b0-c0*c0)/(qspec*qspec-u*u*a0-v*v*b0+2*u*v*c0)
        za = np.sqrt(a0+v*v*lam); zb = np.sqrt(b0+u*u*lam)
        cosine = (c0+u*v*lam)/(za*zb)
        gamma = np.arccos(cosine)
        a = 2*np.pi/(za*np.sin(gamma)); b = 2*np.pi/(zb*np.sin(gamma))
        out = np.column_stack([a, b, gamma])
        mask = np.isfinite(out).all(axis=1) & (a0 >= 0) & (b0 >= 0)
        mask &= qspec*qspec-u*u*za*za-v*v*zb*zb+2*u*v*za*zb*cosine > 1e-5
        mask &= a*a <= (b+.1)**2
        mask &= b*np.abs(cosine) <= a/2 + 1e-12
        for values, bounds in ((a, settings.length_bounds[0]), (b, settings.length_bounds[1]),
                               (np.rad2deg(gamma), settings.angle_bounds[2])):
            mask &= (values >= bounds[0]) & (values <= bounds[1])
        mask &= (gamma >= np.pi/3) & (gamma <= 2*np.pi/3)
    return out[mask]


def hk_choices(abg, uv, coords, qspec, settings):
    """MINKLAUEEVALUATOR: four best HK assignments for every observed peak."""
    a, b, gamma = abg; u, v = uv
    za = 2*np.pi/(a*np.sin(gamma)); zb = 2*np.pi/(b*np.sin(gamma))
    hk = hkl_grid((settings.specular_index_hk, settings.specular_index_hk, 0))[:, :2]
    h, k = hk.T
    xy, z = coords.T
    x = h[None, :] - u*z[:, None]/qspec
    y = k[None, :] - v*z[:, None]/qspec
    rad = (za*za*(x*x+(u*xy[:, None]/qspec)**2)
           + zb*zb*(y*y+(v*xy[:, None]/qspec)**2)
           - 2*za*zb*np.cos(gamma)*(x*y+u*v*(xy[:, None]/qspec)**2)
           - ((h*v-k*u)**2/qspec**2)[None, :]*za*za*zb*zb*np.sin(gamma)**2)
    # MATLAB real(sqrt(negative)) is zero. Retain this seed-indexing behavior only.
    pred = np.sqrt(np.maximum(rad, 0))
    errors = (pred-xy[:, None])**2
    take = np.argsort(errors, axis=1, kind="stable")[:, :settings.redundancy_hk]
    best = np.sqrt(np.mean(np.take_along_axis(errors, take[:, :1], axis=1)))
    return hk[take], float(best)


def stage1_batch(data, payload):
    """One native stage-1 HK batch, shared by direct and process execution."""
    hk, u, v, selected = payload
    qspec, settings = data["qspec"], data["settings"]
    mat = stage1_matrices(hk, u, v, selected, qspec)
    good = np.abs(np.linalg.det(mat)) >= 1e-9
    abg = np.empty((0, 3))
    if np.any(good):
        xyz = np.linalg.solve(mat[good], np.ones((int(good.sum()), 3, 1)))[..., 0]
        abg = abg_from_xyz(xyz, u, v, qspec, settings)
    return len(mat), int((~good).sum()), abg


def _stage1(peaks, seed, settings, state, deadline):
    qspec = peaks.provenance["selected_specular_q"]
    rows = state.pending.abgs if state.disk_candidates else []
    # LPermutation UV: u varies fastest.
    uv_values = [(u, v) for v in range(-settings.specular_uv, settings.specular_uv+1)
                 for u in range(-settings.specular_uv, settings.specular_uv+1)]
    for uv in uv_values:
        state.check(deadline)
        u, v = uv
        abgs = []
        try:
            def tasks():
                for triple in combinations(range(len(seed)), 3):
                    selected = seed[list(triple)[::-1]]
                    for _, hk in cartesian_batches(range(-settings.specular_seed_hk,
                                                         settings.specular_seed_hk+1), 6, settings.batch_size):
                        yield hk, u, v, selected
            for count, singular, abg in state.executor.ordered(
                    "specular_stage1", tasks(), lambda: state.check(deadline)):
                state.counts["specular_stage1_systems"] += count
                state.counts["singular_seed_systems"] += singular
                if len(abg):
                    abgs.append(abg)
        except SearchLimit:
            state.partial("specular_stage1_time_limit")
        if abgs:
            all_abg = np.unique(np.vstack(abgs), axis=0)
            # Native three independent uniquetol labels then unique label triples.
            labels = np.column_stack([unique_tolerance_indices(all_abg[:, j], tol)[1]
                                      for j, tol in enumerate((.15, .15, np.deg2rad(.3)))])
            _, indices = np.unique(labels, axis=0, return_index=True)
            for i in sorted(indices):
                try:
                    state.check(deadline)
                except SearchLimit:
                    state.partial("specular_stage1_time_limit")
                    return rows
                choices, error = hk_choices(all_abg[i], uv, peaks.coordinates, qspec, settings)
                if error < settings.qxy_stage_cutoff:
                    rows.append((all_abg[i], uv, choices, error))
                    state.counts["specular_abg_candidates"] += 1
                    if (settings.max_intermediate_candidates is not None and
                            len(rows) >= settings.max_intermediate_candidates):
                        state.partial("abg_candidate_limit")
                        return rows
        if time.perf_counter() >= deadline:
            break
    return rows


def klr_to_cell(abg, klr):
    """function_NEWCELLPARAMETER_RESTRICTED, vectorized."""
    a, b, gamma = abg
    kappa, lam, rho = klr.T
    delta = -kappa*a*np.sin(gamma)/(2*np.pi)
    mu = -lam*b*np.sin(gamma)/(2*np.pi)
    with np.errstate(invalid="ignore", divide="ignore"):
        se = np.sin(gamma)/np.sqrt(np.sin(gamma)**2+delta**2+mu**2+2*delta*mu*np.cos(gamma))
        alpha = np.arccos(se*(mu+delta*np.cos(gamma))/np.sin(gamma))
        beta = np.arccos(se*(delta+mu*np.cos(gamma))/np.sin(gamma))
        c = 2*np.pi/(se*rho)
    return np.column_stack([np.full(len(klr), a), np.full(len(klr), b), c, alpha, beta,
                            np.full(len(klr), gamma)])


def stage2_batch(data, payload):
    """Unchanged numerical operations for one native stage-2 batch."""
    from .parallel import Events
    state = Events()
    settings = data["settings"]
    row, start = payload
    abg, uv, choices, stage_error = row
    coords = data["coordinates"]
    qspec = data["qspec"]
    a, b, gamma = abg; u, v = uv
    za, zb = 2*np.pi/(abg[:2]*np.sin(gamma))
    guv = u*u*za*za+v*v*zb*zb-2*u*v*za*zb*np.cos(gamma)
    sub = np.sqrt(qspec*qspec-guv)
    h, k = choices[..., 0], choices[..., 1]
    sub3 = h*u*za*za+k*v*zb*zb-(h*v+k*u)*za*zb*np.cos(gamma)
    zterm = (coords[:, 1, None]*qspec-sub3)/sub
    reps, _ = unique_tolerance_indices(coords[:, 1], .01)
    if len(reps) < 3:
        return state.events
    reps = reps[:3]
    red = settings.redundancy_hk
    # Native order: HK redundancy triple first-fast, then L triple, then w.
    radices = [red]*3 + [2*settings.specular_l+1]*3 + [2*settings.specular_w+1]
    size = int(np.prod(radices))
    flat = np.arange(start, min(size, start+settings.batch_size))
    digits, stride = [], 1
    for radix in radices:
        digits.append((flat//stride) % radix); stride *= radix
    ix = np.column_stack(digits[:3]); ls = np.column_stack(digits[3:6])-settings.specular_l
    ws = digits[6]-settings.specular_w
    matrices = np.empty((len(flat), 4, 3)); rhs = np.empty((len(flat), 4))
    matrices[:, 0] = np.column_stack([np.full(len(flat), u), np.full(len(flat), v), ws])
    rhs[:, 0] = sub
    for j, rep in enumerate(reps):
        matrices[:, j+1, :2] = choices[rep, ix[:, j]]
        matrices[:, j+1, 2] = ls[:, j]
        rhs[:, j+1] = zterm[rep, ix[:, j]]
    normal = np.einsum('nki,nkj->nij', matrices, matrices)
    good = np.abs(np.linalg.det(normal)) > 1e-9
    state.counts["specular_stage2_systems"] += len(flat)
    state.counts["singular_klr_systems"] += int((~good).sum())
    if not np.any(good):
        return state.events
    matrices, rhs, normal = matrices[good], rhs[good], normal[good]
    klr = np.linalg.solve(normal, np.einsum('nki,nk->ni', matrices, rhs)[..., None])[..., 0]
    cell = klr_to_cell(abg, klr)
    planes = matrices[:, 0].astype(int)
    mask = (np.abs(klr[:, 2]) >= 1e-4) & np.isfinite(cell).all(axis=1)
    mask &= np.abs(np.sqrt(guv+np.sum(planes*klr, axis=1)**2)-qspec) <= settings.specular_cutoff
    for i in np.flatnonzero(mask):
        data["check_cancel"]()
        if not within_bounds(cell[i], settings):
            state.counts["cell_bounds"] += 1; continue
        # Native niggli_on=true gates from PART_TWO.
        aa, bb, cc, al, be, ga = cell[i]
        type1 = np.all(np.rad2deg(cell[i, 3:]) < 91)
        type2 = np.all(np.rad2deg(cell[i, 3:]) > 89)
        if bb*bb > cc*cc or not (type1 or type2):
            continue
        if cc*abs(np.cos(al)) > bb/2 or cc*abs(np.cos(be)) > aa/2:
            continue
        if type2 and not type1 and (bb*cc*abs(np.cos(al))+aa*cc*abs(np.cos(be))+
                                     aa*aa*abs(np.cos(ga)) > (aa*aa+bb*bb)/2):
            continue
        metric = reciprocal_metric(cell[i])
        lv = (zterm-h*klr[i, 0]-k*klr[i, 1])/klr[i, 2]
        lv[np.abs(lv) <= 1e-4] = 0
        allh = np.dstack([choices, matlab_round(lv)])
        predicted = predict(metric, allh.reshape(-1, 3), plane=planes[i]).reshape(len(coords), red, 3)
        errorxyz = (predicted[..., 2]-np.linalg.norm(coords, axis=1)[:, None])**2
        best = np.argmin(errorxyz, axis=1)
        assigned = allh[np.arange(len(coords)), best].astype(int)
        pp = predicted[np.arange(len(coords)), best]
        # Stage-1 qxy residual is the native seed-index score, not physical qxy.
        diff = pp - np.column_stack([coords, np.linalg.norm(coords, axis=1)])
        rms = np.sqrt(np.mean(diff*diff, axis=0))
        if not np.isfinite(rms).all() or np.any(rms[1:] > np.asarray(settings.rmse_cutoffs)[1:]):
            state.counts["native_residual_cutoff"] += 1; continue
        native = {"stage1_qxy_rmse": stage_error, "seed_qz_rmse": float(rms[1]),
                  "seed_qxyz_rmse": float(rms[2]),
                  "seed_specular_error": float(abs(np.sqrt(planes[i]@metric@planes[i])-qspec))}
        state.add_pending(Candidate(metric, assigned, planes[i], None, native))
    return state.events


def _stage2(row, peaks, settings, state):
    reps, _ = unique_tolerance_indices(peaks.coordinates[:, 1], .01)
    if len(reps) < 3:
        state.counts["insufficient_distinct_qz"] += 1
        return
    size = settings.redundancy_hk**3 * (2*settings.specular_l+1)**3 * (2*settings.specular_w+1)
    tasks = ((row, start) for start in range(0, size, settings.batch_size))
    for events in state.executor.ordered("specular_stage2", tasks,
                                          lambda: state.check(state.search_deadline)):
        for kind, value in events:
            state.check(state.search_deadline)
            if kind == "candidate":
                state.add_pending(value)
            else:
                state.counts[kind] += value


def search(peaks, seed, settings, state):
    state.set_stage("specular_abgamma")
    # Reserve time for the second stage and postprocessing in an anytime run.
    first_deadline = state.started + .4*settings.time_limit_seconds
    rows = _stage1(peaks, seed, settings, state, first_deadline)
    if state.disk_candidates:
        state.pending.flush()
        rows = rows.ordered(settings.candidate_order)
    else:
        rows.sort(key=lambda r: (r[0][0]*r[0][1]*np.sin(r[0][2]) if settings.candidate_order == "area" else r[3],
                                 r[3], tuple(r[0]), r[1]))
    state.set_stage("specular_alphabetac")
    try:
        for row in rows:
            _stage2(row, peaks, settings, state)
            state.counts["completed_abg_sweeps"] += 1
    except SearchLimit as exc:
        state.partial(str(exc))
