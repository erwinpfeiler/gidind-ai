"""Reciprocal geometry, native first-order corrections and Buerger basis search.

GIDInd conventions: q includes 2*pi, cell angles in radians internally.
Derived from function_BUERGER_CELL_OPTIMIZATION_SORT and updateplot(triclinic).
"""
from functools import lru_cache
from itertools import combinations, product
from math import gcd
import numpy as np
from scipy.spatial.distance import cdist

TAU = 2 * np.pi


def matlab_round(x):
    """MATLAB round: ties away from zero, unlike numpy.rint."""
    x = np.asarray(x)
    return np.copysign(np.floor(np.abs(x) + .5), x)


def cartesian_batches(values, repeat, batch_size, *, first_fast=True):
    """LPermutation traversal without constructing the Cartesian product tensor."""
    values = np.asarray(values)
    radix = len(values)
    for start in range(0, radix ** repeat, batch_size):
        flat = np.arange(start, min(start + batch_size, radix ** repeat), dtype=np.int64)
        digits = np.column_stack([(flat // radix ** i) % radix for i in range(repeat)])
        if not first_fast:
            digits = digits[:, ::-1]
        yield start, values[digits]


@lru_cache(maxsize=24)
def hkl_grid(limits, *, sorted_norm=False):
    # MATLAB app: h slowest, l fastest; stable Manhattan order for seed HKLs.
    hkl = np.asarray(list(product(*(range(-v, v + 1) for v in limits))), dtype=int)
    if sorted_norm:
        hkl = hkl[np.argsort(np.abs(hkl).sum(axis=1), kind="stable")]
    hkl.setflags(write=False)
    return hkl


def primitive_planes(limit=2, explicit=None):
    items = explicit if explicit is not None else product(range(-limit, limit + 1), repeat=3)
    unique = set()
    for item in items:
        p = tuple(int(x) for x in item)
        g = gcd(gcd(abs(p[0]), abs(p[1])), abs(p[2]))
        if g == 0:
            continue
        p = tuple(x // g for x in p)
        # +/- defines the same physical plane; reflection grid includes both signs.
        if next(x for x in p if x) < 0:
            p = tuple(-x for x in p)
        unique.add(p)
    return np.asarray(sorted(unique, key=lambda p: (sum(abs(x) for x in p), p)), dtype=int)


def direct_metric(cell):
    a, b, c, alpha, beta, gamma = cell
    return np.array([[a*a, a*b*np.cos(gamma), a*c*np.cos(beta)],
                     [a*b*np.cos(gamma), b*b, b*c*np.cos(alpha)],
                     [a*c*np.cos(beta), b*c*np.cos(alpha), c*c]])


def reciprocal_metric(cell):
    return TAU**2 * np.linalg.inv(direct_metric(cell))


def metric_to_cell(metric):
    np.linalg.cholesky(metric)  # Must be positive definite, not just positive diagonal.
    d = TAU**2 * np.linalg.inv(metric)
    lengths = np.sqrt(np.diag(d))
    angles = [np.arccos(np.clip(d[j, k] / (lengths[j]*lengths[k]), -1, 1))
              for j, k in ((1, 2), (0, 2), (0, 1))]
    return np.r_[lengths, angles]


def metric_design(h):
    h, k, l = np.asarray(h).T
    return np.column_stack([h*h, k*k, l*l, 2*h*k, 2*h*l, 2*k*l])


def six_to_metric(v):
    v = np.asarray(v)
    out = np.empty(v.shape[:-1] + (3, 3))
    out[..., 0, 0], out[..., 1, 1], out[..., 2, 2] = v[..., 0], v[..., 1], v[..., 2]
    out[..., 0, 1] = out[..., 1, 0] = v[..., 3]
    out[..., 0, 2] = out[..., 2, 0] = v[..., 4]
    out[..., 1, 2] = out[..., 2, 1] = v[..., 5]
    return out


def reciprocal_basis(metric, plane=None, projection=None):
    b = np.linalg.cholesky(metric).T
    if plane is None:
        if projection is None:
            return b
        normal = np.linalg.solve(b.T, projection)
    else:
        normal = b @ plane
    normal = normal / np.linalg.norm(normal)
    ref = np.eye(3)[np.argmin(np.abs(normal))]
    x = np.cross(ref, normal); x /= np.linalg.norm(x)
    y = np.cross(normal, x)
    return np.vstack([x, y, normal]) @ b


def predict(metric, hkl, plane=None, projection=None):
    hkl = np.asarray(hkl)
    q2 = np.einsum('...i,ij,...j->...', hkl, metric, hkl)
    if plane is not None:
        p = np.asarray(plane)
        t = metric @ p / np.sqrt(p @ metric @ p)
    elif projection is not None:
        t = projection
    else:
        raise ValueError("Need plane or projection")
    z = hkl @ t
    xy2 = q2 - z*z
    if not np.iscomplexobj(q2):
        invalid = (xy2 < -1e-9) | (q2 < 0)
        xy = np.sqrt(np.maximum(xy2, 0))
        xy = np.where(invalid, np.nan, xy)
    else:
        xy = np.sqrt(xy2)
    return np.column_stack([xy, z, np.sqrt(q2)])


def observed3(coords):
    return np.column_stack([coords, np.linalg.norm(coords, axis=1)])


def assign(metric, coords, grid, *, plane=None, projection=None, channels=(0, 1, 2)):
    pred = predict(metric, grid, plane=plane, projection=projection)
    valid = np.isfinite(pred).all(axis=1) & (pred[:, 1] >= -1e-10)
    valid &= np.any(grid != 0, axis=1)
    if not np.any(valid):
        raise ValueError("No physically valid predicted reflections")
    pred = pred[valid]; h = grid[valid]
    target = observed3(coords)
    # argmin preserves first-grid tie semantics; bounded by grid and peak counts.
    distances = cdist(target[:, channels], pred[:, channels], metric="sqeuclidean")
    idx = np.argmin(distances, axis=1)
    return h[idx].copy(), pred[idx], np.sqrt(distances[np.arange(len(coords)), idx] / len(channels))


def pair_reduced(cell, tol=1e-9):
    """updateplot's three pairwise Buerger inequalities (not full Niggli)."""
    d = direct_metric(cell)
    return all(2*abs(d[i, j]) <= min(d[i, i], d[j, j]) + tol
               for i, j in ((0, 1), (0, 2), (1, 2)))


def within_bounds(cell, settings, *, no_specular=False):
    if not np.isfinite(cell).all() or np.any(cell[:3] <= 0) or np.any(cell[3:] <= 0) or np.any(cell[3:] >= np.pi):
        return False
    try:
        d = direct_metric(cell)
        np.linalg.cholesky(d)
    except np.linalg.LinAlgError:
        return False
    values = np.r_[cell[:3], np.rad2deg(cell[3:])]
    for value, (lo, hi) in zip(values, settings.length_bounds + settings.angle_bounds):
        if value < lo - 1e-9 or value > hi + 1e-9:
            return False
    vol = np.sqrt(np.linalg.det(d))
    lo, hi = settings.without_specular_volume_bounds if no_specular else settings.volume_bounds
    return lo <= vol <= hi


def first_order(cell, h, coords, *, plane=None, continuous=None, channels=(0, 1, 2)):
    """One native normal-equation step, using complex-step analytic derivatives.

    Same physical q functions and one-step objective as the source; avoids its
    expanded hand-coded derivatives and divides-by-zero at specular xy=0.
    No iterative SciPy optimizer or AI refinement is used.
    """
    def evaluate(c):
        g = reciprocal_metric(c)
        if plane is not None:
            return predict(g, h, plane=plane)[:, channels].ravel()
        if continuous is None:
            return np.sqrt(np.einsum('ni,ij,nj->n', h, g, h))
        # Native no-specular free coefficients, refitted between correction steps.
        return predict(g, h, projection=g @ continuous)[:, channels].ravel()

    target = observed3(coords)[:, channels].ravel() if plane is not None or continuous is not None else np.linalg.norm(coords, axis=1)
    prediction = evaluate(cell)
    jac = np.empty((len(prediction), 6))
    for j in range(6):
        perturbed = np.asarray(cell, dtype=complex)
        perturbed[j] += 1e-24j
        jac[:, j] = np.imag(evaluate(perturbed)) / 1e-24
    valid = np.isfinite(jac).all(axis=1) & np.isfinite(prediction)
    # sqrt(qxy^2) is nondifferentiable for exactly specular reflections; qz and
    # |q| contain their information. Suppress numerical complex-step explosions.
    valid &= np.max(np.abs(jac), axis=1) < 1e12
    j = jac[valid]; residual = target[valid] - prediction[valid]
    normal, rhs = j.T @ j, j.T @ residual
    active = np.flatnonzero(np.any(normal != 0, axis=1))
    m = normal[np.ix_(active, active)]
    if len(active) == 0 or not np.isfinite(m).all():
        return cell.copy(), False
    # Source determinant gate, with rank check to avoid unstable exceptions.
    sign, logdet = np.linalg.slogdet(m.T @ m)
    if sign <= 0 or logdet <= np.log(.0001) or np.linalg.matrix_rank(m) < len(active):
        return cell.copy(), False
    try:
        delta = np.linalg.solve(m, rhs[active])
    except np.linalg.LinAlgError:
        return cell.copy(), False
    result = cell.copy(); result[active] += delta
    try:
        np.linalg.cholesky(direct_metric(result))
    except np.linalg.LinAlgError:
        return cell.copy(), False
    if np.any(result[:3] <= 0) or np.any(result[3:] <= 0) or np.any(result[3:] >= np.pi):
        return cell.copy(), False
    return result, True


def reduce_basis(metric, h, *, plane=None, projection=None, check=lambda: None):
    """Native short-vector/triple search, preserving exact reflection transforms.

    Returns metric, transformed H, transformed plane (if given), transformation.
    Direct vectors here are native AAe * 2*pi; enforce the original integer tests.
    """
    b = reciprocal_basis(metric, plane=plane, projection=projection)
    n = len(h) if len(h) <= 12 else len(h)//2
    best = None
    for k, triple in enumerate(combinations(range(n), 3)):
        if k % 128 == 0:
            check()
        selected = h[list(triple)]
        det = abs(np.linalg.det(selected))
        if det > .1:
            key = (round(det, 9), float(np.linalg.norm(selected, axis=1).sum()), triple)
            if best is None or key < best[0]:
                best = (key, selected)
    identity = np.eye(3, dtype=int)
    if best is None:
        return metric, h, plane, identity
    selected = best[1]
    maxh = max(1, int(np.max(np.abs(h))))
    grid = hkl_grid((maxh,)*3)
    grid = grid[np.any(grid != 0, axis=1)]
    d = selected @ b.T
    vectors = np.linalg.solve(d, grid.T).T
    lengths = TAU*np.linalg.norm(vectors, axis=1)
    order = np.lexsort(tuple(np.column_stack([lengths, vectors, grid])[:, j]
                             for j in range(6, -1, -1)))
    vectors, grid, lengths = vectors[order], grid[order], lengths[order]
    upper = int(np.sum(lengths <= max(metric_to_cell(metric)[:3]))) + 2
    vectors, grid = vectors[:upper], grid[:upper]
    for k, triple in enumerate(combinations(range(len(grid)), 3)):
        if k % 64 == 0:
            check()
        direct = vectors[list(triple)].T
        if abs(np.linalg.det(direct))*TAU**3 <= .01:
            continue
        transform = np.linalg.solve(selected, grid[list(triple)].T).T
        newh = h @ transform.T
        if np.linalg.norm(newh - matlab_round(newh)) >= .001:
            continue
        newplane = None if plane is None else transform @ plane
        if newplane is not None and np.linalg.norm(newplane - matlab_round(newplane)) >= .001:
            continue
        # The source's three sign adjustments, applied to H and plane together.
        cosines = lambda a: np.array([np.dot(a[:, j], a[:, k]) / np.linalg.norm(a[:, j]) /
                                      np.linalg.norm(a[:, k]) for j, k in ((1,2),(0,2),(0,1))])
        ca, cb, cg = cosines(direct)
        signs = np.ones(3, dtype=int)
        if abs(abs(cg)+abs(cb)+abs(ca)-abs(cg-cb-ca)) < .0001:
            signs[2] *= -1; cb = -cb; ca = -ca
        if abs(abs(cg)+abs(cb)+abs(ca)-abs(-cg-cb+ca)) < .0001:
            signs[2] *= -1; signs[1] *= -1; cb = -cb; cg = -cg
        if abs(abs(cg)+abs(cb)+abs(ca)-abs(-cg+cb-ca)) < .0001:
            signs[1] *= -1
        direct = direct * signs
        transform = signs[:, None]*transform
        newh = matlab_round(h @ transform.T).astype(int)
        newplane = None if plane is None else matlab_round(transform @ plane).astype(int)
        newb = np.linalg.inv(direct).T
        return newb.T @ newb, newh, newplane, transform
    return metric, h, plane, identity
