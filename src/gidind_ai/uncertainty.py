"""Identifiability checks for conditional local refinement uncertainties.

The Jacobian uses the existing weighted residuals and six normalized cell
parameters. Hessian regularization must not create information in its null space.
"""

import torch
from torch import Tensor

IDENTIFIABILITY_RTOL = 1e-8
NULL_PROJECTION_TOL = 1e-8
PARAMETERS = ("a", "b", "c", "alpha", "beta", "gamma")


def guard_uncertainties(
    uncertainty: Tensor, jacobian: Tensor, *, rtol: float = IDENTIFIABILITY_RTOL
) -> tuple[Tensor, dict[str, Tensor]]:
    """Mark parameter components of unresolved directions as infinite.

    Leading batch dimensions are supported; SVD stays on the input device.
    Rank is assessed before Hessian damping or residual-variance scaling. Finite
    legacy uncertainties are retained verbatim for identifiable parameters.
    This is a local, fixed-assignment check, not a certificate of global uniqueness.
    """
    if not 0 < rtol < 1 or jacobian.shape[-1] != 6 or uncertainty.shape[-1] != 6:
        raise ValueError("Expected six parameters and 0 < rank tolerance < 1")
    finite = torch.isfinite(jacobian).all(dim=(-2, -1))
    safe = torch.where(finite[..., None, None], jacobian, 0)
    # Economy SVD must still return all six right singular vectors when there
    # are fewer than six residuals. Padding with zero rows adds no information.
    if safe.shape[-2] < 6:
        safe = torch.cat((safe, safe.new_zeros((*safe.shape[:-2], 6 - safe.shape[-2], 6))), -2)
    _, singular, right = torch.linalg.svd(safe, full_matrices=False)
    unresolved = singular <= rtol * singular[..., :1]
    null_projection = (right.square() * unresolved[..., :, None]).sum(-2)
    unconstrained = null_projection > NULL_PROJECTION_TOL
    guarded = torch.where(unconstrained, float("inf"), uncertainty)
    guarded = torch.where(finite[..., None], guarded, float("nan"))
    return guarded, {
        "rank": (~unresolved).sum(-1),
        "singular_values": singular,
        "unconstrained": unconstrained,
        "finite": finite,
    }


def diagnostic_row(diagnostics: dict[str, Tensor], index=None) -> dict:
    """Serialize only after fitting; no reporting transfers in the solver loop."""
    row = {k: v if index is None else v[index] for k, v in diagnostics.items()}
    return {
        "rank": int(row["rank"].detach().cpu()),
        "singular_values": row["singular_values"].detach().cpu().tolist(),
        "unconstrained_parameters": [
            name
            for name, flag in zip(PARAMETERS, row["unconstrained"].detach().cpu().tolist())
            if flag
        ],
        "finite_jacobian": bool(row["finite"].detach().cpu()),
        "relative_rank_tolerance": IDENTIFIABILITY_RTOL,
        "scope": "local weighted residual Jacobian; fixed final assignments; normalized cell parameters",
    }
