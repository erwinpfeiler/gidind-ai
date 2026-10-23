# Refinement uncertainties and identifiability

Serial CG and batched LM retain their existing weighted continuous-coordinate
loss, matching schedule, Hessian covariance normalization and uncertainty scales.
Before reporting finite precision, both now check the local weighted residual
Jacobian for six normalized cell parameters, with final assignments fixed.

Singular values no larger than 10⁻⁸ times the largest singular value are treated
as unresolved directions. For each cell parameter, the squared projection onto
this unresolved subspace is computed; a projection above 10⁻⁸ gives infinite
uncertainty for that parameter. Identifiable parameters retain their existing
uncertainties exactly. Nonfinite Jacobians produce NaN uncertainties. Existing
uncertainty-based acceptance checks reject both infinite and NaN values.

This check precedes Hessian regularization and residual-variance scaling.
Consequently, a nearly zero residual cannot make a flat parameter direction
look precise. The check is local and conditional on matching: full rank does
not establish a unique global indexing solution. At a symmetry point the
linearized model can be singular even if nonlinear information exists; such a
case does not justify a finite Gaussian/Laplace uncertainty from this calculation.

Low-level refinement results include an `identifiability` record with rank,
six singular values, affected parameter names, the relative threshold and
the scope of the calculation. SVD and null-space calculations stay on-device
for the batched backend. Empty matching has rank zero and infinite
uncertainties when uncertainty evaluation is enabled. When callers explicitly
disable uncertainty evaluation, the existing zero-placeholder convention is
retained; zero placeholders must not be interpreted as measured precision.

The serial empty-match fallback also now returns degree-valued angles in
`refined_predicted_cell_deg`. Previously that branch returned radians despite
the field name. Valid optimizer trajectories and loss calculations are unchanged.

Tests reproduce the recorded case 601, test degree units after empty matching,
check full-rank controls, mixed null directions, short/empty Jacobians, scale
invariance and isolation of a nonfinite batched candidate. Case 601 has a
rank-three Jacobian and now fails the finite-uncertainty acceptance criterion.

## Completed reviewer study

The original results remain under `reviewer_runs/ablation_10000_seed42/`.
The separate derived study `reviewer_runs/ablation_10000_seed42_corrected/`
converts the original serial empty-match angle units, reconstructs final
one-to-one matching and applies the rank safeguard to saved uncertainties.
It does not rerun the neural network or optimizer. Losses, timings and histories
remain those of the original run. Selection/acceptance and all exports are
regenerated. The corrected database also stores every original result in its
`uncertainty_corrections` table; code identities and provenance are in
`correction_manifest.json` and `correction_source_snapshot/`.

The correction can resume with:

```bash
.venv/bin/python scripts/correct_saved_refinement_uncertainties.py
```

Use the corrected folder for subsequent paper plots. It is a derived analysis,
not a new independent benchmark. Its original source snapshot records how fits
were obtained; the separate correction snapshot records the reassessment.
