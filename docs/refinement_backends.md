# Serial and batched orientation refinement

The original CG solver remains the default and is available permanently.
The new single-process backend refines all candidate orientations together,
with independent float64 Levenberg–Marquardt steps for each six-parameter cell.

```sh
gidind-ai infer --input gidind_peak_lists/pq_without_specular.csv \
  --output-dir reviewer_outputs/pq_batched --device cuda --refinement-backend batched

gidind-ai infer --input gidind_peak_lists/pq_without_specular.csv \
  --output-dir reviewer_outputs/pq_serial --device cpu --refinement-backend serial
```

`--refinement-backend auto` currently resolves to **serial on both CPU and CUDA**.
CUDA automatic selection is gated off until hardware correctness and performance
checks pass. Explicit `batched` also works on CPU for diagnostics. GPU speedups
have not been established on the development Mac, which has no CUDA device.

The Python inference/benchmark APIs and single-phase regressor entry points accept
`refinement_backend`. `GixdRegressor(..., refinement_backend="batched")` selects
it for the instance's synthetic refinement workflows as well. The scalar
`direct_inversion_pytorch` API defaults to `serial` and accepts the same selector.

The batched backend retains the current geometric cell parameters, q domain,
weighted L2 objective, matching cutoffs, hybrid schedule, and acceptance rules.
It uses no Scattailab codec, Cholesky parameterization, alternative constraints,
reflection compaction, Sinkhorn approximation or new compiled dependency.
The adapted matching, residual and LM structure comes from Scattailab's legacy
GIWAXS refinement; the original implementation is not assumed to be a validated
performance baseline.

Exact rectangular Hungarian assignment runs in Torch on the selected device;
the cutoff is applied **after** solving the assignment, as in the serial solver.
Fractional matching keeps the same per-observed-peak weight normalization.
There are batch-wide termination checks in assignment/optimization, but no
per-element CPU reads or per-orientation optimizer loop. This does not eliminate
all host overhead, and small problems may remain faster on CPU.

Every orientation has its own step acceptance, damping and stopping state.
Failed candidates produce nonfinite uncertainty/loss diagnostics rather than
poisoning successful rows. Final hybrid reporting still uses one-to-one
assignments; the reported q loss is the last optimized objective, preserving
the existing distinction between these two assignments. Local uncertainties
use the original Hessian regularization, unweighted residual variance and
degrees-of-freedom convention. They remain conditional local estimates.

Result JSON includes backend, optimizer, per-candidate iterations/loss histories
and timing metadata. These are additive fields. CUDA single-phase execution
never creates process pools. One-worker CPU refinement also runs directly.
Legacy partial/multiphase algorithms are unchanged and reject their unsafe
CUDA process-pool path; run those algorithms on CPU.

## Compare from identical initializations

Download the four specular-free files first with the repository's downloader.
Then prepare a frozen set of initial predictions once:

```sh
python -m gidind_ai.refinement_validation \
  --peak-dir gidind_peak_lists --output-dir reviewer_outputs/refinement_inputs \
  --prepare-only --synthetic-per-system 20

python -m gidind_ai.refinement_validation \
  --prepared-inputs reviewer_outputs/refinement_inputs/prepared_inputs.json \
  --output-dir reviewer_outputs/refinement_comparison --devices cpu,cuda \
  --threads 1 --warmup 1 --repeats 5
```

The runner freezes all 49 network initializations for each experimental and
synthetic pattern, uses the pinned standard/high-missingness case profiles,
and records the prepared-input checksum. Synthetic cells come from the frozen
held-out arrays, balanced across seven source crystal systems with seed 42.
The runner does not retrain or modify input measurements.

`comparison.json` retains all candidates, cells, uncertainties, accepted flags,
q losses, repeat times, GPU peak memory, hardware, threads and library versions.
Synthetic records include ground-truth cell correctness for all orientations.
The preparation JSON records the true orientation and exact source row.
CPU serial, CUDA serial and CUDA batched comparisons use the same prepared data;
CPU batched is also recorded as a diagnostic.

For a quick integration run, use `--devices cpu --quick --warmup 0 --repeats 1`.
This reduces iteration budgets and is explicitly marked as a smoke test. It
must not be reported as paper-scale validation or hardware performance evidence.
The runner writes progress after each completed configuration, so partial
results remain available if a long hardware run is interrupted.

Automatic CUDA selection must not be enabled until the tests, all four real
cases, a balanced held-out evaluation, accepted-solution comparisons and five
timing repetitions have been reviewed on CUDA hardware. Regressions keep the
release gate disabled; the validation runner never changes it automatically.

## Development CPU preflight

The 2026-10-20 preflight used the same frozen initial predictions for both
solvers, normal case iteration budgets, all 49 orientations, the four
specular-free experimental inputs and one held-out sample per crystal system.
It was a single CPU run without warm-up, so it is not the five-repeat hardware
performance evaluation. Both solvers accepted exactly the same orientation sets:

| Pattern | Serial accepted | Batched accepted |
| --- | ---: | ---: |
| PQ | 3 | 3 |
| Cu-BOD | 1 | 1 |
| MAPbBr3 | 6 | 6 |
| NAIB-S11 | 1 | 1 |
| Triclinic | 4 | 4 |
| Monoclinic | 4 | 4 |
| Orthorhombic | 4 | 4 |
| Tetragonal | 0 | 0 |
| Rhombohedral | 6 | 6 |
| Hexagonal | 6 | 6 |
| Cubic | 4 | 4 |

Accepted parameters agreed within 1.1e-7 in model-space units except NAIB-S11.
For NAIB-S11's accepted (001) candidate, length differences reached 0.0422 in
scaled angstroms and angle differences 0.0424 degrees. CG loss was 2.0703e-5
and LM loss 2.3888e-5; both passed the existing acceptance thresholds. Both
fits were stationary for their final assignments (maximum normalized gradients
8.1e-10 and 4.2e-15). Increasing LM's inner budget from 10 to 50 did not change
its result. These distinct stationary fits require review in the larger
validation; no equivalence of all fitted parameters is claimed.

Some rejected orientations have different nonfinite-uncertainty diagnostics.
The reference can also return zero uncertainties for empty matches, while the
new backend marks empty fits as invalid with nonfinite uncertainty/loss.
Nonfinite local Hessian variances are retained rather than made positive by
changing the uncertainty definition. Fixed-assignment uncertainty parity is
covered separately by a numerical test. The tetragonal sample was not recovered
by either solver. A larger balanced set, CUDA numerical tests and synchronized
five-repeat CUDA measurements remain release requirements.

The comparison runner reports matching, optimization, uncertainty, other time
and total time. Serial phases are instrumented externally around matching,
CG minimization, explicit Hessian and residual-variance evaluation, with nested
calls excluded from their parent phase; remaining setup/reporting is other time.
