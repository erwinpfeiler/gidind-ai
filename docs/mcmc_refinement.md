# GPU ensemble MCMC refinement

MCMC is an offline alternative to the default gradient/Laplace refinement. It
samples six cell parameters for each fixed candidate surface orientation. The
neural prediction initializes the walkers; it is not used as a prior. This is a
fresh implementation from the DE-MC transition described by ter Braak (2006),
Statistics and Computing 16, 239–249, DOI
[10.1007/s11222-006-8769-1](https://doi.org/10.1007/s11222-006-8769-1).
No historical commits, removed samplers, backups, or archived MCMC results were
used. Current project geometry and the attributed MIT Niggli reducer are reused.
No new runtime dependencies are required: sampling uses PyTorch, summaries use
NumPy, and rank normalization uses SciPy, all already declared by this project.

## Running inference

```sh
gidind-ai infer \
  --input examples/real_cases/pq.csv \
  --output-dir outputs/pq_mcmc \
  --device cuda \
  --refinement-algorithm mcmc \
  --q-sigma 0.003 \
  --mcmc-orientation-chunk-size 4
```

Replace `0.003` with the measured one-coordinate standard deviation of peak
positions in physical Å⁻¹. The same scalar applies to qxy and qz. It must be
finite and positive, even when the synthetic data would otherwise be noiseless.
The loader's scale s satisfies q_model = q_physical / s. The sampler therefore
uses sigma_model = sigma_physical / s and reports cell lengths divided by s.
Stored log likelihoods include the change of density back to physical q units.
The existing `--refinement-backend` selects only local-fit execution; MCMC uses
its own chunk settings. CUDA requests fail explicitly when CUDA is unavailable.
CPU execution is supported using `--device cpu`.

The prior is uniform over configured cell bounds, conditioned on positive volume
and the configured minimum cell height. Each measured reflection retained inside
the measurement window is reassigned to its nearest eligible simulated reflection
at every evaluation. Duplicate assignments are allowed. The Gaussian residual
log likelihood sums both coordinate residuals for every observation, with its
normalization; there is no distance cutoff or rejection of poorly matched
observations. Invalid proposals and simulations without eligible reflections
have log probability minus infinity.

This matches the manuscript's nearest-reflection Gaussian residual model. It is
not a marginal likelihood over peak assignments. Extra measured peaks can affect
the result substantially; missing simulated reflections are not penalized.
Orientation scores are the maximum retained residual log likelihood, suitable as
a goodness-of-fit ranking under these assumptions, not Bayesian evidence or
orientation probabilities. Different orientations can remain indistinguishable.

## Sampler settings and basis

Defaults are four independent ensembles, 32 walkers per ensemble, 2,000 burn-in
sweeps, 4,000 production sweeps, thinning interval two, seed 42, DE scale
2.38/sqrt(12), Gaussian proposal jitter 1e-6, and initialization width 0.005.
Both widths are in normalized cell coordinates. One sweep updates all walkers.
The first retained sweep is production sweep two; leftover production sweeps
shorter than a thinning interval are not stored. Acceptance rates cover all
production proposals, including those not retained.

Every setting is exposed as `--mcmc-ensembles`, `--mcmc-walkers`,
`--mcmc-burn-in`, `--mcmc-production`, `--mcmc-thin`, `--mcmc-seed`,
`--mcmc-gamma`, `--mcmc-jitter`, `--mcmc-initial-sigma`, and
`--mcmc-initialization-attempts`. There is no adaptation or automatic stopping.
The walker count must be even and at least four; at least two independent
ensembles are required for diagnostics.

Perturbed initial cells are Niggli reduced and checked for physical validity,
with up to 1,000 attempts per walker. Candidate Miller indices refer to each
resulting reduced basis. No orientation-preserving basis transform is claimed
between a perturbed unreduced cell and its reduced initialization. Sampling then
keeps this basis and fixed orientation: proposals are neither Niggli reduced nor
clipped. Niggli reduction does not define the sampling prior. Near reduction
boundaries, initialization may select distinct local basins; the sampler does
not promise a global search or a unique crystallographic representation.

Walkers are split into two groups. Each group's proposals use distinct ordered
donors from the frozen complementary group in the same ensemble. The second
group sees the accepted first-group updates. No donor is shared across
orientations or independent ensembles. Independent streams make results
reproducible across orientation chunk sizes on the same device. CPU and GPU RNGs
are not expected to produce identical trajectories.

`--mcmc-orientation-chunk-size` defaults to one; larger values batch orientations
alongside walkers and ensembles. `--mcmc-reflection-chunk-size` defaults to 256
and bounds the simulated-reflection axis of the nearest-distance calculation.
The distance working set still scales with observed peak count. Retained samples
are transferred to CPU storage; likelihood evaluation stays on the selected
device in float64. Niggli initialization and diagnostics run on the CPU. Larger
orientation chunks trade GPU memory for throughput; CPU memory also includes the
complete retained sample arrays.

## Results and convergence

Inference writes `results.json`, `report.txt`, and `posterior_samples.npz`.
With figures enabled, it writes six marginal histograms for each of the five
highest-scoring orientations. The default local-fit reconstruction figures and
Laplace fields are unchanged when `gradient_laplace` is selected.

The compressed archive contains physical cell samples with axes
`[orientation, ensemble, draw, walker, parameter]` and log likelihoods with axes
`[orientation, ensemble, draw, walker]`. Parameters are a,b,c in Å and
alpha,beta,gamma in degrees. It includes orientation indices, production acceptance
rates, settings, and q scaling. Failed orientations have NaN sample slots and
minus-infinity likelihood slots; failures are reported explicitly in JSON.

Successful initialization produces posterior means, standard deviations, sample
covariance and central 95% intervals. It does not imply convergence. Diagnostics
compare the corresponding walker trajectory across independently seeded
ensembles, never interacting walkers as separate chains. Rank-normalized split
and folded Rhat, bulk ESS, and tail ESS are computed for every walker/parameter
pair. A run is labeled converged only if every Rhat is below 1.01 and both ESS
estimates are at least 400. Insufficient draws or stuck chains remain
unconverged; unavailable Rhat values are JSON null. The diagnostic definitions
follow [Vehtari et al. (2021)](https://doi.org/10.1214/20-BA1221).
Conservative extremes across walkers are also reported. Diagnostics describe
mixing in the sampled region and cannot establish that all modes were found.

## Validation and benchmarks

Run the statistical, likelihood, artifact, and workflow checks:

```sh
MPLBACKEND=Agg python -m pytest tests/test_mcmc.py tests/test_workflows.py -q
```

The tests include Gaussian and bounded uniform targets, numerical integration of
a conditional diffraction posterior, a fixed-assignment small-noise Laplace
comparison, q scaling and covariance units, initialization failures, and exact
CPU orientation-chunk reproducibility. CUDA likelihood parity is tested at
rtol 1e-10 and atol 1e-10 when CUDA is present.

The checkpoint-free runner tests well-constrained and weakly constrained noisy
synthetic patterns against competing orientations. It checks CPU/CUDA likelihood
parity, compares orientation chunks one and three, and writes timings, peak CUDA
allocation, effective samples per second, recovery errors, intervals, diagnostics,
and repeated-trial coverage:

```sh
PYTHONPATH=src python scripts/validate_mcmc.py \
  --device cuda --output-dir outputs/mcmc_cuda_validation
```

For a short execution check, add `--quick`. A CPU check can use
`--device cpu --quick --max-laue-index 2`; this does not validate GPU performance
or uncertainty coverage. Full coverage defaults to four noise realizations per
case, adjustable with `--trials`; this is descriptive, not a precise calibration
study. Inspect convergence before interpreting the coverage or ESS rates.

The standard checkpoint-based `gidind-ai benchmark` also accepts
`--refinement-algorithm mcmc --q-sigma VALUE` and all MCMC settings. It injects
independent Gaussian coordinate noise with that known sigma before regenerating
network inputs. It writes one posterior archive per synthetic example, recovery
errors and descriptive marginal coverage. For this benchmark, disable any
additional configured testing noise if the intended noise is exactly sigma;
existing missing/extra-reflection testing settings still apply.

CUDA execution and performance remain pending until the GPU checks run on actual
hardware. No historical paper-result reproduction or seconds-level runtime is
claimed for this implementation.
