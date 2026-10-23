# Source provenance and numerical adaptations

Original GIDInd by M. Kainz and J. Simbrunner (2021), Graz University of
Technology. Repository: https://github.com/m-kainz/GIDInd, commit
`d15a4b8808d20869e7b92c87bff04a66d9c58a3b`, source archive
`GIDInd_5.4.21.0.zip`, directory `GIDInd_2021_04_05`.

The upstream MATLAB archive/source was used without modifying it. It is not
bundled in this release. `provenance/upstream_source_sha256.json` records the
original source hashes; obtain the distribution from the upstream repository. Translation and adaptations were performed
with LLM assistance for this project's local benchmark work. Derived code
remains GPL-3.0-or-later; the original license text is included as `LICENSE`.

## Source mapping

| Python subsystem | MATLAB source |
| --- | --- |
| Strict loaders and seed selection | `testinputdata`, `function_INITIALIZE_GUI` |
| Enumeration/rounding | `LPermutation`, `function_PERMVEC_NEW`, MATLAB `round` |
| Specular ABgamma equations/filtering | `function_NEWMATRIXFILLER`, `function_XYZ`, `function_ABG_RESTRICTED`, `function_CALC_ABGfromXYZ_RESTRICTED_v7`, `function_SUB2_CONDITION`, `function_ABGNIGGLI` |
| Redundant HK assignments | `function_NEWABGSUBS`, `function_NEWINDEXING_26`, `function_MINKLAUEEVALUATOR`, `function_NEWRMSD_GXY` |
| Specular KLR and cell reconstruction | `function_CALCSUBSIS`, `function_NEWPACKING`, `function_PART_TWO_INDEXING_QXY_QZ_SORT`, `function_NEWCELLPARAMETER_RESTRICTED` |
| No-specular seed generation, metric fit and indexing | Main `.mlapp`, `updateplot`, `mono==0 && tri==1` branch |
| First-order correction | `function_BUERGER_CELL_OPTIMIZATION_SORT` and triclinic `updateplot` normal-equation sections |
| Short-vector basis reduction and transforms | `function_BUERGER_CELL_OPTIMIZATION_SORT`, `function_BUERGER_CELL`, triclinic `updateplot` final reduction |
| Defaults | Main app private properties and dialog component defaults; not `SCRIPT_indexing_SORT.m` |

## Deliberate adaptations; parity is not claimed

Version 0.2 adds two explicit variants. `--algorithm upstream` retains the
previous port and the adaptations below. The default `--algorithm corrected`
also changes no-specular tail selection as documented in
[ALGORITHM_VARIANTS.md](ALGORITHM_VARIANTS.md). It rejects uninformative or
rank-deficient seed systems, forbids zero reflection indices, and evaluates a
bounded tail-HKL beam. It also anchors the native score window after the first
accepted discrete orientation. These are declared search changes, not a claim of exact
MATLAB execution or complete enumeration of six reflection assignments.

1. CPU float64 replaces MATLAB's mixed single/double arrays. Independent systems
   are processed in bounded batches. Nonsingular stacked normal solves replace
   repeated tall-matrix backslash in the small seed systems. Ill-conditioned
   systems can consequently differ; determinants and rank gates are retained.
2. GUI state, dialogs, sleeps, plotting, manual candidate selection and manual
   stop buttons are removed. Limits and deterministic anytime stage scheduling
   are explicit and recorded. Intermediate and output caps mark a partial search.
3. Inputs are strict physical coordinates. The original initialization selects
   rows containing exactly one zero, which can include in-plane qz=0 rows as
   supposed specular peaks. This port selects qxy=0,qz>0 only, in original order.
   It does not discard valid in-plane peaks from the measured data. Specular
   seeds still exclude either-zero coordinates as the original does.
4. Absolute `uniquetol` grouping uses sorted first representatives. Exact MATLAB
   release-specific floating-point boundary behavior is not certified. Seed HK
   traversal and stable nearest-index ties are preserved; independent candidate
   batches may differ in tie order after deduplication.
5. Pairwise Buerger inequalities and the native observed-reflection short-vector
   basis search are retained. These are not presented as a complete Niggli
   reduction proof. All basis changes transform assigned HKLs and plane
   coefficients together, rather than copying the source's inconsistent angle
   reordering. Final bounds apply in the reported reduced basis.
6. First-order corrections remain single normal-equation steps (specular: one;
   no-specular: qnorm step then qnorm/qz step with projection refits). Complex-step
   differentiation of the reciprocal-metric equations replaces expanded analytic
   derivatives. Exactly specular transverse derivatives are omitted when
   singular; their longitudinal and qnorm observations remain. No iterative
   nonlinear optimizer and no AI gradient/Laplace procedure is substituted.
7. Nonphysical reciprocal metrics and genuinely negative transverse q squared
   are rejected, not silently converted to their real parts. The native
   `real(sqrt(...))` zero behavior is retained only for specular stage-one HK
   seed scoring. Reconstruction always uses physically valid reciprocal geometry.
8. No-specular continuous projection coefficients remain intermediate quantities.
   Final orientation is a **discrete adaptation**: evaluate bounded primitive
   integer contact planes, normalize normals in the reciprocal metric, reindex
   all measured peaks and recompute physical errors. No free coefficients are
   rounded into Miller indices. Nonprimitive specular orders are tested separately.
9. Specular search preserves first-order equations and cell gates but applies
   final residual checks to every retained observation, including all measured
   specular peaks. Source optimization drops the first sorted measurement via
   `datapoints(2:end,:)`; this port keeps observation/assignment alignment explicit.
   Final physical qxy/z/xyz thresholds are checked again after reduction/projection.
10. The no-specular relative-window filter includes exact zero-score synthetic
    solutions. Source `Sol(:,1)>0` and `<Min*(1+perc)` would remove those. Native
    seed ranking (`norm(qnorm_squared residual)/n + norm(qz residual)/n`) is
    preserved as a distinct score, not mislabeled as a physical RMSE.
11. A source typo calculates no-specular `errorqxyz` from `sumerrorqxy`; the port
    recomputes xyz RMSE from xyz residuals. The original PQ export also appears
    to interchange xy/z error labels. Output columns here follow their equations.
12. Duplicate seeds with identical assignments retain the best native seed score.
    Final candidates with the same integer plane, specular order and all assigned
    HKLs retain only the best physical residual, following upstream's deduplication
    by indexed reflections rather than counting several numerical fits separately.
    This does not prove crystallographic inequivalence between surviving cells.
    Final solutions are ranked by
    independently recomputed qxyz/qz/qxy errors; native seed scores remain visible.

## Reference evidence and limitations

The included PQ export's listed cell, plane and HKLs reconstruct qxyz RMSE
approximately 0.00212 Å^-1, consistent with its reported 0.0021. The DIP
no-specular export's listed cell, plane and HKLs do not reconstruct its claimed
errors; even its qnorm RMSE is approximately 0.224 Å^-1, independent of orientation.
It cannot be treated as a correct discrete-orientation oracle. The PQ export is retained as a minimal numerical test fixture; the broader
research audit is not bundled.

Fresh comparisons with the Windows/MATLAB executable remain required before
claiming equivalent solution sets. All timings refer to this port, not MATLAB
GIDInd. Search exhaustion is always relative to the recorded index/seed/bound
settings. No statistical cell-parameter uncertainties are implemented because
the source routines do not supply them.

Minimal MAPbBr3 counterexample traces in `tests/fixtures` exercise the original
qz-only tail selection's zero-HKL/singular-system trap. They are targeted
regression fixtures, not unbiased benchmark results or executable parity.

## Optional multiprocessing execution

The upstream specular `function_PART_ONE_INDEXING_QXY` and the triclinic branch
of `updateplot` contain `parfor` loops. The port now offers `--workers N` using
spawned processes around its existing bounded NumPy kernels; `workers=1` uses
the same kernels directly. Equations, batch shapes, native traversal order,
cutoffs, tie decisions and global duplicate-cache clearing are unchanged.
The coordinator commits tasks in order, owns all candidate decisions and file
writing, and leaves correction/reduction/discrete projection serial. No BLAS
multithreading, GPU implementation, or new mathematical search is introduced.
No-specular refits may be evaluated speculatively for seeds later discarded by
the coordinator's global cache; those outcomes do not affect counts or solutions.

Completed-work parity is verified against pre-refactor serial outputs captured
from commit e6029d4, plus serial/2-worker/4-worker comparisons. Frozen numerical
hashes in tests/fixtures/pre_multicore.json apply to the pinned Linux x86_64
Python 3.12 environment; other environments still compare their own serial and
parallel outputs exactly. Time-limited runs can explore different amounts of
work and must not be expected to return identical partial solutions. Pool
startup consumes the existing wall-time budgets. Timings continue to apply to
this port and do not establish original executable performance or equivalence.
