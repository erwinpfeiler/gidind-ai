# Conventional GIDInd Python port

A CPU float64 Python adaptation of the triclinic GIDInd algorithms, with and
without specular reflections. Original GIDInd is by M. Kainz and J. Simbrunner.
This port uses a bounded integer contact-plane search; equivalence to the
original MATLAB executable is **unverified**. Read [PROVENANCE.md](PROVENANCE.md)
and [ALGORITHM_VARIANTS.md](ALGORITHM_VARIANTS.md) before interpreting results.
The default `corrected` variant repairs declared seed-search issues;
`--algorithm upstream` retains the earlier port behavior.

## Setup and use

Run the repository's root `bash setup.sh`; it installs this package in the same
`.venv` as GIDInd-AI. No MATLAB license or runtime is needed. From the repository root:

```bash
.venv/bin/gidind-reference index \
  --input .assets/real_cases/mapbbr3_without_specular.csv \
  --mode without-specular --workers 1 --time-limit-seconds 60 \
  --output-dir reviewer_outputs/gidind_mapbbr3
```

For a standalone installation, run `python -m pip install ./GIDInd/python_port`.
Runtime dependencies are NumPy, SciPy and openpyxl. `gidind-reference --help`
lists indexing, benchmark and post-run evaluation commands. Optional cluster
launchers are retained as templates; adapt scheduler directives, environment
paths and input directories to your installation before use.

Inputs are named `q_xy,q_z` CSVs or two-column CSV/XLSX files in Å⁻¹. Coordinates
are never rescaled. Origin placeholders are removed; without-specular mode also
removes q_xy ≤0.05 Å⁻¹. Removed rows and input checksums are recorded. Specular
mode requires a genuine q_xy=0, q_z>0 observation. All 49 primitive contact-plane
candidates with indices bounded by two are available by default.

`--workers N` uses spawned CPU processes and ordered commits of bounded batches;
one worker runs directly. Each worker uses one BLAS thread. Completed identical
work gives serial/parallel parity in the same numerical environment. A fixed
time budget can explore different amounts of work. API callers using multiple
workers must use an importable script with a `__main__` guard.

## Outputs and limitations

Runs save settings, environment, status, ranked solutions, assignments, residuals,
checkpoints and timing/event records. Disk-backed candidate storage avoids loading
the entire pool during checkpointing. Partial searches, timeouts and technical
failures remain explicit. See [CANDIDATE_STORAGE.md](CANDIDATE_STORAGE.md) and
[TIMING_AND_EVALUATION.md](TIMING_AND_EVALUATION.md) for reporting scopes.

The fitter never receives reference cells. Reference-based success and first-correct
times are evaluated afterward; they are retrospective measurements, not automatic
stopping criteria. Search exhaustion applies only to the configured bounded search
and does not establish uniqueness or guarantee a correct physical cell. Timings
from this Python port must not be labeled as original MATLAB GIDInd timings.
Cell-parameter uncertainties are not implemented.

The release includes source, GPL license, source hashes and minimal regression
fixtures. MATLAB binaries, original distribution archives, saved experimental
runs and manuscript/SI analysis workflows are excluded. GPL-3.0-or-later applies
to this adapted port; see [LICENSE](LICENSE).

```bash
.venv/bin/python -m pytest GIDInd/python_port/tests
```
