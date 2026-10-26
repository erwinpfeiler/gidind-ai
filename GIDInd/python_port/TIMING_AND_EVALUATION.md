# Reference-blind timing and independent budget runs

The fitter never receives a reference cell, reference HKLs, correctness labels
or reference tolerances. Finding a correct result does **not** stop a run.
The usual search/time/candidate/solution limits remain. Correctness is evaluated
afterward by a separate `evaluate` command.

## Recorded times

Every new run saves `solution_events.jsonl`. Accepted solutions and improved
duplicate replacements are logged and flushed in actual coordinator order.
Each event includes its cell, plane, residuals, stable solution slot, actual
validation wall time and the originating candidate's discovery time. Solution
slots describe the evolving pool, not the final sorted solution IDs.

`candidate_timing.json` accompanies the intermediate candidates. Discovery means
when the coordinator committed a retained raw candidate, not when a worker
finished speculative work. A better replacement gets its own discovery time;
the assignment's first-seen time remains available separately.

The post-run report provides:

| Field | Meaning |
| --- | --- |
| `first_any_validated_seconds` | First accepted solution, regardless of reference correctness |
| `first_correct_validated_seconds` | First retained, validated solution matching the reference, even if later replaced/displaced |
| `first_correct_top_k_seconds` | First event where the running retained top K includes a reference match |
| `retrospective_correct_candidate_discovery_seconds` | Earliest recorded discovery of a candidate that subsequently produced a reference-matching solution |
| `final_best_match_rank`, `final_top_k_success` | Reference quality of the actual final ranked output |
| `total_wall_seconds`, `aggregate_cpu_seconds` | Actual full-run measurements, not an oracle-truncated runtime |

The port still finalizes after search. Therefore validated times often fall
near the budget, and **candidate discovery is not convergence or usable-solution
latency**. These fields honestly describe this two-phase implementation. Logging
overhead is included in run time; post-run reference-evaluation time is reported
separately. No source equations or search order are changed by reference checks.

First-correct/top-K events are retrospective measurements, not stopping rules.
For example, a correct result can appear early and then fall out of the final
top K. Both the early event and final failure are reported. Missing success
times remain `null` with censoring/status information, never a fabricated
"12-hour convergence time". Failure/interruption are separate technical outcomes.
Historical results without timelines can still be graded for final correctness,
but first-correct times are unavailable; timestamps cannot be reconstructed.
PQ/Cu-BOD duplicate entries inherit canonical timing and are marked reused.

## Correctness criteria

Provide trusted reference cells in Å/degrees, integer contact planes and a source
for each of `pq`, `mapbbr3`, `naib_s11`, `cu_bod_mof`. Freeze the same reference
criteria for both compared methods. A fit from a competing method is not
independent ground truth merely because it has small residuals.

`configs/reference_evaluation.example.json` is a **synthetic schema example**,
not experimental reference data or a recommendation of its illustrative
tolerances. Replace its cases and justify the tolerances from the experiment.
Case-level `criteria` overrides are supported and recorded explicitly.

`configs/reference_evaluation.experimental.json` records the user's revision
table, excluding its GIDInd-AI columns. The user approved its provisional common
tolerances on 2026-10-20: 2% lengths, 2° angles, 5% volume, 2° normal deviation,
and 0.05 Å⁻¹ RMSE cutoffs. PQ uses confirmed (102); MAPbBr3 uses confirmed (001).
NAIB-S11 uses tentative (001), explicitly flagged in each evaluation JSON/CSV
row: orientation-dependent correctness labels are conditional on that assumption.
Cu-BOD retains the table's cell (9.666,10.87,10.87) and short-axis normal (100),
equivalent to (010) in (10.87,9.666,10.87). The user confirmed (100) in the paper
on 2026-10-20; never attach (010) to the unpermuted table cell. The Cu reference
column is labeled GIDInd, so its provenance differs from the three literature
references. The report preserves these qualifications and the full reference
configuration; confirm the NAIB-S11 orientation before presenting its labels as
definitive in the paper. Reference tolerances are comparison choices, not
parameter uncertainties.

Matching checks lengths, angles, volume and residuals, allowing bounded integer
direct-basis transforms with determinant ±1. Contact-plane normals are transformed
consistently in the reciprocal metric and compared up to sign. Thus sign changes,
axis permutations and simple sheared equivalent cells can match without exact
HKL-list equality. `basis_limit` bounds the tested transform coefficients; a miss
is **not matched within that domain**, not a proof of lattice inequivalence.
Supercells are not silently treated as the same primitive lattice. References
must use consistent lattice conventions. The matcher uses independent metric
calculations, not the fitter's reduction routine.

The residual criterion expects RMSE over **all retained observed peaks**. To
apply this evaluator to GIDInd-AI, normalize its outputs to the same cell/plane,
all-peak RMSE and event schema, or call the independent `match_reference` API.
An AI loss on a subset or with different scaling is not interchangeable with
these residuals. No AI search/training implementation is changed here.

## ESRF budget ladder

For the urgent two-hour revision workflow, prefer
`bash GIDInd/python_port/submit_revision_2h.sh` after updating the checkout;
it runs smoke then one full point, gated on technical smoke success.
See [RUNTIME_REPAIR.md](RUNTIME_REPAIR.md). The longer ladder below is optional.


Once existing arrays **and their collectors** finish, update the checkout and run:

```sh
cd /path/to/gidind-ai
git pull --ff-only
bash GIDInd/python_port/submit_esrf_array.sh ladder --algorithm corrected
```

This submits **six array tasks, 32 CPUs each**, with no concurrency cap. Each task
runs its own case freshly at **120, 600, 3600 and 7200 seconds** (2 min, 10 min,
1 h, 2 h), sequentially. Budgets sum to 3 h 12 min per task; the
4-hour allocation leaves 48 minutes for process/output overhead. Search limits
can end a run earlier. No earlier solution/reference stops a later budget.
Each budget retains eight entries and six actual computations: **24 computations
and 32 requested entries** in the full ladder. These are independent runs, not
checkpoints or continuations of the 2-hour run. The collector writes one root
`benchmark.json` plus a `benchmark.json` under each `budget_*s/` directory.

`--algorithm upstream` selects a separate upstream-compatibility ladder. The
usual smoke/full commands remain available. Do not submit a standalone full run
in addition to a ladder just to obtain its 2-hour point: it is already included.

For local sequential execution instead of a SLURM array:

```sh
GIDInd/python_port/run.sh benchmark --input-dir gidind_peak_lists_all \
  --config GIDInd/python_port/configs/benchmark_settings.json \
  --algorithm corrected --workers 8 --budgets-seconds 120 600 3600 7200 \
  --output-dir /NEW/OUTPUT/DIRECTORY
```

After collection, evaluate with the completed trusted reference file:

```sh
GIDInd/python_port/run.sh evaluate --run-dir /EXACT/LADDER/DIRECTORY \
  --references /PATH/TO/trusted_references.json --top-k 5 \
  --output-dir /NEW/EVALUATION/DIRECTORY
```

This writes `evaluation.json` and `evaluation.csv` without modifying fit results.
Use `--case mapbbr3` when evaluating one case directory instead of a benchmark.
Missing reference cases are labeled unavailable, not failures. The reference
file, evaluator hash, effective criteria and top K are preserved in the report.

The ladder report includes the **smallest tested budget whose final top K
succeeds**. This is not an exact continuous-run convergence time: search stage
budgets and candidate filters can make success nonmonotonic across independent
budgets. Report the measured points, full-run errors/ambiguity/work counts, and
the distinct first-correct event times rather than assuming monotonic success.
