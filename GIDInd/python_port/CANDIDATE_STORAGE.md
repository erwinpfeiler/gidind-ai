# Lossless candidate storage, version 0.5

The previous Python port stopped native candidate collection at a fixed count.
This was a port resource safeguard, not a stopping rule in MATLAB's triclinic
no-specular branch. It caused the two-hour ESRF MAPbBr3 and Cu-BOD no-specular
cases to stop after about 263 and 47 seconds. Those results remain explicitly
`search_limited`; they must not be described as two-hour unsuccessful searches.

## Collection, sorting and filtering

The original no-specular code writes candidate rows into `Sol`, finishes the
configured enumeration, sorts by native score, and retains scores strictly
below `minimum * 1.2` before correction/refinement. Version 0.5 follows this
collect-then-sort/filter structure without a default intermediate count limit.
The port's previously declared numerical changes, duplicate-assignment handling,
zero-score handling and corrected-mode discrete-window anchor remain unchanged.
This is not new evidence of original MATLAB executable equivalence.

Each run owns `native_candidates.sqlite3`. Native metric/HKL/projection/plane,
scores, discovery timing and stable insertion slots are stored in the database.
An equal assignment retains the earlier candidate on a score tie; a strictly
better score replaces it in the same slot and retains first-seen timing.
An indexed score/slot traversal reproduces Python's stable score sort. No
top-N eviction or global-score pruning occurs during collection. Float64
metrics and scores retain their values; integer assignments are losslessly
compressed. The equations, worker kernels, ordered commits and final ranking
are unchanged. Specular stage-1 retained ABG rows also use disk storage and the
same area/error/parameter/UV/tie order. Existing per-UV uniqueness arrays remain
unchanged; this change does not rewrite those numerical batches.

The coordinator alone accesses SQLite. Its page cache is 8 MiB, mmap is disabled
and temporary sorting is configured for disk. Candidate and timing dictionaries
do not grow with stored row count. Transactions commit every 2048 native/ABG
writes and at every checkpoint and normal close, using SQLite's full synchronous
rollback-journal mode. Checkpoints record storage metadata rather than repeatedly
serializing the whole candidate pool. After a forced death, opening the database
rolls back an unfinished transaction; only committed rows are promised. A database
failure is a technical failure, never a successful exhausted search. Keep the
database and any SQLite journal together when recovering a failed run.

Disk usage grows with distinct candidates. Lossless storage does not make disk
space unlimited. Store the run on a filesystem with sufficient space; disk I/O
is included in measured stage and wall times. Neither linear scaling nor a
disk/RAM estimate for the entire two-hour search is guaranteed by short tests.

## Settings and outputs

- `max_intermediate_candidates: null` disables the old count stop for native
  candidates and specular ABG rows. It is now the default in `Settings` and all
  bundled configurations, including `revision_2h_32cpu.json`.
- A positive integer explicitly restores the legacy stopping cap for historical
  comparisons. Reaching it still records a partial search. It is not a RAM limit.
- `candidate_snapshot_max_rows: 10000` controls only small legacy NPZ/JSON exports.
  Above this count, these redundant exports are removed and the database remains
  the complete native-candidate checkpoint. This setting never stops enumeration,
  removes database rows or restricts finalization.
- The declared two-hour budget, stage reserves, seed/index bounds, native filters,
  corrected tail beam and 32-worker setting remain unchanged. The separate
  `max_solutions: 20000` finalization stop remains explicit and reports
  `solution_limit` if reached. It does not stop seed enumeration.
- `results.json.candidate_storage` records the database path, row counts, SQLite
  version, cache setting and database size at the final checkpoint. The final
  solutions, assignments, scores and reference-blind event journals remain in
  their existing formats. Deduplicated aliases point to the canonical sibling
  database without copying it, so moving the full run directory preserves links.

To inspect a database without materializing it in memory:

```python
from gidind_reference.candidate_store import CandidateStore
store = CandidateStore("path/to/native_candidates.sqlite3", create=False)
try:
    print(store.metadata())
    for candidate in store.view():  # Native-score order, stable insertion ties.
        print(candidate.native_scores)
        break
finally:
    store.close()
```

This opens an existing database and permits SQLite crash recovery; it never
creates or reuses a database for a fresh search. New indexing runs refuse an
existing database. There is no automatic search resume or solver retry.

## Restart the cluster workflow

Let the current array and collector finish before changing the checkout, or
explicitly cancel them before updating. Existing plans freeze the source hash
and settings and must not be reused with version 0.5. On the login node:

```sh
cd /path/to/gidind-ai
git pull --ff-only
bash GIDInd/python_port/submit_revision_2h.sh
```

The launcher creates new directories and runs the technical smoke gate before
six independent two-hour, 32-worker tasks. No references guide or stop searches.
The storage change allows more work than the former capped traversal, so new
counts, solutions and timings can differ. Compare completed identical work for
numerical parity; compare frozen budgets separately for benchmark outcomes.
