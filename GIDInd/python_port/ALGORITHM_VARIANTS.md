# Corrected and upstream-compatibility searches

Version 0.2 defaults to `--algorithm corrected`. Both `index` and `benchmark`
accept `--algorithm corrected|upstream`; the CLI overrides JSON's `algorithm`.
The same override is supported by `overnight.sh`, the SLURM array submission
helper, and `python -m gidind_reference.slurm_array prepare`. The prepared array
freezes the choice. Both paths retain float64, discrete final orientations,
single-process or ordered spawned workers, input validation, deadlines and
checkpoint behavior.

`--algorithm upstream` preserves the **previous port's upstream search choices**,
including the greedy qz-only fifth/sixth HKL selection and its zero-HKL trap.
Its completed-work counts, candidate arrays and output order are checked against
the pre-multiprocessing numerical fixtures. It is not certified exact MATLAB:
the numerical, reduction, input and discrete-orientation adaptations documented
in PROVENANCE.md still apply. The original MATLAB files are unchanged.

## Corrected triclinic search without specular signal

The four-HKL enumeration and reciprocal-metric equations are retained. The
corrected path makes these explicit changes:

1. Reject four-HKL systems whose four metric-design rows are not independent.
   Two more rows cannot complete six independent metric coefficients in that
   case. This also excludes a zero HKL among those first four rows.
2. Exclude four-observation choices with all qz=0 from orientation seeding.
   Such a choice fits projection=0 and makes the qz-only tail heuristic
   uninformative. In-plane observations remain in all-peak fitting and scoring.
3. Exclude `(0,0,0)` from fifth/sixth HKL choices. A nonzero measured reflection
   cannot be indexed by the reciprocal origin.
4. Keep the original best four-observation choice and best observed tail-pair
   choice, but evaluate **up to K nonzero HKLs for each tail peak**, ordered by
   qz error. Evaluate their Cartesian product in stable increasing summed qz
   error. `tail_hkl_candidates` is a positive JSON integer, default **8**;
   each usable quartet therefore produces at most **64** tail systems.
   Singular and nonpositive systems are rejected, and every surviving system
   goes through the existing all-peak indexing/refit and subsequent ranking.
5. Anchor the no-specular relative native-score window at the first candidate
   that passes final integer-plane residual checks. The previous port anchors
   it at the best continuous projection, even if that candidate has no accepted
   integer orientation. That can discard all useful discrete solutions before
   they are evaluated. The relative-window width is unchanged; candidates are
   still processed in native-score order. This fixes an interaction with the
   port's discrete-orientation adaptation, not a demonstrated MATLAB defect.

No known cell, reference HKL, neural output or sample-name branch is supplied to
this search. It uses the original measured seed selection; qz=0 observations
are not removed from the input. Error thresholds are unchanged. The specular
search has no newly identified tail-selection fix and is numerically unchanged
between the two modes.

This is a bounded alternative search, not a claim that every possible assignment
is enumerated or every sample converges. K is a search parameter, not an exact
tie tolerance: nearby alternatives can have different errors. Increasing K
does more work and can change the candidate set. The best four-observation and
tail-pair heuristics remain. Candidate/solution/time caps also still apply;
in particular this change does not remove the specular ABG candidate cap.

## Records and interpretation

`effective_settings` records `algorithm` and `tail_hkl_candidates`; top-level
`algorithm_variant` and `algorithm_note` identify the implementation. The port
version, source hash/snapshot, and mode are retained. Legacy records with no
algorithm setting describe the previous upstream-compatible implementation.
Search work records the variant and bounded tail-search scope.

Version 0.5 removes the port's default intermediate count stops in both variants
and stores native candidates and retained specular ABG rows losslessly on disk.
It retains the filters/window described above and stable score ordering, without
top-N eviction. Explicit positive legacy caps remain available; bundled configs
use null. See [CANDIDATE_STORAGE.md](CANDIDATE_STORAGE.md).

`without_specular_hkl_quadruples` still counts committed four-HKL attempts,
including invalid/zero combinations in the configured enumeration. Corrected
`unusable_four_hkl` counts quartets that cannot produce usable seed systems;
`corrected_tail_systems` counts expanded six-HKL systems. The singular-six,
nonpositive-metric and refit rejection counts concern these expanded systems,
so they need not sum to the number of four-HKL attempts. Report both four-HKL
attempts and tail-system evaluations when comparing work and throughput.

Fixed-work serial/multicore parity is tested **within each variant**. Variants
are expected to have different work, candidates and runtime. Benchmark tables
must distinguish the corrected port from the upstream-compatible port; neither
timing should be labeled a measurement of the original MATLAB executable.

For an ESRF array, after prior arrays and collectors finish:

```sh
bash GIDInd/python_port/submit_esrf_array.sh smoke --algorithm corrected
bash GIDInd/python_port/submit_esrf_array.sh full --algorithm corrected
# Separate run and directory for the upstream-compatibility comparison:
bash GIDInd/python_port/submit_esrf_array.sh full --algorithm upstream
```

Do not change the checkout or environment while a prepared array or its
collector is active. The current allocation is six cases with 32 workers each and a two-hour full
budget (updated on 22 October 2026); see RUNTIME_REPAIR.md. Existing JSON files without an algorithm field now select the
corrected default. Specify the variant explicitly for paper benchmark runs.
