# Runtime control

CLI indexing runs inside an isolated subprocess with a wall-time watchdog.
The supervisor records child exit codes/signals and explicitly labels timeouts,
interruptions and technical failures. It does not convert them to successful
search exhaustion or silently retry the solver. Child processes use isolated
process groups on POSIX systems; cleanup runs once per supervised shutdown.

The solver checkpoints compact state and journals complete validated fits.
Cooperative cancellation drains ordered committed work; the watchdog can stop
non-returning coordinators and descendants. Complete solution journals remain
available after an interrupted run. Timing includes process startup, numerical
work, checkpoints, output and process exit; internal event timestamps retain
their separately documented coordinator scope.

Fatal diagnostics inspect the faulting thread. Asynchronous timed/all-thread
stack walks are not used. These runtime controls do not change numerical
cutoffs or supply an automatic criterion for knowing the physical answer.

Cluster launchers are optional templates. Adapt partition directives, paths,
CPU counts and budgets to the target system, then run a technical smoke test.
No actual cluster execution is claimed by the local shell-dispatch tests.
