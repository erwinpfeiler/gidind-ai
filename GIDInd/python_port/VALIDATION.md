# Port validation

Run the installed AI/port environment's tests from the repository root:

```bash
.venv/bin/python -m pytest GIDInd/python_port/tests
```

Tests cover coordinate parsing and specular selection; reciprocal geometry;
MATLAB enumeration and rounding; seed equations and corrected-tail counterexamples;
cell reduction and transformed assignments; full small synthetic recovery;
serial/parallel completed-work parity; candidate storage; result persistence;
reference-blind timing; process cancellation/watchdogs; and shell dispatch with
mock scheduler commands.

The minimal fixtures include one original PQ spreadsheet export, targeted
MAPbBr3 seed traces, a finalization example and a frozen serial-work signature.
Frozen numerical hashes are environment-specific; other platforms compare their
own serial and parallel runs directly. Saved research/cluster runs are not
included. Shell tests do not constitute execution on a SLURM cluster.

Original MATLAB executable equivalence remains unverified. Synthetic recovery
and exported examples are evidence for this adapted implementation, not proof
of complete upstream parity. Physical CUDA tests do not apply to this CPU port.
