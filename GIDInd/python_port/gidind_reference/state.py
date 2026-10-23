"""Bounded deterministic search state, budgets and progress checkpoints."""
from collections import Counter
from dataclasses import dataclass
import time


class SearchLimit(Exception):
    pass


@dataclass
class Candidate:
    metric: object
    hkl: object
    plane: object = None
    projection: object = None
    native_scores: dict = None
    timing: dict = None


class State:
    def __init__(self, settings, checkpoint=None, *, started=None, candidate_path=None):
        self.settings = settings
        self.started = time.perf_counter() if started is None else started
        self.deadline = self.started + settings.time_limit_seconds
        self.search_deadline = self.deadline - min(60., .15*settings.time_limit_seconds)
        self.counts = Counter()
        self.stages = {}
        self.stage = "initialization"
        self.stage_started = self.started
        self.partial_reasons = []
        from .candidate_store import CandidateStore
        self.pending = CandidateStore(candidate_path) if candidate_path is not None else []
        self.disk_candidates = isinstance(self.pending, CandidateStore)
        self.pending_revision = 0
        self.pending_keys = {}
        self.solutions = []
        self.first_solution_wall_seconds = None
        self.checkpoint = checkpoint
        self.last_checkpoint = self.started
        self.interrupted = False
        self.checkpoint_compute_io = 0.
        self.candidate_timings = {}
        self.next_candidate_id = 0
        self.solution_event_count = 0
        self.on_solution_event = None

    def _record_candidate(self, candidate, previous=None, previous_timing=None):
        now = time.perf_counter()-self.started
        if previous_timing is None:
            previous_timing = self.candidate_timings.pop(id(previous),{}) if previous is not None else {}
        candidate.timing = {
            'candidate_id': self.next_candidate_id,
            'discovered_wall_seconds': now,
            'assignment_first_seen_wall_seconds': previous_timing.get('assignment_first_seen_wall_seconds',now),
        }
        if not self.disk_candidates:
            self.candidate_timings[id(candidate)] = candidate.timing
        self.next_candidate_id += 1
        self.pending_revision += 1

    def record_solution(self, candidate, slot, solution, operation):
        """Reference-blind timeline; no evaluator or correctness test runs here."""
        event = dict(sequence=self.solution_event_count,wall_seconds=time.perf_counter()-self.started,
            solution_slot=slot,operation=operation,
            solution={k:solution[k] for k in ('cell','contact_plane','rmse','volume_angstrom3',
                                             'max_combined_peak_error')},
            candidate_timing=candidate.timing or self.candidate_timings.get(id(candidate)))
        self.solution_event_count += 1
        if self.on_solution_event is not None:
            self.on_solution_event(event)

    def set_stage(self, name):
        now = time.perf_counter()
        self.stages[self.stage] = self.stages.get(self.stage, 0.) + now - self.stage_started
        self.stage, self.stage_started = name, now

    def check(self, deadline=None):
        now = time.perf_counter()
        if now >= (self.deadline if deadline is None else min(deadline, self.deadline)):
            raise SearchLimit("time_limit")
        if self.checkpoint and now - self.last_checkpoint >= self.settings.checkpoint_seconds:
            # Measure the next interval from completion. A slow checkpoint must
            # not cause another full checkpoint at the very next candidate.
            try:
                self.checkpoint(self)
            finally:
                self.last_checkpoint = time.perf_counter()
            if self.last_checkpoint >= (self.deadline if deadline is None else min(deadline,self.deadline)):
                raise SearchLimit('time_limit')

    def partial(self, reason):
        if reason not in self.partial_reasons:
            self.partial_reasons.append(reason)

    def add_pending(self, candidate):
        if self.disk_candidates:
            import json
            from .candidate_store import assignment_key, score
            key = assignment_key(candidate)
            previous = self.pending.lookup(key)
            if previous is not None:
                slot, old_score, timing = previous
                if score(candidate) < old_score:
                    self._record_candidate(candidate, previous_timing=json.loads(timing))
                    self.pending.put(slot, key, candidate)
                self.counts['duplicate_native_assignments'] += 1
            else:
                self._check_candidate_cap()
                self._record_candidate(candidate)
                self.pending.put(len(self.pending), key, candidate)
            return
        key = (candidate.hkl.astype('<i4').tobytes(),
               None if candidate.plane is None else tuple(candidate.plane))
        if key in self.pending_keys:
            i = self.pending_keys[key]
            score = candidate.native_scores.get("seed_linear_sumerr", candidate.native_scores.get("seed_qxyz_rmse", 0.))
            old = self.pending[i].native_scores
            if score < old.get("seed_linear_sumerr", old.get("seed_qxyz_rmse", 0.)):
                self._record_candidate(candidate,self.pending[i])
                self.pending[i] = candidate
            self.counts["duplicate_native_assignments"] += 1
            return
        self._check_candidate_cap()
        self.pending_keys[key] = len(self.pending)
        self._record_candidate(candidate)
        self.pending.append(candidate)

    def _check_candidate_cap(self):
        cap = self.settings.max_intermediate_candidates
        if cap is not None and len(self.pending) >= cap:
            self.partial("intermediate_candidate_limit")
            raise SearchLimit("intermediate_candidate_limit")
