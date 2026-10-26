"""Explicit controls, initialized from the MATLAB app properties/dialog defaults."""
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path


@dataclass
class Settings:
    algorithm: str = "corrected"
    tail_hkl_candidates: int = 8
    # Main app properties; dialog has volume lower bound 1 rather than 50.
    length_bounds: list = field(default_factory=lambda: [[3., 30.] for _ in range(3)])
    angle_bounds: list = field(default_factory=lambda: [[60., 120.] for _ in range(3)])
    volume_bounds: list = field(default_factory=lambda: [50., 100000.])
    without_specular_volume_bounds: list = field(default_factory=lambda: [1., 100000.])
    specular_seed_peaks: int = 3
    seed_peak_rows: list | None = None
    specular_seed_hk: int = 3
    specular_index_hk: int = 6
    specular_l: int = 6
    specular_uv: int = 2
    specular_w: int = 3
    redundancy_hk: int = 4
    specular_cutoff: float = .01
    qxy_stage_cutoff: float = .05
    rmse_cutoffs: list = field(default_factory=lambda: [.05, .05, .05])  # xy,z,xyz
    without_specular_seed_hkl: list = field(default_factory=lambda: [1, 1, 2])
    without_specular_index_hkl: list = field(default_factory=lambda: [3, 3, 3])
    without_specular_seed_peaks: int = 8
    without_specular_max_peak_error: float = .1
    without_specular_relative_window: float = .2
    specular_filter_cutoff: float = .05
    plane_limit: int = 2
    planes: list | None = None
    candidate_order: str = "area"
    workers: int = 1
    batch_size: int = 2048
    time_limit_seconds: float = 7200.
    max_solutions: int = 20000
    max_intermediate_candidates: int | None = None  # Optional legacy stopping cap; disabled by default.
    candidate_snapshot_max_rows: int = 10000  # Maximum rows in compatibility NPZ exports, not search pruning.
    checkpoint_seconds: float = 15.

    def validate(self):
        if self.algorithm not in ("upstream", "corrected"):
            raise ValueError("algorithm must be upstream or corrected")
        for name in ("length_bounds", "angle_bounds"):
            value = getattr(self, name)
            if len(value) != 3 or any(len(p) != 2 or not all(math.isfinite(x) for x in p)
                                     or p[0] <= 0 or p[0] > p[1] for p in value):
                raise ValueError(f"Invalid {name}")
        if any(p[1] >= 180 for p in self.angle_bounds):
            raise ValueError("Angles must be strictly between 0 and 180 degrees")
        for name in ("volume_bounds", "without_specular_volume_bounds"):
            p = getattr(self, name)
            if len(p) != 2 or not all(math.isfinite(x) for x in p) or not 0 < p[0] <= p[1]:
                raise ValueError(f"Invalid {name}")
        positive_ints = ("specular_seed_peaks", "specular_seed_hk", "specular_index_hk",
                         "specular_l", "specular_uv", "specular_w", "redundancy_hk",
                         "without_specular_seed_peaks", "plane_limit", "batch_size",
                         "max_solutions", "candidate_snapshot_max_rows", "workers", "tail_hkl_candidates")
        for name in positive_ints:
            x = getattr(self, name)
            if isinstance(x, bool) or not isinstance(x, int) or x < 1:
                raise ValueError(f"{name} must be a positive integer")
        x = self.max_intermediate_candidates
        if x is not None and (type(x) is not int or x < 1):
            raise ValueError('max_intermediate_candidates must be null or a positive integer')
        for name in ("without_specular_seed_hkl", "without_specular_index_hkl"):
            value = getattr(self, name)
            if len(value) != 3 or any(type(x) is not int or x < 0 for x in value) or max(value) == 0:
                raise ValueError(f"Invalid {name}")
        for name in ("specular_cutoff", "qxy_stage_cutoff", "time_limit_seconds",
                     "checkpoint_seconds", "without_specular_max_peak_error"):
            x = getattr(self, name)
            if not math.isfinite(x) or x <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if len(self.rmse_cutoffs) != 3 or any(not math.isfinite(x) or x <= 0 for x in self.rmse_cutoffs):
            raise ValueError("rmse_cutoffs must contain three positive finite values")
        for name in ("specular_filter_cutoff", "without_specular_relative_window"):
            x = getattr(self, name)
            if not math.isfinite(x) or x < 0:
                raise ValueError(f"Invalid {name}")
        if self.specular_seed_peaks < 3 or self.without_specular_seed_peaks < 6:
            raise ValueError("Need at least three specular or six no-specular seed peaks")
        if self.candidate_order not in ("area", "qxy_error"):
            raise ValueError("candidate_order must be area or qxy_error")
        if self.seed_peak_rows is not None and (len(set(self.seed_peak_rows)) != len(self.seed_peak_rows)
                or any(type(x) is not int or x < 1 for x in self.seed_peak_rows)):
            raise ValueError("seed_peak_rows must contain unique positive source-file row numbers")
        if self.planes is not None:
            if not self.planes or any(len(p) != 3 or any(type(x) is not int for x in p)
                                      or not any(p) for p in self.planes):
                raise ValueError("planes must be a nonempty list of nonzero integer triples")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path=None, **overrides):
        data = json.loads(Path(path).read_text()) if path else {}
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data).validate()
