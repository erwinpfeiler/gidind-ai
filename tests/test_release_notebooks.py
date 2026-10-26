"""Reviewer release contracts: real optimizer schedule, notebooks and device routing."""

from pathlib import Path
from types import SimpleNamespace
import json
import torch

ROOT = Path(__file__).resolve().parents[1]


def test_exactly_four_notebooks_and_four_case_execution_cells():
    paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert len(paths) == 4
    for path in paths:
        notebook = json.loads(path.read_text())
        code = "\n".join(
            "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"
        )
        assert "mcmc" not in code.lower()
        assert "/Users/" not in code
        assert notebook["metadata"]["kernelspec"]["name"] == "gidind-ai"
    notebook = json.loads(paths[-1].read_text())
    assert sum("run_real_case(ROOT" in "".join(c["source"]) for c in notebook["cells"]) == 4


def test_preview_is_1300_fits_and_schedule_is_explicit():
    from gidind_ai.figure3_benchmark import condition_grid
    from gidind_ai.notebook_workflows import _args

    args = _args("/tmp/demo", 100, 42, "cpu")
    assert len(condition_grid(args)) * args.samples == 1300
    assert args.outer_loops == 20 and args.maxiter == 500 and args.q_cutoff == 0.056


def test_actual_serial_hybrid_calls_ten_one2one_then_ten_fractional(monkeypatch):
    from gixd_data_generator import GixdDataGenerator
    import polishing_fit as module
    from gidind_ai.batched_refinement import Geometry, simulate

    generator = GixdDataGenerator("uniform", "testing", load_unit_cell_generator=False)
    cell = torch.tensor([8.0, 9.0, 10.0, 82.0, 85.0, 87.0], dtype=torch.float64)
    geometry = Geometry.from_generator(generator, "cpu")
    observations = simulate(cell, torch.tensor([0, 0, 1]), geometry.laue)
    observations = observations[
        (observations[:, 0] > 0.05) & (observations[:, 0] < 2) & (observations[:, 1] < 2)
    ]
    calls = []
    original = module.match_laue_indices

    def match(*args, **kwargs):
        calls.append(kwargs.get("method", args[6] if len(args) > 6 else None))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "match_laue_indices", match)
    monkeypatch.setattr(
        module,
        "torch_minimize",
        lambda fun, x, **kw: SimpleNamespace(x=x.clone(), fun=fun(x).item(), nit=0, status=0),
    )
    module.direct_inversion_pytorch(
        cell,
        observations[:, 0],
        observations[:, 1],
        (0, 0, 1),
        generator,
        method="cg",
        q_cutoff=0.056,
        matching_method="hybrid",
        outer_loops=20,
        maxiter=500,
        return_uncertainties=False,
        refinement_backend="serial",
    )
    assert calls[:10] == ["one2one"] * 10
    assert calls[10:20] == ["simulated_fractional"] * 10


def test_training_uses_cuda_or_cpu_and_safe_precision():
    source = (ROOT / "src/gixd_regressor.py").read_text()
    assert 'accelerator="gpu" if torch.cuda.is_available() else "cpu"' in source
    assert "torch.cuda.is_bf16_supported()" in source
