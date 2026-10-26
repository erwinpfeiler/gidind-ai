"""Build the four maintained reviewer notebooks; no computation occurs here."""

from pathlib import Path
import nbformat as nb

root = Path(__file__).resolve().parents[1]
setup = """from pathlib import Path
from gidind_ai.notebook_workflows import initialize
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
ROOT = initialize(ROOT)
DEVICE = "auto"  # CUDA when available, otherwise CPU
SEED = 42
"""
common = """Run `bash setup.sh` at the repository root, then open this notebook with the **GIDInd-AI (.venv)** kernel. Run cells from top to bottom. Figures are displayed inline and saved as SVG/PDF; raw records remain under `reviewer_outputs/`.

All inference uses the pinned `real_cases_model.ckpt`, serial float64 CG with L2 loss, and 10 one-to-one + 10 simulated-fractional matching rounds (up to 500 inner steps per round). No MCMC is run here. These are reproducible evaluations on the previously inspected frozen paper split, not a newly untouched estimate of generalization.
"""


def write(name, title, description, cells):
    notebook = nb.v4.new_notebook(
        cells=[
            nb.v4.new_markdown_cell(f"# {title}\n\n{description}\n\n{common}"),
            nb.v4.new_code_cell(setup),
            *cells,
        ]
    )
    notebook.metadata.kernelspec = {
        "display_name": "GIDInd-AI (.venv)",
        "language": "python",
        "name": "gidind-ai",
    }
    notebook.metadata.language_info = {"name": "python", "version": "3.11"}
    nb.write(notebook, root / "notebooks" / name)


write(
    "01_convergence.ipynb",
    "Synthetic convergence benchmark",
    "Default: 100 seeded cells covering all seven systems, **known true contact planes**, and a 0.056 Å⁻¹ cutoff. A cell succeeds only when every length error is <0.1 Å and every angle error is <0.1°. Failures stay in the denominator.",
    [
        nb.v4.new_code_cell("SAMPLES = 100  # larger preset: 10_000\n"),
        nb.v4.new_code_cell(
            "from gidind_ai.notebook_workflows import run_convergence\nsummary = run_convergence(ROOT, samples=SAMPLES, seed=SEED, device=DEVICE)"
        ),
    ],
)
write(
    "02_figure3bc.ipynb",
    "Figure 3b/c: robustness to missing, spurious and shifted peaks",
    "Default: **13 settings × 100 cells = 1,300 fits**. The paired cohort and contact planes are fixed across settings. Panel b changes the peak count from −20 to +20 in steps of five. Panel c uses five levels of independent zero-mean per-coordinate Gaussian jitter, σ=0…0.02 Å⁻¹. This is positional noise, not a common systematic calibration shift. Removal shortfalls and empty patterns are recorded. The full-grid preset uses 61 settings.",
    [
        nb.v4.new_code_cell(
            "SAMPLES = 100  # larger preset: 10_000\nFULL_GRID = False  # True: all counts and 21 shift levels (61 settings)\n"
        ),
        nb.v4.new_code_cell(
            "from gidind_ai.notebook_workflows import run_figure3\noutput = run_figure3(ROOT, samples=SAMPLES, seed=SEED, device=DEVICE, full_grid=FULL_GRID)"
        ),
    ],
)
training_common = """# Train a new model

Run `bash setup.sh` and use the GIDInd-AI (.venv) kernel. The default is a **two-epoch demonstration**, with 128 generated training cells per epoch, batch size eight and 64 development-validation cells. Training uses CUDA if available, otherwise CPU. Full training is a separate explicit preset (817 epochs, 319,447 planned updates).

The trainer also performs a final forward-only SWA batch-normalization refresh; it does not add optimizer updates. The demo has 32 planned optimizer updates. The preserved published GMM and its original tetragonal/monoclinic coordinate mapping are reused. Legacy validation is developmental; it is not a protected independent final test. The demonstration does not reproduce the historical released weights. The test split is not used for optimization or checkpoint selection. Outputs include a rolling checkpoint, manifests, CSV loss history and SVG/PDF plots. No refinement or MCMC is run in this notebook.
"""
n = nb.v4.new_notebook(
    cells=[
        nb.v4.new_markdown_cell(training_common),
        nb.v4.new_code_cell(setup),
        nb.v4.new_code_cell("FULL_TRAINING = False"),
        nb.v4.new_code_cell(
            "from gidind_ai.notebook_workflows import train_demo\noutput = train_demo(ROOT, full_training=FULL_TRAINING)"
        ),
    ]
)
n.metadata.kernelspec = {
    "display_name": "GIDInd-AI (.venv)",
    "language": "python",
    "name": "gidind-ai",
}
nb.write(n, root / "notebooks/03_training.ipynb")
write(
    "04_real_cases.ipynb",
    "Four experimental peak lists",
    "Each case has one execution cell below shared setup. All inputs are specular-free. All four use q_scale=0.90 and the same square q window below 1.80 inverse angstroms on both axes, with cutoff parameter 0.06 in scaled coordinates (physical cutoff 0.054 inverse angstroms). This window was selected using these four examples and is a developmental preset, not independently validated on new materials. Unlike the synthetic benchmark, all 49 candidate contact planes are searched. Ranking uses fit information only; reference cells never select a result. Rejected candidates remain visible. The requested common 10+10 schedule differs from some historical case presets, so the results need not exactly reproduce those earlier runs. Uncertainties are conditional local estimates, not calibrated confidence intervals.",
    [
        nb.v4.new_code_cell("from gidind_ai.notebook_workflows import run_real_case"),
        *[
            nb.v4.new_code_cell(f"{case} = run_real_case(ROOT, {case!r}, device=DEVICE)")
            for case in ["pq", "cu_bod_mof", "mapbbr3", "naib_s11"]
        ],
    ],
)
print("Built four reviewer notebooks")
