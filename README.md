# GIDInd-AI

GIDInd-AI predicts unit-cell parameters from grazing-incidence wide-angle X-ray scattering (GIWAXS) peak lists and refines them with PyTorch. It works directly on experimental peak lists: specular reflections and a known crystal system are **not** required.

For the methodological details, please refer to the paper.

## Setup

Requirements: Git, Python 3.11–3.13, internet during setup, and about 5 GB of free disk space. macOS and Linux are supported. GIDInd-AI runs on CPU by default and uses CUDA automatically when an NVIDIA GPU is available.

```bash
git clone https://github.com/erwinpfeiler/gidind-ai.git
cd gidind-ai
bash setup.sh  # or: bash setup.sh --python python3.12
.venv/bin/jupyter lab notebooks
```

The setup script creates a virtual environment, installs the package together with the conventional GIDInd Python port, registers the **GIDInd-AI (.venv)** Jupyter kernel, and downloads the released models and data (checksum-verified). No Hugging Face account and no MATLAB license are required.

## The notebooks

In Jupyter, select the **GIDInd-AI (.venv)** kernel and run the cells from top to bottom.

| Notebook | What it shows |
|---|---|
| [01 — Convergence](notebooks/01_convergence.ipynb) | Unit-cell recovery from synthetic data |
| [02 — Figure 3b/c](notebooks/02_figure3bc.ipynb) | Robustness to missing/spurious peaks and positional noise |
| [03 — Training](notebooks/03_training.ipynb) | A small demonstration training run |
| [04 — Real cases](notebooks/04_real_cases.ipynb) | Unit-cell inference for four experimental datasets |

All results — per-run data, figures (SVG/PDF), settings and manifests — are written to `reviewer_outputs/`. Runtime presets can be adjusted in the notebooks (e.g. larger sample counts); non-default settings get their own output directories.

## Python and command-line use

The notebooks build on importable functions in `gidind_ai.notebook_workflows`; everything is also usable from Python or the command line:

```bash
.venv/bin/gidind-ai infer \
  --input examples/real_cases/pq_without_specular.csv \
  --case-profile pq --refinement-backend serial \
  --output-dir reviewer_outputs/pq_cli
```

An optional MCMC refinement backend is available for posterior sampling (see [docs/mcmc_refinement.md](docs/mcmc_refinement.md)); alternative refinement backends are described in [docs/refinement_backends.md](docs/refinement_backends.md).

Setup also installs `gidind-reference`, a headless Python port of the original GIDInd tool ([usage](GIDInd/python_port/README.md)).

## License

GIDInd-AI is licensed **AGPL-3.0-only**; model weights under **CC BY-SA 4.0**; COD-derived data under **CC0**; the conventional GIDInd port under **GPL-3.0-or-later**; bundled Niggli code retains its **MIT** license. See [third-party notices](THIRD_PARTY_NOTICES.md).
