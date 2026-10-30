# GIDInd-AI

Predict unit-cell parameters from GIWAXS peak lists, then refine them with PyTorch. Specular reflections and a known crystal system are not required for experimental inference.

**Reviewer release: v1.0.0.** Start with the four notebooks below. They use the public, checksum-verified `real_cases_model.ckpt` and frozen data assets; no Hugging Face account or token is required.

## Set up once

Requirements: Git, Python **3.11–3.13**, internet during setup, and approximately 5 GB of free space for CPU setup (CUDA wheels can require more). macOS and Linux are supported. An NVIDIA CUDA installation/driver compatible with your PyTorch build enables GPU execution; CPU execution works too.

```bash
git clone https://github.com/erwinpfeiler/gidind-ai.git
cd gidind-ai
bash setup.sh  # or: bash setup.sh --python python3.12
.venv/bin/jupyter lab notebooks
```

Setup creates `.venv`, installs the package and conventional GIDInd Python port, registers the **GIDInd-AI (.venv)** kernel, downloads both released models and required data into `.assets/`, and verifies their checksums. It also prefetches the original torchvision initialization used for training. Repeating setup reuses verified assets. No MATLAB license/runtime is required.

PyTorch uses CUDA when available and otherwise CPU; these notebooks do not use Apple's MPS backend. To use a different CUDA wheel, install the appropriate official PyTorch/torchvision pair in `.venv` before rerunning setup. Check with:

```bash
.venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

## Run the notebooks

Select **GIDInd-AI (.venv)** and run cells from top to bottom.

| Notebook | Default run | Main output |
|---|---|---|
| [01 — Convergence](notebooks/01_convergence.ipynb) | 100 seeded cells, all seven systems; known contact plane | NN-only/refined recovery and length/angle error violins |
| [02 — Figure 3b/c](notebooks/02_figure3bc.ipynb) | 100 paired cells × 13 perturbation settings: **1,300 fits** | Missing/spurious-peak and positional-noise convergence curves |
| [03 — Training](notebooks/03_training.ipynb) | Two small training epochs; CUDA if available | Training/validation loss, checkpoint and run manifest |
| [04 — Real cases](notebooks/04_real_cases.ipynb) | One execution cell each for PQ, Cu-BOD, MAPbBr₃ and NAIB-S11 | Ranked unit cells, uncertainties and measured/simulated peak overlays |

The Figure 3 preview can take tens of minutes on a laptop; it is 1,300 fits, not 100. Its `FULL_GRID=True` preset uses 61 settings. Set `SAMPLES = 10_000` for the larger convergence preset; increase sample counts deliberately. Nondefault counts, seeds and the full Figure 3 grid get separate output directories. Training is a demonstration, not a replacement for the released checkpoint; its full preset plans 817 epochs / 319,447 optimizer updates. A final forward-only SWA batch-normalization refresh adds no optimizer updates.

All inference notebooks use **serial float64 CG, L2 loss, 10 one-to-one matching rounds followed by 10 simulated-fractional rounds**, with up to 500 optimizer iterations per round. Synthetic cutoff: **0.056 Å⁻¹**. All four experimental profiles use **q_scale = 0.90**, a common square window below **1.80 Å⁻¹** on each axis, and matching cutoff parameter **0.06 in scaled coordinates** (physical cutoff **0.054 Å⁻¹**). All experimental inputs are specular-free; all 49 contact-plane candidates are evaluated. MCMC is never used in the notebooks.

Synthetic recovery requires **every length error <0.1 Å and every angle error <0.1°**. Failed fits remain in the denominator. This known-orientation benchmark measures a different task from searching experimental orientations.

### Saved outputs

`reviewer_outputs/` contains per-cell/condition data, settings, checkpoint/asset identities, manifests and **SVG/PDF** figures. Figure 3 stores one result per fit in SQLite and can resume with identical settings; changed settings require a new output directory. Training writes rolling checkpoints; use the saved configuration with `python -m gidind_ai.training --config CONFIG --resume latest` to resume.

The shared experimental window was selected using these four examples; it is a developmental preset, not an independently validated default for arbitrary materials. It retains 24/74 PQ, 10/10 Cu-BOD, 4/12 MAPbBr₃ and 20/87 NAIB-S11 input peaks. The packaged `release_models.json` defines current executable profiles; the immutable input-asset manifest preserves historical preprocessing metadata.

Real-case candidates remain visible when rejected. A low peak-position loss or a local uncertainty alone does not prove a uniquely identified cell. Local uncertainties are conditional estimates, not calibrated confidence intervals. The new common 10+10 schedule differs from some historical experimental presets.

## Python and command-line use

The notebooks call shared, importable functions in `gidind_ai.notebook_workflows`. Existing inference/benchmark APIs, `RefinementSettings`, explicit checkpoint selection and COD preparation remain available.

```bash
.venv/bin/gidind-ai infer \
  --input examples/real_cases/pq_without_specular.csv \
  --case-profile pq --refinement-backend serial \
  --output-dir reviewer_outputs/pq_cli
```

### MCMC (optional; outside notebooks)

```bash
.venv/bin/gidind-ai infer \
  --input examples/real_cases/pq_without_specular.csv \
  --release-model high_missingness --refinement-algorithm mcmc \
  --q-sigma 0.005 --device cpu --output-dir reviewer_outputs/pq_mcmc
```

`--q-sigma` must be replaced by the measured physical coordinate uncertainty in Å⁻¹. CUDA can accelerate sampling. Read the [MCMC assumptions and diagnostics](docs/mcmc_refinement.md) before interpreting posterior output. [Batched refinement](docs/refinement_backends.md) remains explicitly selectable; notebook results use the reference serial optimizer.

### Conventional GIDInd Python port

Setup also installs `gidind-reference`. See its [usage](GIDInd/python_port/README.md) and [provenance](GIDInd/python_port/PROVENANCE.md). This is an adapted headless Python implementation; equivalence to the original MATLAB executable is not established. Original GIDInd is credited to M. Kainz and J. Simbrunner. MATLAB binaries and saved comparison runs are not bundled.


## License

GIDInd-AI: **AGPL-3.0-only**. Model weights: **CC BY-SA 4.0**. COD-derived data: **CC0**, as documented in the asset manifest. The conventional GIDInd port remains **GPL-3.0-or-later**. Bundled Niggli code retains its **MIT** license. See [third-party notices](THIRD_PARTY_NOTICES.md).
