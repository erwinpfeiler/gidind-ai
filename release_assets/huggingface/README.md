---
license: cc-by-sa-4.0
library_name: pytorch
tags:
  - crystallography
  - grazing-incidence-x-ray-diffraction
  - scientific-machine-learning
  - unit-cell-prediction
---

# gidind-ai publication checkpoint

This repository contains the PyTorch checkpoint used by **gidind-ai** to
propose crystallographic unit-cell candidates from grazing-incidence X-ray
diffraction (GIXD/GIWAXS) peak lists. The accompanying `gidind-ai` software
performs candidate generation followed by gradient/Laplace refinement; it does
not use TensorFlow or MCMC refinement.

## Checkpoint

| Field | Value |
| --- | --- |
| File | `default_model.ckpt` |
| Repository revision | `374aed64cf921eb47e3e03452c4337e0e636ff93` |
| SHA-256 | `b7efe2de22fdac9ece25b954634ffbc0a6df5570b01b80a2212a80cdae34dc2a` |
| Framework | PyTorch / PyTorch Lightning checkpoint |

The filename and revision above are the stable reviewer-facing release. Pin
this commit hash in reproducibility work rather than relying on the repository
default branch.

## Use with gidind-ai

Install the accompanying source package, including the optional Hugging Face
download support, then run an approved peak-list example:

```bash
python -m pip install '.[models]'
gidind-ai infer \
  --input examples/real_cases/pq.csv \
  --output-dir reviewer_outputs/pq \
  --device cpu
```

The command downloads this exact checkpoint automatically. To state the source
explicitly, use:

```bash
gidind-ai infer \
  --input examples/real_cases/pq.csv \
  --output-dir reviewer_outputs/pq \
  --model-repo epfeiler/gidind-ai \
  --revision 374aed64cf921eb47e3e03452c4337e0e636ff93 \
  --device cpu
```

`gidind-ai` writes a JSON result with checkpoint provenance, initial and
refined cells, Laplace uncertainties, convergence state, and matching metrics.
The associated repository also provides four paper-case wrappers and a small,
fixed-seed CPU benchmark.

## Intended use and limitations

This checkpoint is intended for research use with the supplied gidind-ai
workflow and peak-list format. It provides candidate unit cells, not a final
structural determination. Results depend on peak extraction, calibration,
orientation hypotheses, and the physical appropriateness of the experimental
sample; they should be checked by a domain expert. It is not a substitute for
crystallographic validation.

The checkpoint is a PyTorch/Lightning serialization file. Only load it from
this trusted, pinned release.

## License and citation

The model weights are released under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Reuse requires
attribution and distribution of adaptations under the same license. Please cite
the associated manuscript and the gidind-ai software release when they become
available.
