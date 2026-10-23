#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${GIDIND_PYTHON:-python3}"
if [[ "${1:-}" == "--python" ]]; then PYTHON="$2"; shift 2; fi
if [[ $# -gt 0 ]]; then echo 'Usage: bash setup.sh [--python python3.12]' >&2; exit 2; fi
"$PYTHON" -c 'import sys; assert (3,11) <= sys.version_info[:2] <= (3,13), "Python 3.11–3.13 is required"'
cd "$ROOT"
export HF_HOME="${HF_HOME:-$ROOT/.assets/hf-cache}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.assets/torch}"
if [[ ! -x .venv/bin/python ]]; then "$PYTHON" -m venv .venv; fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[models,paper,benchmark,reviewer,test]'
.venv/bin/python -m pip install -e GIDInd/python_port
.venv/bin/python -m ipykernel install --prefix "$ROOT/.venv" --name gidind-ai --display-name 'GIDInd-AI (.venv)'
# Every project download is public and explicitly disables implicit HF tokens.
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 .venv/bin/python -m gidind_ai.setup_assets --root "$ROOT/.assets"
.venv/bin/python -c 'import torch; print("CUDA available:", torch.cuda.is_available()); print("Setup complete. Run .venv/bin/jupyter lab notebooks")'
