#!/bin/sh
set -eu
port_root="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
python_bin="${PYTHON_BIN:-python3}"
if ! "$python_bin" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)'; then
    echo "The pinned environment requires Python 3.12 or newer. Set PYTHON_BIN to a suitable interpreter." >&2
    exit 1
fi
"$python_bin" -m venv "$port_root/.venv"
# Exact laptop dependency versions; Python 3.12 or newer is needed for this lock.
"$port_root/.venv/bin/python" -m pip install -r "$port_root/provenance/environment.lock.txt"
"$port_root/.venv/bin/python" -m pip install --no-deps --no-build-isolation -e "$port_root"
echo "Ready: $port_root/run.sh --help"
