#!/usr/bin/env sh
set -eu
python -m pytest -q tests GIDInd/python_port/tests
python -m compileall -q src tests scripts GIDInd/python_port/gidind_reference
python -m build --wheel --no-isolation
