#!/bin/sh
set -eu
port_root="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if [ ! -x "$port_root/.venv/bin/python" ]; then
    echo "Create the environment first: python3 -m venv '$port_root/.venv'" >&2
    echo "Then install: '$port_root/.venv/bin/python' -m pip install -e '$port_root'" >&2
    exit 1
fi
exec "$port_root/.venv/bin/python" -m gidind_reference "$@"
