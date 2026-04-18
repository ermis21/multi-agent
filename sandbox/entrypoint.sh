#!/bin/bash
set -e

# Create Python venv in workspace on first run so agents can install packages
# and run Python experiments without affecting the container's system Python.
VENV="${WORKSPACE_DIR:-/workspace}/venv"
if [ ! -f "$VENV/bin/activate" ]; then
    echo "[sandbox] Creating Python venv at $VENV..."
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --quiet --upgrade pip
    echo "[sandbox] Venv ready."
fi

exec "$@"
