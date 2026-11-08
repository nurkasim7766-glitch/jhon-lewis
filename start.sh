#!/usr/bin/env bash
# Linux/macOS launcher: creates .venv on first run, installs deps, starts the web app.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "First run: creating virtual environment ..."
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip >/dev/null
  echo "Installing dependencies ..."
  .venv/bin/python -m pip install -e ".[dev]"
fi
.venv/bin/python -c "import facepipe.web.app" 2>/dev/null || .venv/bin/python -m pip install -e ".[dev]"

PORT="${FACEPIPE_PORT:-8765}"
echo "Starting facepipe on http://127.0.0.1:${PORT}/  (Ctrl+C to stop)"
exec .venv/bin/python -m facepipe.web --port "$PORT" "$@"
