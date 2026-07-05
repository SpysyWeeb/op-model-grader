#!/usr/bin/env bash
# op-model-grader launcher. Despite the filename, this also runs on macOS
# (both ship bash and python3) -- named for Linux since that's this repo's
# primary target; start_windows.bat is the Windows counterpart.
# First run creates a local .venv and installs the tool; after that it just starts.
# No arguments -> opens the desktop UI. Any arguments are passed to the CLI instead.
set -e
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 was not found. Install Python 3.10+ from your package manager or python.org." >&2
  exit 1
fi

if ! .venv/bin/python -c "import opgrader" >/dev/null 2>&1; then
  echo "First run: setting up (this can take a minute)..."
  [ -x .venv/bin/python ] || python3 -m venv .venv
  .venv/bin/python -m pip install --quiet --upgrade pip
  .venv/bin/python -m pip install --quiet -e .
fi

if [ $# -eq 0 ]; then
  set -- --ui
fi
exec .venv/bin/python -m opgrader "$@"
