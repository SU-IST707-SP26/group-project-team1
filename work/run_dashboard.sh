#!/usr/bin/env bash
# run_dashboard.sh — launch the 4th & Context Shiny dashboard in a Codespace.
#
# Usage (from anywhere in the repo):
#   bash run_dashboard.sh
#
# What it does:
#   1. Confirms the CSV is where constants.py expects it
#   2. Installs Python dependencies (skips if already installed)
#   3. Launches the Shiny app on port 8000
#
# Stop the app with Ctrl+C.

set -e  # exit on first error

# ---------------------------------------------------------------------------
# Resolve paths so the script works no matter where you run it from
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$SCRIPT_DIR/fourth_and_context"
CSV_PATH="$SCRIPT_DIR/encoded_fourth_downs.csv.gz"

# Pretty output helpers
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
RESET='\033[0m'

info()  { echo -e "${GREEN}==>${RESET} $1"; }
warn()  { echo -e "${YELLOW}!! ${RESET} $1"; }
fail()  { echo -e "${RED}xx ${RESET} $1" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. Sanity checks
# ---------------------------------------------------------------------------
if [[ ! -d "$APP_DIR" ]]; then
    fail "fourth_and_context/ directory not found at $APP_DIR"
fi

if [[ ! -f "$CSV_PATH" ]]; then
    fail "encoded_fourth_downs.csv.gz not found at repo root.
       Expected: $CSV_PATH"
fi

info "Repo layout looks good"

# ---------------------------------------------------------------------------
# 2. Install dependencies (only if something is missing)
# ---------------------------------------------------------------------------
cd "$APP_DIR"

# Quick check: try importing each top-level package. If any fails, run pip.
if python -c "import shiny, xgboost, sklearn, pandas, numpy, matplotlib, plotly" 2>/dev/null; then
    info "Dependencies already installed — skipping pip"
else
    info "Installing dependencies (~2 min on first run)..."
    pip install -q -r requirements.txt || fail "pip install failed"
fi

# ---------------------------------------------------------------------------
# 3. Launch
# ---------------------------------------------------------------------------
info "Launching Shiny app on port 8000..."
info "First launch trains models (~30-60s). Watch for 'Uvicorn running' below,"
info "then click the 'Open in Browser' popup (or use the PORTS tab)."
info "Press Ctrl+C to stop."
echo ""

exec python -m shiny run app.py --host 0.0.0.0 --port 8000
