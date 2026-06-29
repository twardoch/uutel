#!/usr/bin/env bash
# install.sh — Install uutel in editable mode
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> Installing package (editable)..."
uv pip install --system --upgrade -e .

echo "==> Install complete."
