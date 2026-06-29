#!/usr/bin/env bash
# publish.sh — Build, install, bump version, and publish uutel to PyPI
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> Running build..."
bash "$SCRIPT_DIR/build.sh"

echo "==> Running install..."
bash "$SCRIPT_DIR/install.sh"

echo "==> Bumping version via gitnextver..."
uvx gitnextver@latest

echo "==> Building sdist + wheel..."
uvx hatch build

echo "==> Publishing to PyPI..."
uv publish

echo "==> Done. Published $(uvx hatch version)"
