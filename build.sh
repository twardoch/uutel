#!/usr/bin/env bash
# build.sh — Lint, format, and build uutel
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> Cleaning previous builds..."
uvx hatch clean

echo "==> Formatting and linting..."
fd -e py -x uvx autoflake -i {}
fd -e py -x uvx pyupgrade --py312-plus {}
fd -e py -x uvx ruff check --output-format=github --fix --unsafe-fixes {}
fd -e py -x uvx ruff format --respect-gitignore --target-version py312 {}
uvx hatch fmt

echo "==> Building sdist + wheel..."
uvx hatch build

echo "==> Build complete."
