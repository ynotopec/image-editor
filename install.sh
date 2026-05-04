#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "$PROJECT_DIR")"
VENV_DIR="${HOME}/venv/${PROJECT_NAME}"

if ! command -v uv >/dev/null 2>&1; then
  echo "[install] Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

mkdir -p "$(dirname "$VENV_DIR")"

if [ ! -d "$VENV_DIR" ]; then
  echo "[install] Creating venv at $VENV_DIR"
  uv venv "$VENV_DIR"
else
  echo "[install] Reusing existing venv at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"

if [ -f "uv.lock" ]; then
  uv sync --active --frozen
else
  uv sync --active
fi

echo "[install] Done. Activate with: source $VENV_DIR/bin/activate"
