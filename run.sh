#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "$PROJECT_DIR")"
VENV_DIR="${HOME}/venv/${PROJECT_NAME}"
HOST="${1:-0.0.0.0}"
PORT="${2:-8080}"

if [ ! -d "$VENV_DIR" ]; then
  echo "Virtualenv not found: $VENV_DIR"
  echo "Run: ./install.sh"
  exit 1
fi

if [ -f "$PROJECT_DIR/.env" ]; then
  set -a
  source "$PROJECT_DIR/.env"
  set +a
fi

source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"

exec uv run uvicorn app:app --host "$HOST" --port "$PORT"
