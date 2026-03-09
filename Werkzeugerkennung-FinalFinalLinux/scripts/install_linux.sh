#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
VENV_DIR="$BACKEND_DIR/.venv"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1"
    exit 1
  fi
}

require_cmd git
require_cmd python3

mkdir -p "$BACKEND_DIR"

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip

REQUIREMENTS_FILE="$BACKEND_DIR/requirements.txt"
if [ ! -f "$REQUIREMENTS_FILE" ]; then
  echo "Missing $REQUIREMENTS_FILE"
  exit 1
fi

"$VENV_DIR/bin/python" -m pip install -r "$REQUIREMENTS_FILE"

ENV_EXAMPLE="$BACKEND_DIR/.env.example"
ENV_FILE="$BACKEND_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
  if [ ! -f "$ENV_EXAMPLE" ]; then
    echo "Missing $ENV_EXAMPLE"
    exit 1
  fi
  cp "$ENV_EXAMPLE" "$ENV_FILE"
  echo "Created $ENV_FILE"
fi

DEFAULT_DATA_DIR="${HOME}/WerkzeugerkennungData"
mkdir -p "$DEFAULT_DATA_DIR"

echo ""
echo "Install complete."
echo "Next steps:"
echo "  ./scripts/run_backend_linux.sh"
