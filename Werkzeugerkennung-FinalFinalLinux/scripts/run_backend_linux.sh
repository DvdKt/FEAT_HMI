#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
ENV_FILE="$BACKEND_DIR/.env"
VENV_PY="$BACKEND_DIR/.venv/bin/python"

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing $ENV_FILE. Run ./scripts/install_linux.sh first."
  exit 1
fi

if [ ! -x "$VENV_PY" ]; then
  echo "Missing virtualenv at $VENV_PY. Run ./scripts/install_linux.sh first."
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

BACKEND_HOST="${BACKEND_HOST:-0.0.0.0}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
DATA_DIR="${DATA_DIR:-$HOME/WerkzeugerkennungData}"

mkdir -p "$DATA_DIR"

LAN_IP="$(ipconfig getifaddr en0 2>/dev/null || true)"
if [ -z "$LAN_IP" ]; then
  LAN_IP="$(ipconfig getifaddr en1 2>/dev/null || true)"
fi
if [ -z "$LAN_IP" ]; then
  LAN_IP="127.0.0.1"
fi

echo "Backend URL:    http://${LAN_IP}:${BACKEND_PORT}"
echo "Healthcheck:    http://${LAN_IP}:${BACKEND_PORT}/health"
echo "Data directory: ${DATA_DIR}"
echo "Press Ctrl+C to stop."
echo ""

cd "$ROOT_DIR"
exec "$VENV_PY" -m uvicorn backend_server:app --host "$BACKEND_HOST" --port "$BACKEND_PORT"
