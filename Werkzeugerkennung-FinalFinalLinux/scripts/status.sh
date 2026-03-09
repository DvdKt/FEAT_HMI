#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$ROOT_DIR/backend/.env"

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing $ENV_FILE. Run ./scripts/install_mac.sh first."
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "Missing required command: curl"
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

BACKEND_PORT="${BACKEND_PORT:-8000}"
HEALTH_URL="http://localhost:${BACKEND_PORT}/health"

if curl -fs "$HEALTH_URL" >/dev/null; then
  echo "Backend is reachable at $HEALTH_URL"
else
  echo "Backend is not reachable at $HEALTH_URL"
  exit 1
fi
