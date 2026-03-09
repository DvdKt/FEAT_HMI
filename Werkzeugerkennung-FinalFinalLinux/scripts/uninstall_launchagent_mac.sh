#!/usr/bin/env bash
set -euo pipefail

PLIST_NAME="com.werkzeugerkennung.backend"
PLIST_PATH="$HOME/Library/LaunchAgents/${PLIST_NAME}.plist"

launchctl unload "$PLIST_PATH" >/dev/null 2>&1 || true
rm -f "$PLIST_PATH"

echo "Removed LaunchAgent: $PLIST_PATH"
