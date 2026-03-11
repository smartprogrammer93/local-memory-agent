#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="qwen-memory-agent"
SERVICE_FILE="$(cd "$(dirname "$0")" && pwd)/systemd/${SERVICE_NAME}.service"
TARGET="/etc/systemd/system/${SERVICE_NAME}.service"

# Ensure inbox directory exists
mkdir -p "$(dirname "$0")/inbox"

# Symlink service file
ln -sf "$SERVICE_FILE" "$TARGET"
echo "Linked $SERVICE_FILE -> $TARGET"

# Reload systemd, enable and start
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl restart "$SERVICE_NAME"

echo ""
systemctl status "$SERVICE_NAME" --no-pager
