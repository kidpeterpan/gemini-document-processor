#!/bin/bash
# Remove runtime data: durable store, uploads, generated artifacts, exports, logs.
# Usage: ./cleanup.sh
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="$BASE_DIR/runtime"

echo "========================================"
echo "Starting cleanup process..."
echo "========================================"

if [ -d "$RUNTIME_DIR" ]; then
  echo "Removing runtime directory ($RUNTIME_DIR)..."
  rm -rf "$RUNTIME_DIR"
  echo "✓ runtime/ removed (store, uploads, artifacts, exports, logs)"
else
  echo "! runtime directory not found at $RUNTIME_DIR (nothing to do)"
fi

echo "Removing stray log files..."
find "$BASE_DIR" -name "*.log" -type f -delete
echo "✓ Log files removed"

echo "========================================"
echo "Cleanup completed!"
echo "========================================"
