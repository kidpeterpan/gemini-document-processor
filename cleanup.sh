#!/bin/bash

# Script to clean up results, uploads, and log files
# Usage: ./cleanup.sh

# Set the base directory - default to the current directory
BASE_DIR="$(pwd)"

# File paths
RESULTS_DIR="$BASE_DIR/results"
UPLOADS_DIR="$BASE_DIR/uploads"

echo "========================================"
echo "Starting cleanup process..."
echo "========================================"

# Clean results directory
if [ -d "$RESULTS_DIR" ]; then
    echo "Cleaning results directory..."
    rm -rf "$RESULTS_DIR"/*
    echo "✓ Results directory emptied"
else
    echo "! Results directory not found at $RESULTS_DIR"
fi

# Clean uploads directory
if [ -d "$UPLOADS_DIR" ]; then
    echo "Cleaning uploads directory..."
    rm -rf "$UPLOADS_DIR"/*
    echo "✓ Uploads directory emptied"
else
    echo "! Uploads directory not found at $UPLOADS_DIR"
fi

# Remove log files
echo "Removing log files..."
find "$BASE_DIR" -name "*.log" -type f -delete
echo "✓ Log files removed"

# Done
echo "========================================"
echo "Cleanup completed!"
echo "========================================"