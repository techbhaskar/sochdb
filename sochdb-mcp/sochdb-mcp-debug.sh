#!/bin/bash
# SochDB MCP Server debug wrapper - logs all I/O for debugging
# This script logs stdin/stdout to help debug Goose integration

DB_PATH="/Users/sushanth/sochdb/sochdb_data"
LOG_FILE="/tmp/sochdb-mcp-debug.log"

# Create log file
echo "=== SochDB MCP Debug Session Started: $(date) ===" >> "$LOG_FILE"

# Use tee to capture both input and output
# Create FIFOs for bidirectional logging
STDIN_LOG="/tmp/sochdb-mcp-stdin.log"
STDOUT_LOG="/tmp/sochdb-mcp-stdout.log"

# Log all input and output
exec 3>&1  # Save stdout
exec 4>&2  # Save stderr

# Run the MCP server with full path
/Users/sushanth/sochdb/target/release/sochdb-mcp --db "$DB_PATH" 2>> "$LOG_FILE"
