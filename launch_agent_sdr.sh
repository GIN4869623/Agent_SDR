#!/bin/bash
# ── Agent SDR Launch Script ──
# Architecture: Claude Code pattern (MCP-centric agent loop)
# Entry point:   src/server.py (delegated from main.py)
#
# Usage:
#   ./launch_agent_sdr.sh           # Launch web UI
#   python -m src.cli               # Launch MCP stdio (for Claude Code)

PROJECT_DIR="$HOME/Desktop/Agent_SDR"
VENV_DIR="$PROJECT_DIR/.venv"
HOST="127.0.0.1"
PORT="8000"
URL="http://$HOST:$PORT"

cd "$PROJECT_DIR" || exit 1

# Kill any existing process on the port (handles "address already in use")
if lsof -ti:$PORT >/dev/null 2>&1; then
    echo "Port $PORT is in use, killing old process..."
    fuser -k $PORT/tcp 2>/dev/null
    sleep 1
fi

# Start the server
gnome-terminal -- bash -c "cd '$PROJECT_DIR'; source '$VENV_DIR/bin/activate'; uvicorn main:app --host $HOST --port $PORT; exec bash"
sleep 3

# Open web UI
xdg-open "$URL" >/dev/null 2>&1
