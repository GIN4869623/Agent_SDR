#!/bin/bash

PROJECT_DIR="/home/njupt/Desktop/Agent_SDR"
VENV_DIR="$PROJECT_DIR/.venv"
HOST="127.0.0.1"
PORT="8000"
URL="http://$HOST:$PORT"

cd "$PROJECT_DIR" || exit 1

# 如果服务没启动，就启动它
if ! lsof -i:$PORT >/dev/null 2>&1; then
    gnome-terminal -- bash -c "cd '$PROJECT_DIR'; source '$VENV_DIR/bin/activate'; uvicorn main:app --host $HOST --port $PORT --reload; exec bash"
    sleep 3
fi

# 打开前端页面
xdg-open "$URL" >/dev/null 2>&1
