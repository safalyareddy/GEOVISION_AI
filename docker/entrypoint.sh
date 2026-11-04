#!/usr/bin/env bash
set -e

# Hugging Face Spaces port defaults to 7860
PORT="${PORT:-7860}"
API_PORT="${API_PORT:-8000}"
export API_URL="http://127.0.0.1:${API_PORT}"

echo "=========================================================="
echo " Starting GeoVision AI Platform in Container Environment "
echo " Public UI Port:      ${PORT}"
echo " Internal API Port:   ${API_PORT}"
echo " API Endpoint:        ${API_URL}"
echo " User UID/GID:        $(id -u):$(id -g)"
echo " Working Directory:   $(pwd)"
echo "=========================================================="

# Ensure cache directories exist with appropriate permissions
mkdir -p /app/checkpoints "${HOME}/.cache/geovision/checkpoints" /tmp/geovision

# Pre-flight check / acquisition of champion FPN checkpoint
python -c "
import sys
from src.utils.model_registry import resolve_checkpoint_path, is_model_available, CHAMPION_MODEL
print(f'Checking availability of champion model ({CHAMPION_MODEL})...')
path = resolve_checkpoint_path(CHAMPION_MODEL, allow_download=True)
if path and path.exists():
    print(f'Champion model verified at: {path}')
else:
    print('WARNING: Champion model could not be pre-loaded at startup. Will attempt runtime download when requested.')
" || true

# Signal handling for graceful shutdown
cleanup() {
    echo "Received termination signal. Shutting down gracefully..."
    if [ -n "$STREAMLIT_PID" ] && kill -0 "$STREAMLIT_PID" 2>/dev/null; then
        echo "Stopping Streamlit (PID $STREAMLIT_PID)..."
        kill -TERM "$STREAMLIT_PID" 2>/dev/null || true
    fi
    if [ -n "$UVICORN_PID" ] && kill -0 "$UVICORN_PID" 2>/dev/null; then
        echo "Stopping FastAPI / Uvicorn (PID $UVICORN_PID)..."
        kill -TERM "$UVICORN_PID" 2>/dev/null || true
    fi
    wait "$STREAMLIT_PID" 2>/dev/null || true
    wait "$UVICORN_PID" 2>/dev/null || true
    echo "GeoVision AI services stopped cleanly."
    exit 0
}

trap cleanup SIGTERM SIGINT SIGQUIT

# 1. Start FastAPI microservice on internal port
echo "Starting FastAPI microservice on 127.0.0.1:${API_PORT}..."
uvicorn src.api.main:app \
    --host 127.0.0.1 \
    --port "${API_PORT}" \
    --log-level info &
UVICORN_PID=$!

# Wait for FastAPI to be healthy
echo "Waiting for FastAPI healthcheck..."
for i in $(seq 1 30); do
    if curl -s "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; then
        echo "FastAPI microservice is healthy on port ${API_PORT}."
        break
    fi
    sleep 0.5
done

# 2. Start Streamlit dashboard on public port (7860 for Hugging Face Spaces)
echo "Starting Streamlit dashboard on 0.0.0.0:${PORT}..."
streamlit run app.py \
    --server.port "${PORT}" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --browser.gatherUsageStats false &
STREAMLIT_PID=$!

# Wait on background processes
wait -n "$STREAMLIT_PID" "$UVICORN_PID"
cleanup
