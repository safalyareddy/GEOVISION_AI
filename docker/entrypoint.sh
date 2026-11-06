#!/usr/bin/env bash
set -e

# Port defaults to 7860 (Hugging Face Spaces and Cloud default)
PORT="${PORT:-7860}"

echo "=========================================================="
echo " Starting GeoVision AI Platform in Container Environment "
echo " Public UI Port:      ${PORT}"
echo " User UID/GID:        $(id -u):$(id -g)"
echo " Working Directory:   $(pwd)"
echo " Architecture:        Streamlit + PyTorch + ONNX Runtime"
echo "=========================================================="

# Ensure cache directories exist with appropriate permissions
mkdir -p /app/checkpoints "${HOME}/.cache/geovision/checkpoints" /tmp/geovision

# Pre-flight check / acquisition of all four registered checkpoints
python -c "
import sys
from src.utils.model_registry import resolve_checkpoint_path, is_model_available, SUPPORTED_MODELS, CHAMPION_MODEL
print('Pre-flight checkpoint verification for all architectures:')
for m in SUPPORTED_MODELS:
    path = resolve_checkpoint_path(m, allow_download=False)
    status = 'Verified' if (path and path.exists()) else 'Missing'
    print(f'  • {m:<16}: {status} ({path})')
" || true

# Signal handling for graceful shutdown
cleanup() {
    echo "Received termination signal. Shutting down gracefully..."
    if [ -n "$STREAMLIT_PID" ] && kill -0 "$STREAMLIT_PID" 2>/dev/null; then
        echo "Stopping Streamlit (PID $STREAMLIT_PID)..."
        kill -TERM "$STREAMLIT_PID" 2>/dev/null || true
    fi
    wait "$STREAMLIT_PID" 2>/dev/null || true
    echo "GeoVision AI services stopped cleanly."
    exit 0
}

trap cleanup SIGTERM SIGINT SIGQUIT

# Start Streamlit dashboard on configured port
echo "Starting Streamlit dashboard on 0.0.0.0:${PORT}..."
streamlit run app.py \
    --server.port "${PORT}" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --browser.gatherUsageStats false &
STREAMLIT_PID=$!

# Wait on Streamlit process
wait "$STREAMLIT_PID"
cleanup
