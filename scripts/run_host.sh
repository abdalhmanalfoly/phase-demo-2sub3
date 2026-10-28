#!/usr/bin/env bash
# Run the SAGA 3D host. Serves the sensor page, operator page, REST API and the sensor WebSocket.
# HTTPS (needed for phone cameras) is enabled automatically when SAGA_TLS_CERT and SAGA_TLS_KEY point to files.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a
PY=".venv/bin/python"; [ -x "$PY" ] || { echo "No .venv found. Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2; exit 1; }
HOST="${SAGA_HOST:-0.0.0.0}"; PORT="${SAGA_PORT:-8765}"
TLS=()
if [ -n "${SAGA_TLS_CERT:-}" ] && [ -f "${SAGA_TLS_CERT}" ] && [ -f "${SAGA_TLS_KEY:-/nonexistent}" ]; then
  TLS=(--ssl-certfile "$SAGA_TLS_CERT" --ssl-keyfile "$SAGA_TLS_KEY"); SCHEME=https
else
  SCHEME=http
  echo "NOTE: serving plain HTTP. Phones will block the camera on http://<LAN-IP>; see docs/setup.md (scripts/make_dev_cert.sh)." >&2
fi
echo "Operator: $SCHEME://localhost:$PORT/client/operator.html   Sensor: $SCHEME://<this-machine-LAN-IP>:$PORT/client/" >&2
# --ws-max-size caps frames at the protocol level (the app enforces its own, smaller limit too)
exec "$PY" -m uvicorn host.app.main:app --host "$HOST" --port "$PORT" --ws-max-size 65536 "${TLS[@]}"
