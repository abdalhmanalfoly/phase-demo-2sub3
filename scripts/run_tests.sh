#!/usr/bin/env bash
# Run every automated test: Python host + contract, then browser-logic tests under Node.
set -uo pipefail
cd "$(dirname "$0")/.."
rc=0
echo "== Python (host, protocol, JS<->Python contract) =="; .venv/bin/python -m pytest || rc=1
if command -v node >/dev/null; then
  echo "== Node (browser protocol/state/transport) =="; (cd sensor-client && node --test tests/*.test.mjs) || rc=1
else echo "node not found: skipping browser-logic tests (contract test is skipped too)"; fi
exit $rc
