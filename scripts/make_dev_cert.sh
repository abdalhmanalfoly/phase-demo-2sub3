#!/usr/bin/env bash
# Create a LOCAL development certificate for https://<LAN-IP>:8765 so phone browsers allow camera access.
# Preferred: mkcert (installs a local CA you then trust on the phone). Fallback: openssl self-signed
# (the phone shows a warning you must accept; some browsers refuse camera access on untrusted certs).
# Usage: scripts/make_dev_cert.sh 192.168.1.50 [extra-name ...]
set -euo pipefail
cd "$(dirname "$0")/.."
[ "$#" -ge 1 ] || { echo "usage: $0 <LAN-IP-or-hostname> [more names...]" >&2; exit 1; }
mkdir -p certs
if command -v mkcert >/dev/null; then
  mkcert -cert-file certs/dev.crt -key-file certs/dev.key localhost 127.0.0.1 ::1 "$@"
  echo "mkcert CA root (install/trust it on each phone): $(mkcert -CAROOT)/rootCA.pem"
else
  SAN="DNS:localhost,IP:127.0.0.1"
  for n in "$@"; do if [[ "$n" =~ ^[0-9.]+$ ]]; then SAN="$SAN,IP:$n"; else SAN="$SAN,DNS:$n"; fi; done
  openssl req -x509 -newkey rsa:2048 -nodes -days 30 -keyout certs/dev.key -out certs/dev.crt \
    -subj "/CN=saga3d-dev" -addext "subjectAltName=$SAN"
  chmod 600 certs/dev.key
  echo "Self-signed certificate created (valid 30 days). Phones must accept the warning."
fi
echo "Set in .env:  SAGA_TLS_CERT=certs/dev.crt  SAGA_TLS_KEY=certs/dev.key   (certs/ is git-ignored)"
