#!/usr/bin/env bash
# Idempotent setup for the *deck* toolchain (slides build + PDF export).
# NOTE: This does NOT set up SATA — see docs/setup-sata.md (needs a GPU box).
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[setup] python deps (python-pptx, PyMuPDF)…"
python3 - <<'PY' || pip install -q python-pptx PyMuPDF
import importlib, sys
for m in ("pptx", "fitz"):
    importlib.import_module(m)
print("python deps already present")
PY

echo "[setup] LibreOffice Impress (for pptx -> pdf)…"
if ! ls /usr/lib/libreoffice/share/registry/impress.xcd >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libreoffice-impress >/dev/null
  else
    echo "[setup] WARN: apt-get unavailable; install libreoffice-impress manually."
  fi
else
  echo "[setup] libreoffice-impress already present"
fi

echo "[setup] node deps for the deck generator…"
( cd scripts/deck && npm install --silent )

echo "[setup] done. Build the deck with:  bash scripts/build-deck.sh"
