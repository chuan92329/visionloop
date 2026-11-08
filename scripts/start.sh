#!/usr/bin/env bash
# Build the UI once and serve everything from http://127.0.0.1:8000
# (a release zip ships the built UI, marked by frontend/dist/.release, so Node.js is not needed)
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -f frontend/dist/.release ]; then
  [ -d frontend/node_modules ] || (cd frontend && npm install)
  (cd frontend && npm run build)
fi
cd backend && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
