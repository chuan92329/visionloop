#!/usr/bin/env bash
# Development mode: API with auto-reload on :8000 and the Vite dev server on :5173.
set -euo pipefail
cd "$(dirname "$0")/.."
(cd backend && python -m uvicorn app.main:app --reload --port 8000) &
API_PID=$!
trap 'kill $API_PID 2>/dev/null' EXIT
cd frontend && npm run dev
