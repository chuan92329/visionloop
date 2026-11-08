# Windows PowerShell development mode: API on :8000 (auto-reload) + Vite dev server on :5173
$ErrorActionPreference = "Stop"
$root = Join-Path $PSScriptRoot ".."
$api = Start-Process python -ArgumentList "-m","uvicorn","app.main:app","--reload","--port","8000" `
  -WorkingDirectory (Join-Path $root "backend") -PassThru -NoNewWindow
try {
  Set-Location (Join-Path $root "frontend")
  npm run dev
} finally {
  Stop-Process -Id $api.Id -ErrorAction SilentlyContinue
}
