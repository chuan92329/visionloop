# Windows: open VisionLoop in its own window, the same as the VisionLoop shortcut
#   powershell -ExecutionPolicy Bypass -File scripts\desktop.ps1
# desktop.py rebuilds the web UI by itself when it is out of date.
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$py = "python"
$settingsPath = Join-Path $Root "local_settings.json"
if (Test-Path $settingsPath) {
  $s = Get-Content $settingsPath -Raw -Encoding UTF8 | ConvertFrom-Json
  if ($s.platform_python -and (Test-Path $s.platform_python)) { $py = $s.platform_python }
}

# pythonw.exe has no console window
$pyw = Join-Path (Split-Path $py -Parent) "pythonw.exe"
if (-not (Test-Path $pyw)) { $pyw = $py }
Start-Process -FilePath $pyw -ArgumentList "`"$(Join-Path $Root 'scripts\desktop.py')`"" -WorkingDirectory (Join-Path $Root "backend")
