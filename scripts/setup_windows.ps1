<#
.SYNOPSIS
  Creates the Python environments VisionLoop needs (Windows). Nothing has to be installed first.

.DESCRIPTION
  Python comes from conda when it is installed; otherwise the script downloads uv, which
  installs Python into this folder (tools\python, environments in envs\). Two environments:

  visionloop  the platform itself (FastAPI, Pillow ...). Small, no PyTorch.
  vl-models   PyTorch with CUDA + RF-DETR + transformers (Grounding DINO, SAM 2)
              + the official YOLOv9 repository (cloned into models\yolov9) if you want it.

  It also downloads the default model weights (all Apache-2.0, about 2-3 GB) so the first
  click in the app does not wait; other sizes are downloaded the first time they are used.
  YOLOv9 / GELAN are GPL-3.0, so the script asks before downloading them.

  The script can be run again at any time; existing environments are reused and
  only missing packages are installed. At the end it writes local_settings.json
  so that the VisionLoop shortcut and the platform find both environments automatically, and
  runs scripts\check_models_env.py to verify the GPU.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1

.EXAMPLE
  # platform only (no PyTorch, no models)
  powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -SkipModels
#>
param(
  [string]$PlatformEnv = "visionloop",
  [string]$ModelsEnv = "vl-models",
  [string]$PythonVersion = "3.11",
  # PyTorch wheel index: cu130 (CUDA 13.0, RTX 20xx and newer) or cu126 (older GPUs / drivers)
  [string]$Cuda = "cu130",
  # YOLOv9 pretrained weights to download for fine-tuning (t, s, m, c, e)
  [string[]]$Yolov9Weights = @("c"),
  [switch]$SkipModels,
  # YOLOv9 / GELAN are GPL-3.0: "ask" (default) asks during setup, "yes" / "no" answer in advance
  [ValidateSet("ask", "yes", "no")] [string]$Yolov9 = "ask",
  [switch]$SkipYolov9,   # same as -Yolov9 no
  # do not download the default model weights now (each is downloaded the first time it is used)
  [switch]$SkipWeights,
  # do not create the VisionLoop shortcuts on the desktop and in the Start menu
  [switch]$NoShortcut,
  # where Python comes from. "auto": conda if it is installed, otherwise uv, which this script
  # downloads itself (nothing needs to be installed first); "uv" / "conda" force one of them
  [ValidateSet("auto", "uv", "conda")] [string]$PythonManager = "auto"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is very slow with the progress bar
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Started = Get-Date

function Step([string]$msg) {
  Write-Host ""
  Write-Host "==> $msg" -ForegroundColor Cyan
}

function Info([string]$msg) { Write-Host "    $msg" -ForegroundColor DarkGray }

function Run([string]$exe, [string[]]$argList) {
  Info "$exe $($argList -join ' ')"
  # Out-Host: keep the command's output on screen only; otherwise it becomes part of the
  # return value of any function that calls Run (e.g. Ensure-Env returning the python path)
  & $exe @argList | Out-Host
  if ($LASTEXITCODE -ne 0) { throw "指令失敗（exit $LASTEXITCODE）：$exe $($argList -join ' ')" }
}

function Find-Conda {
  if ($env:CONDA_EXE -and (Test-Path $env:CONDA_EXE)) { return $env:CONDA_EXE }
  $cmd = Get-Command conda.exe -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }
  $bases = @(
    "$env:USERPROFILE\anaconda3", "$env:USERPROFILE\miniconda3", "$env:USERPROFILE\miniforge3",
    "$env:LOCALAPPDATA\anaconda3", "$env:LOCALAPPDATA\miniconda3",
    "$env:ProgramData\anaconda3", "$env:ProgramData\miniconda3"
  )
  foreach ($b in $bases) {
    $p = Join-Path $b "Scripts\conda.exe"
    if (Test-Path $p) { return $p }
  }
  return $null
}

function Get-EnvPython([string]$conda, [string]$name) {
  $json = & $conda env list --json | Out-String | ConvertFrom-Json
  foreach ($p in $json.envs) {
    if ((Split-Path $p -Leaf) -eq $name) {
      $py = Join-Path $p "python.exe"
      if (Test-Path $py) { return $py }
    }
  }
  return $null
}

function Ensure-CondaEnv([string]$conda, [string]$name) {
  $py = Get-EnvPython $conda $name
  if ($py) {
    Info "已有環境 $name（$py）"
    return $py
  }
  # conda-forge only: avoids the Anaconda "Terms of Service" prompt of the default channels
  Run $conda @("create", "-y", "-n", $name, "-c", "conda-forge", "--override-channels", "python=$PythonVersion")
  $py = Get-EnvPython $conda $name
  if (-not $py) { throw "建立環境 $name 失敗" }
  return $py
}

# --- uv: a single program that downloads Python itself, so nothing has to be installed first.
# Everything stays inside this folder (tools\ and envs\); other Pythons on the computer are untouched.
function Get-Uv {
  $uv = Join-Path $Root "tools\uv\uv.exe"
  if (Test-Path $uv) { return $uv }
  $cmd = Get-Command uv.exe -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }
  Step "下載 uv（Python 安裝工具，約 20 MB）"
  $zip = Join-Path $env:TEMP "visionloop-uv.zip"
  Invoke-WebRequest "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip" -OutFile $zip
  $dir = Join-Path $Root "tools\uv"
  New-Item -ItemType Directory -Force -Path $dir | Out-Null
  Expand-Archive $zip -DestinationPath $dir -Force
  Remove-Item $zip
  if (-not (Test-Path $uv)) { throw "下載 uv 失敗" }
  return $uv
}

function Ensure-UvEnv([string]$uv, [string]$name) {
  $dir = Join-Path $Root "envs\$name"
  $py = Join-Path $dir "Scripts\python.exe"
  if (Test-Path $py) {
    Info "已有環境 $name（$py）"
    return $py
  }
  # --seed puts pip in the environment, so the steps below are the same as with conda
  Run $uv @("venv", "--seed", "--python", $PythonVersion, $dir)
  if (-not (Test-Path $py)) { throw "建立環境 $name 失敗" }
  return $py
}

function Ensure-Env([string]$name) {
  if ($useConda) { return Ensure-CondaEnv $conda $name }
  return Ensure-UvEnv $uv $name
}

# Node.js is only needed to build the web UI from a git checkout (a release zip ships it built)
function Get-PortableNode {
  $dir = Join-Path $Root "tools\node"
  if (Test-Path (Join-Path $dir "node.exe")) { return $dir }
  $index = Invoke-RestMethod "https://nodejs.org/dist/index.json"
  $version = ($index | Where-Object { $_.version -like "v22.*" } | Select-Object -First 1).version
  Step "下載 Node.js $version（用來建置網頁介面）"
  $zip = Join-Path $env:TEMP "visionloop-node.zip"
  Invoke-WebRequest "https://nodejs.org/dist/$version/node-$version-win-x64.zip" -OutFile $zip
  $tools = Join-Path $Root "tools"
  Expand-Archive $zip -DestinationPath $tools -Force
  Remove-Item $zip
  Move-Item (Join-Path $tools "node-$version-win-x64") $dir
  return $dir
}

# ---------------------------------------------------------------------------
Step "準備 Python"
$conda = $null
$uv = $null
$hasUvEnvs = Test-Path (Join-Path $Root "envs\$PlatformEnv\Scripts\python.exe")
if ($PythonManager -eq "conda" -or ($PythonManager -eq "auto" -and -not $hasUvEnvs)) { $conda = Find-Conda }
if ($PythonManager -eq "conda" -and -not $conda) {
  throw "找不到 conda。請先安裝 Miniconda 或 Anaconda，或改用 -PythonManager uv（不需要事先安裝任何東西）。"
}
$useConda = [bool]$conda
if ($useConda) {
  Info "使用 conda：$conda"
} else {
  # Python installed by uv goes here as well, and only Python versions uv itself installed are used
  $env:UV_PYTHON_INSTALL_DIR = Join-Path $Root "tools\python"
  $env:UV_PYTHON_PREFERENCE = "only-managed"
  $uv = Get-Uv
  Info "使用 uv：$uv（Python 與環境都放在這個資料夾的 tools\ 和 envs\，不影響電腦上的其他 Python）"
}

$settingsPath = Join-Path $Root "local_settings.json"
$settings = @{}
if (Test-Path $settingsPath) {
  $old = Get-Content $settingsPath -Raw -Encoding UTF8 | ConvertFrom-Json
  foreach ($p in $old.PSObject.Properties) { $settings[$p.Name] = $p.Value }
}

# ---------------------------------------------------------------------------
Step "平台環境：$PlatformEnv"
$platformPy = Ensure-Env $PlatformEnv
Run $platformPy @("-m", "pip", "install", "--upgrade", "pip")
Run $platformPy @("-m", "pip", "install", "-r", (Join-Path $Root "backend\requirements.txt"))
# the desktop window (scripts\desktop.ps1); Windows only, so not in requirements.txt
Run $platformPy @("-m", "pip", "install", "pywebview>=5,<7")
$settings["platform_python"] = $platformPy

# node.exe installed into the conda env lives next to python.exe
$envDir = Split-Path $platformPy -Parent
$env:PATH = "$envDir;$envDir\Library\bin;$envDir\Scripts;$env:PATH"
if (Test-Path (Join-Path $Root "frontend\dist\.release")) {
  Info "網頁介面已經建置好（Release 版本），不需要 Node.js"
} else {
  $nodeDir = Join-Path $Root "tools\node"
  if (Test-Path (Join-Path $nodeDir "node.exe")) { $env:PATH = "$nodeDir;$env:PATH" }
  if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    if ($useConda) {
      Step "安裝 Node.js 到 $PlatformEnv（用來建置網頁介面）"
      Run $conda @("install", "-y", "-n", $PlatformEnv, "-c", "conda-forge", "--override-channels", "nodejs=22")
    } else {
      $env:PATH = "$(Get-PortableNode);$env:PATH"
    }
  } else {
    Info "Node.js $(node --version) 已安裝"
  }
  Step "建置網頁介面"
  Push-Location (Join-Path $Root "frontend")
  try {
    Run "npm.cmd" @("install")
    Run "npm.cmd" @("run", "build")
  } finally { Pop-Location }
}

# ---------------------------------------------------------------------------
if (-not $SkipModels) {
  Step "模型環境：$ModelsEnv（PyTorch $Cuda、RF-DETR、Grounding DINO、SAM 2）"
  $modelsPy = Ensure-Env $ModelsEnv
  Run $modelsPy @("-m", "pip", "install", "--upgrade", "pip")

  Step "安裝 PyTorch（CUDA 版，約 2.5 GB，需要一點時間）"
  Run $modelsPy @("-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/$Cuda")

  Step "安裝 RF-DETR 與 transformers"
  # current setuptools (older ones have known vulnerabilities); the pkg_resources the YOLOv9 code
  # imports is supplied by yolov9_compat.py when setuptools no longer has it
  Run $modelsPy @("-m", "pip", "install", "rfdetr[train]>=1.11,<2", "transformers>=5.1,<6", "pillow", "pyyaml")
  Run $modelsPy @("-m", "pip", "install", "--upgrade", "setuptools")
  $settings["models_python"] = $modelsPy

  if (-not $SkipWeights) {
    Step "下載預設的模型權重（Apache-2.0，約 2～3 GB）"
    Info "DINOv2 Base、SAM 2.1 Large、Grounding DINO Base，以及這張顯示卡能訓練的最大 RF-DETR"
    Run $modelsPy @((Join-Path $Root "scripts\fetch_weights.py"))
  }

  $wantYolov9 = if ($SkipYolov9) { "no" } else { $Yolov9 }
  if ($wantYolov9 -eq "ask" -and (Test-Path (Join-Path $Root "models\yolov9\train_dual.py"))) {
    $wantYolov9 = "yes"  # installed on an earlier run: keep it up to date
  }
  if ($wantYolov9 -eq "ask") {
    Write-Host ""
    Write-Host "YOLOv9 / GELAN（約 300 MB）使用 GPL-3.0 授權：" -ForegroundColor Yellow
    Write-Host "  自己使用（包括公司內部）沒有限制；但若把它和你的程式一起散布，整個程式必須同樣以 GPL-3.0 開源。"
    Write-Host "  不安裝也可以：其他模型（RF-DETR、初步標註、SAM 2 等）照常可用，之後可用 -Yolov9 yes 補裝。"
    try { $answer = Read-Host "要下載 YOLOv9 / GELAN 嗎？[Y/n]" } catch { $answer = "n" }  # no console to ask in
    $wantYolov9 = if ($answer -match "^\s*(n|no|否)\s*$") { "no" } else { "yes" }
  }
  if ($wantYolov9 -eq "no") { Info "略過 YOLOv9 / GELAN" }

  if ($wantYolov9 -eq "yes") {
    Step "YOLOv9（原作者版本，GPL-3.0，下載到 models\yolov9）"
    $modelsDir = Join-Path $Root "models"
    $repo = Join-Path $modelsDir "yolov9"
    New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null
    if (-not (Test-Path (Join-Path $repo "train_dual.py"))) {
      if (Get-Command git -ErrorAction SilentlyContinue) {
        Run "git" @("clone", "--depth", "1", "https://github.com/WongKinYiu/yolov9.git", $repo)
      } else {
        $zip = Join-Path $env:TEMP "yolov9-main.zip"
        Info "下載 https://github.com/WongKinYiu/yolov9/archive/refs/heads/main.zip"
        Invoke-WebRequest "https://github.com/WongKinYiu/yolov9/archive/refs/heads/main.zip" -OutFile $zip
        Expand-Archive $zip -DestinationPath $modelsDir -Force
        Rename-Item (Join-Path $modelsDir "yolov9-main") $repo
        Remove-Item $zip
      }
    } else {
      Info "已有 $repo"
    }

    # its requirements, minus torch/torchvision (already installed with CUDA)
    $req = Get-Content (Join-Path $repo "requirements.txt") | Where-Object {
      $_ -and -not $_.TrimStart().StartsWith("#") -and $_ -notmatch "^\s*torch(vision)?\b"
    }
    $reqFile = Join-Path $env:TEMP "visionloop-yolov9-requirements.txt"
    Set-Content -Path $reqFile -Value $req -Encoding ASCII
    Run $modelsPy @("-m", "pip", "install", "-r", $reqFile)

    $wdir = Join-Path $repo "weights"
    New-Item -ItemType Directory -Force -Path $wdir | Out-Null
    # gelan-c: the YOLOv9 family's size without an auxiliary branch; gelan-c-seg: its segmentation model
    foreach ($name in @($Yolov9Weights | ForEach-Object { "yolov9-$_.pt" }) + @("gelan-c.pt", "gelan-c-seg.pt")) {
      $dest = Join-Path $wdir $name
      if (Test-Path $dest) { Info "已有 $name"; continue }
      try {
        Info "下載 $name"
        Invoke-WebRequest "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/$name" -OutFile $dest
      } catch {
        Write-Host "    找不到 $name，這個尺寸會從頭訓練" -ForegroundColor Yellow
        if (Test-Path $dest) { Remove-Item $dest }
      }
    }
    $settings["yolov9_repo"] = $repo
  }
}

# ---------------------------------------------------------------------------
Step "寫入 local_settings.json"
$settings | ConvertTo-Json | Set-Content -Path $settingsPath -Encoding UTF8
Get-Content $settingsPath | ForEach-Object { Info $_ }

if (-not $SkipModels) {
  Step "檢查模型環境"
  & $settings["models_python"] (Join-Path $Root "scripts\check_models_env.py")
}

if (-not $NoShortcut) {
  Step "建立「VisionLoop」捷徑（桌面與開始功能表）"
  $shell = New-Object -ComObject WScript.Shell
  $startMenu = Join-Path ([Environment]::GetFolderPath("Programs")) "VisionLoop.lnk"
  foreach ($lnkPath in @((Join-Path ([Environment]::GetFolderPath("Desktop")) "VisionLoop.lnk"), $startMenu)) {
    $lnk = $shell.CreateShortcut($lnkPath)
    # straight to pythonw (no console, no PowerShell in between): the window opens at once, with
    # Windows' usual launch feedback; desktop.py rebuilds the web UI itself when it is out of date
    $lnk.TargetPath = Join-Path (Split-Path $platformPy -Parent) "pythonw.exe"
    $lnk.Arguments = "`"$(Join-Path $Root 'scripts\desktop.py')`""
    $lnk.WorkingDirectory = Join-Path $Root "backend"
    $lnk.IconLocation = "$(Join-Path $Root 'scripts\visionloop.ico'),0"
    $lnk.WindowStyle = 1
    $lnk.Description = "VisionLoop 標註與訓練平台"
    $lnk.Save()
    Info $lnkPath
  }
}

$mins = [math]::Round(((Get-Date) - $Started).TotalMinutes, 1)
Write-Host ""
if ($NoShortcut) {
  Write-Host "完成（$mins 分鐘）。之後執行 scripts\desktop.ps1 就能開啟。" -ForegroundColor Green
} else {
  Write-Host "完成（$mins 分鐘）。之後雙擊桌面上的「VisionLoop」就能開啟。" -ForegroundColor Green
}
