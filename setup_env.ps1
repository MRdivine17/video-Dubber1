# One-time setup (Windows, NVIDIA GPU): Python env, every package in requirements.txt,
# model weights, and the web UI build.
#
# The Python environment goes on the system SSD by default (-VenvDir to change it): a failing
# data disk silently corrupted installed packages, which crashes Python at import time.
# Models, videos and outputs stay inside this project folder.
param([string]$VenvDir = (Join-Path $env:USERPROFILE ".venvs\video-dubber"))

$root = $PSScriptRoot
$env:UV_LINK_MODE = "hardlink"     # cache and env on the same drive: no duplicate copies
$py = Join-Path $VenvDir "Scripts\python.exe"

if (-not (Test-Path $py)) { uv venv --python 3.11 $VenvDir }

# 1. All Python packages (CUDA PyTorch included) from the pinned requirements file.
uv pip install --python $py -r (Join-Path $root "requirements.txt") --index-strategy unsafe-best-match
uv pip check --python $py

# 2. Check the installed files are intact, then download the model weights once.
& $py (Join-Path $root "scripts\verify_install.py")
& $py (Join-Path $root "scripts\download_models.py")

# 3. Web UI.
Push-Location (Join-Path $root "web"); npm install; npm run build; Pop-Location

& $py -c "import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available())"
Write-Output "Done. Start the web app with start_studio.cmd, or dub from the command line with dub.cmd <url>."
