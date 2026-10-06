# Reinstall packages whose files were damaged on disk (found by scripts\verify_install.py),
# then install / audit everything in requirements.txt. Retries through network drops.
param([string]$Packages = "")   # comma-separated, e.g. "av,torch"
$pkgList = @($Packages -split "[,\s]+" | Where-Object { $_ })   # new name: $Packages is typed [string]
$root = Split-Path $PSScriptRoot -Parent
$env:UV_CACHE_DIR = Join-Path $root ".uv-cache"
$env:UV_HTTP_TIMEOUT = "180"
$env:UV_CONCURRENT_DOWNLOADS = "4"
$py = Join-Path $root ".venv\Scripts\python.exe"

# Cached copies live on the same disk and may be damaged too: drop them so fresh files are downloaded.
if ($pkgList.Count) { uv cache clean @pkgList 2>&1 | ForEach-Object { "$_" } }
$reinstall = @(); foreach ($p in $pkgList) { $reinstall += @("--reinstall-package", $p) }

for ($i = 1; $i -le 8; $i++) {
    Write-Output "--- install attempt $i"
    uv pip install --python $py -r (Join-Path $root "requirements.txt") --index-strategy unsafe-best-match @reinstall 2>&1 |
        ForEach-Object { "$_" } | Where-Object { $_ -notmatch "^\s+[\+\-~] |Downloading|Downloaded|Building|Built" }
    if ($LASTEXITCODE -eq 0) { Write-Output "INSTALL_OK"; break }
    Start-Sleep -Seconds 6
}
uv pip check --python $py 2>&1 | ForEach-Object { "$_" }
