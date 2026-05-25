# PowerShell — package the clean fork into a zip + bita upload to bitahub.
#
# Usage:
#   .\scripts\package_and_upload.ps1                     # pack only
#   .\scripts\package_and_upload.ps1 -Upload             # pack + upload
#   .\scripts\package_and_upload.ps1 -Upload -Endpoint <ep> -Bucket <bk>
#
# Defaults follow memory project_uv2m_bitahub.md.

param(
    [switch]$Upload,
    [string]$Endpoint = "https://www.bitahub.com",
    [string]$Bucket   = "b20260514152010197qfyxbs",
    [string]$Prefix   = "code/",
    [string]$OutDir   = "$PSScriptRoot\..\..\_pack"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path "$PSScriptRoot\..").Path
$RepoName = Split-Path $RepoRoot -Leaf      # 'UniMatch-V2-clean'
$Stamp    = Get-Date -Format "yyyyMMdd_HHmmss"
$Staging  = Join-Path $OutDir "$RepoName"
$Zip      = Join-Path $OutDir "$RepoName`_$Stamp.zip"

# ---- exclude rules (build/cache/checkpoints/logs) ----
$Excludes = @(
    "__pycache__", ".git", ".idea", ".vscode", ".pytest_cache", ".mypy_cache",
    "_pack", "exp", "experiments", "training-logs", "runs", "tb_logs",
    "*.pth", "*.pt", "*.ckpt", "*.zip", "*.tar", "*.tar.gz",
    "*.log", "*.tmp", "*.swp", "*.png", "*.jpg",
    "wandb", ".DS_Store",
    # upstream splits / docs we never use on bitahub
    "*.pdf"
)
# Directory exclusions that need ABSOLUTE paths for robocopy /XD
$ExcludeAbsDirs = @(
    "$PSScriptRoot\..\splits\coco",
    "$PSScriptRoot\..\splits\ade20k",
    "$PSScriptRoot\..\splits\cityscapes",
    "$PSScriptRoot\..\splits\pascal"
) | ForEach-Object { (Resolve-Path -LiteralPath $_ -ErrorAction SilentlyContinue).Path } |
    Where-Object { $_ -ne $null }

Write-Host "[pack] repo  = $RepoRoot"
Write-Host "[pack] out   = $Zip"

if (Test-Path $Staging) { Remove-Item -Recurse -Force $Staging }
New-Item -ItemType Directory -Force -Path $Staging | Out-Null

# robocopy with exclusions; suppress its "success" exit codes (0..7) which
# PowerShell treats as errors with $ErrorActionPreference=Stop.
$xd = $Excludes | Where-Object { -not ($_ -like "*.*") } | ForEach-Object { @("/XD", $_) }
$xf = $Excludes | Where-Object {       $_ -like "*.*"   } | ForEach-Object { @("/XF", $_) }
$xdAbs = $ExcludeAbsDirs | ForEach-Object { @("/XD", $_) }
$rcArgs = @($RepoRoot, $Staging, "/E", "/NFL", "/NDL", "/NJH", "/NJS", "/NP") + $xd + $xdAbs + $xf
& robocopy @rcArgs | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy failed with exit $LASTEXITCODE" }

# create zip — include the staging dir itself (NOT its contents at root),
# so unzip on the server produces /data/code/UniMatch-V2/ rather than scattering
# unimatch_v2.py, configs/, etc. into /data/code/.
if (Test-Path $Zip) { Remove-Item $Zip -Force }
Compress-Archive -Path $Staging -DestinationPath $Zip -CompressionLevel Optimal
$Size = (Get-Item $Zip).Length / 1MB
Write-Host ("[pack] zipped {0:N1} MB -> {1}" -f $Size, $Zip)

function Print-UnpackCommand {
    Write-Host ""
    Write-Host "On the bitahub server, unpack with:"
    Write-Host "  cd /data/code && rm -rf ${RepoName}.old && \"
    Write-Host "    [ -d ${RepoName} ] && mv ${RepoName} ${RepoName}.old; \"
    Write-Host "    unzip -q /data/code/$(Split-Path $Zip -Leaf) -d /data/code"
}

if (-not $Upload) {
    Write-Host "[done] re-run with -Upload to push to bitahub."
    Write-Host "[scp] fallback: scp this zip to /data/code/ before unpacking."
    Print-UnpackCommand
    exit 0
}

# ---- bita upload (single zip, with retry) ----
if (-not (Get-Command bita -ErrorAction SilentlyContinue)) {
    throw "bita CLI not in PATH. Install it first or scp the zip manually."
}

$env:BITA_EP = $Endpoint
$env:BITA_BK = $Bucket
Write-Host "[upload] $Zip -> $Bucket/$Prefix"
$tries = 0; $maxTries = 5
while ($tries -lt $maxTries) {
    $tries += 1
    & bita upload -e $Endpoint -b $Bucket -o $Prefix -l $Zip
    if ($LASTEXITCODE -eq 0) { Write-Host "[upload] OK"; break }
    Write-Warning "[upload] attempt $tries failed (exit $LASTEXITCODE). Retrying in 30s..."
    Start-Sleep -Seconds 30
}
if ($LASTEXITCODE -ne 0) { throw "bita upload failed after $maxTries attempts." }

Print-UnpackCommand
