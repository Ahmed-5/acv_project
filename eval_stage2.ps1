# Evaluate the Stage-2 (FixRes) best checkpoint with full rigor.
#
# Usage:
#   .\eval_stage2.ps1                          # default: use EMA, hflip TTA, Eigen crop
#   .\eval_stage2.ps1 -Ckpt path\to\file.pth   # custom checkpoint
#   .\eval_stage2.ps1 -NoTta -NoEma            # disable flags
#   .\eval_stage2.ps1 -ImgSize "480,640"       # override eval resolution
param(
    [string]$Ckpt = "checkpoints/dog_depth_nyu_v4_ft.pth",
    [string]$ImgSize = "480,640",
    [int]$BatchSize = 4,
    [int]$NumWorkers = 0,
    [switch]$NoEma,
    [switch]$NoTta,
    [switch]$NoEigenCrop,
    [switch]$GargCrop
)

$env:PYTHONIOENCODING = 'utf-8'

if (-not (Test-Path $Ckpt)) {
    Write-Host "[eval_stage2] Checkpoint not found: $Ckpt" -ForegroundColor Red
    Write-Host "Available checkpoints:"
    Get-ChildItem -Path "checkpoints" -Filter "*.pth" -ErrorAction SilentlyContinue |
        ForEach-Object { Write-Host "  $($_.FullName)" }
    exit 1
}

$cmd = @(
    "python", "eval_nyu.py",
    "--checkpoint", $Ckpt,
    "--img-size", $ImgSize,
    "--batch-size", $BatchSize,
    "--num-workers", $NumWorkers
)

if (-not $NoEma)        { $cmd += "--use-ema" }
if (-not $NoTta)        { $cmd += "--tta-hflip" }
if (-not $NoEigenCrop)  { $cmd += "--eigen-crop" }
if ($GargCrop)          { $cmd += "--garg-crop" }

Write-Host "[eval_stage2] Running: $($cmd -join ' ')" -ForegroundColor Cyan
& $cmd[0] $cmd[1..($cmd.Length - 1)]
