# run_v4_pc.ps1 -- NEAT and the nine algorithms under ONE identical setup:
#   * inputs used exactly as built (NO normalisation of any kind)
#   * no cap on any position for any method (outputs are linear before the softmax)
#   * same parameter limits for every method: weights and biases within -30 .. +30
#   * same data, draws, windows, seeds, fitness (Sharpe) and 5 bp cost rule; Equal-Weight and Markowitz pay daily
#     rebalancing costs
# Results go to NEW folders (A5_n15 for NEAT, B3_algos_n15 for the algorithms); earlier runs are not touched.
# Safe to stop and restart: finished jobs are kept.
#
#   NEAT, all 8 feature sets:        powershell -ExecutionPolicy Bypass -File .\run_v4_pc.ps1 -Configs ALL -Only NEAT
#   Algorithms, all 8 feature sets:  powershell -ExecutionPolicy Bypass -File .\run_v4_pc.ps1 -Configs ALL -Only ALGOS
param([string]$Configs = "ALL", [ValidateSet("BOTH","NEAT","ALGOS")][string]$Only = "BOTH")
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
if ($Configs -eq "ALL") { $Configs = "PRICE_ONLY,CORR_ONLY,FULL,CORR_ALONE,SENT_ONLY,PRICE_SENT,CORR_SENT,PRICE_CORR_SENT" }
foreach ($chk in @(@("neat_sharpe_parallel_aws_32.py","no normalisation of any kind"), @("neat_sweep.py","output_map"), @("algos_sweep.py","param_bounds"))) {
    if (-not (Test-Path "$PSScriptRoot\$($chk[0])") -or -not (Select-String -Path "$PSScriptRoot\$($chk[0])" -Pattern $chk[1] -Quiet)) {
        Write-Host "STOP: $($chk[0]) is missing or an OLD version in this folder."; exit 1 } }
python -c "import cma" 2>$null
if ($LASTEXITCODE -ne 0) { Write-Host "Installing the 'cma' package ..."; pip install cma }
$root = "C:\Users\rahul\neat_runs"
New-Item -ItemType Directory -Force -Path $root | Out-Null

function Set-Common {
    Get-ChildItem Env:NEAT_* | ForEach-Object { Remove-Item "Env:$($_.Name)" }
    Get-ChildItem Env:ALGO_* -ErrorAction SilentlyContinue | ForEach-Object { Remove-Item "Env:$($_.Name)" }
    Remove-Item Env:OUTPUT_MAP -ErrorAction SilentlyContinue
    $env:PYTHONUTF8   = "1"
    $env:NEAT_MASTER  = "C:\Users\rahul\Downloads\master_dataset_32_clean.csv"
    $env:NEAT_N       = "15"
    $env:NEAT_SEEDS   = "42,7,123,1,2"
    $env:NEAT_JOBS    = "10"
    $env:NEAT_CONFIGS = $Configs
    $env:OUTPUT_MAP   = "free"
}

if ($Only -ne "ALGOS") {
    Set-Common
    $env:NEAT_SWEEP_DIR = "$root\A5_n15"
    Write-Host "=== NEAT (raw inputs, no cap) on $Configs started $(Get-Date) ==="
    python -u neat_sweep.py 2>&1 | Tee-Object -FilePath "$root\A5_n15.log" -Append
    Write-Host "=== NEAT finished $(Get-Date). Results: $root\A5_n15 ==="
}
if ($Only -eq "NEAT") { exit 0 }

Set-Common
$env:NEAT_SWEEP_DIR = "$root\B3_algos_n15"
Write-Host "=== NINE ALGORITHMS (raw inputs, no cap, limits +-30) on $Configs started $(Get-Date) ==="
python -u algos_sweep.py 2>&1 | Tee-Object -FilePath "$root\B3_algos_n15.log" -Append
Write-Host "=== ALL DONE $(Get-Date). NEAT: $root\A5_n15   Algorithms: $root\B3_algos_n15 ==="
