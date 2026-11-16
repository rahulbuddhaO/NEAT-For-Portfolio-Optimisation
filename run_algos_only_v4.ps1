# run_algos_only_v4.ps1 -- ONLY the nine comparison algorithms (NEAT is finished; it is NOT rerun).
# Put this file in C:\Users\rahul\Downloads\files-v4 and run:  powershell -ExecutionPolicy Bypass -File .\run_algos_only_v4.ps1
# Safe to stop and restart: finished jobs are kept (same output folder).
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
foreach ($f in @("algos_sweep.py","neat_sweep.py","neat_sharpe_parallel_aws_32.py")) {
    if (-not (Test-Path "$PSScriptRoot\$f")) { Write-Host "STOP: $f is missing from this folder."; exit 1 } }

# Use the REAL Python, never the Microsoft Store alias (that alias caused the "can't open file ...WindowsApps\python.exe" error).
$py = "C:\Users\rahul\AppData\Local\Python\pythoncore-3.14-64\python.exe"
if (-not (Test-Path $py)) { Write-Host "STOP: Python not found at $py"; exit 1 }
& $py -c "import cma" 2>$null
if ($LASTEXITCODE -ne 0) { Write-Host "Installing cma ..."; & $py -m pip install cma }

$root = "C:\Users\rahul\neat_runs"
New-Item -ItemType Directory -Force -Path $root | Out-Null
Get-ChildItem Env:NEAT_* -ErrorAction SilentlyContinue | ForEach-Object { Remove-Item "Env:$($_.Name)" }
Get-ChildItem Env:ALGO_* -ErrorAction SilentlyContinue | ForEach-Object { Remove-Item "Env:$($_.Name)" }
$env:PYTHONUTF8     = "1"
$env:NEAT_MASTER    = "C:\Users\rahul\Downloads\master_dataset_32_clean.csv"
$env:NEAT_SWEEP_DIR = "$root\B3_algos_n15"
$env:NEAT_N         = "15"
$env:NEAT_SEEDS     = "42,7,123,1,2"
$env:NEAT_JOBS      = "10"
$env:NEAT_CONFIGS   = "PRICE_ONLY,CORR_ONLY,FULL,CORR_ALONE,SENT_ONLY,PRICE_SENT,CORR_SENT,PRICE_CORR_SENT"

Write-Host "=== NINE ALGORITHMS started $(Get-Date)  (python: $py) ==="
# *> sends all output (including stderr) to the log without PowerShell's NativeCommandError wrapping; the console shows the live tail via Get-Content in another window.
& $py -u algos_sweep.py *> "$root\B3_algos_n15.log"
Write-Host "=== finished $(Get-Date). Results: $root\B3_algos_n15 ==="
