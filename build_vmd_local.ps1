# Build the 4-channel VMD bank LOCALLY (no 357 MB download). Low priority, 4 processes x 1 thread. ASCII only.
# Run in D:\DCWRU_Data\Paderborn_Experiment :  powershell -ExecutionPolicy Bypass -File build_vmd_local.ps1 [-Workers 4]
param([int]$Workers = 4)
$PY = "D:\chuqishiyan1\zhubao\pump_reproduction\.venv_gpu\Scripts\python.exe"
$env:OMP_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"; $env:OPENBLAS_NUM_THREADS = "1"
(Get-Process -Id $PID).PriorityClass = "BelowNormal"          # child python processes inherit the low priority
if (-not (Test-Path "centrifugal\pool\pool_motor2_100_ch1245.npz")) { Write-Host "missing centrifugal\pool\pool_motor2_100_ch1245.npz (run merge_v19.ps1 from v19-B first)" -ForegroundColor Red; exit 1 }
& $PY -c "import vmdpy" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "installing vmdpy ..." -ForegroundColor Cyan
    & $PY -m pip install vmdpy
    if ($LASTEXITCODE -ne 0) { & $PY -m pip install vmdpy -i https://pypi.tuna.tsinghua.edu.cn/simple }
    if ($LASTEXITCODE -ne 0) { Write-Host "pip install vmdpy FAILED" -ForegroundColor Red; exit 1 }
}
Copy-Item (Join-Path $PSScriptRoot "build_vmd_pool.py") "centrifugal\" -Force
Push-Location centrifugal
& $PY build_vmd_pool.py --pool pool\pool_motor2_100_ch1245.npz --workers $Workers
$rc = $LASTEXITCODE
Pop-Location
if ($rc -ne 0) { Write-Host "VMD build FAILED" -ForegroundColor Red; exit 1 }
Write-Host "VMD bank ready: now run  powershell -ExecutionPolicy Bypass -File run_batch_v19_cf.ps1" -ForegroundColor Green
