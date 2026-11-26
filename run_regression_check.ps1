# da-v1 step 0: baseline port regression check (addendum A1 section 7 item 3). ASCII only. Windows PowerShell 5.1.
# E0 (Bg4, no T_adapt), C1_fold1 of splits_da_v2, seed 0, evaluating ALL 800 target windows (--eval-part all).
# PASS = H within 0.694737 +/- 0.005. Writes regression_result.txt. Run from the folder that holds this script.
# Usage:  powershell -ExecutionPolicy Bypass -File .\run_regression_check.ps1
# Optional: -Splits <dir with the v2 npz files>  -OldSplits <dir of splits_ch1245> (only used if the check fails)
param(
    [string]$DataRoot  = "D:\DCWRU_Data\Paderborn_Experiment",
    [string]$Splits    = "",
    [string]$OldSplits = "",
    [string]$PY        = "D:\chuqishiyan1\zhubao\pump_reproduction\.venv_gpu\Scripts\python.exe"
)
$ErrorActionPreference = "Continue"
if ($Splits -eq "")    { $Splits    = Join-Path $DataRoot "centrifugal\splits_da_v2" }
if ($OldSplits -eq "") { $OldSplits = Join-Path $DataRoot "centrifugal\splits_ch1245" }
Set-Location -Path $PSScriptRoot
$env:PUMP_DEVICE = "cuda"
$env:OMP_NUM_THREADS = "2"; $env:MKL_NUM_THREADS = "2"; $env:OPENBLAS_NUM_THREADS = "2"; $env:NUMEXPR_NUM_THREADS = "2"
(Get-Process -Id $PID).PriorityClass = "BelowNormal"

$DATA = Join-Path $Splits "motor2_100_ch1245_C1_fold1.npz"
if (-not (Test-Path $PY))   { Write-Host "missing python: $PY" -ForegroundColor Red; exit 1 }
if (-not (Test-Path $DATA)) { Write-Host "missing split file: $DATA (use -Splits <dir>)" -ForegroundColor Red; exit 1 }
$EXP = @{ "m2_paderborn_kr.py" = "e86d3c0e737616eeb948e59e7941b67c1c8514c6995af7f170fd2bc991383005";
          "kr_modules.py"      = "7dfe0e59aed59963b03386fd78a0d00ef4966d2d776d6846112385f1b88a303f";
          "gpu_runtime.py"     = "ea887065b94e6c07692121bbd249012d36b808ebc914283050afc9d52d075046" }
foreach ($k in $EXP.Keys) {
    $h = (Get-FileHash $k -Algorithm SHA256).Hash.ToLower()
    Write-Host "$k sha256 $h"
    if ($h -ne $EXP[$k]) { Write-Host "STOP: $k is not the da-v1 delivery version (expected $($EXP[$k]))." -ForegroundColor Red; exit 1 }
}
$TAG = "DA1_REG_E0_C1_f1_seed0_all"
if (Test-Path "runs_paderborn\$TAG\results_exp2.json") {
    Write-Host "$TAG already exists: only re-checking it (delete runs_paderborn\$TAG to re-run)" -ForegroundColor Yellow
} else {
    $ARGS_E0 = @("--known","0,1","--unknown","2,3,4","--steps","1500","--warmup","300","--sup-steps","600","--device","cuda",
                 "--k-score","denergy","--energy-T","0.05","--unk-init","kr","--known-subcenters","2","--guard-expand","0.2",
                 "--label-balance-key","train_installation","--label-record-key","train_measurement",
                 "--record-key","train_installation","--refresh-every","300","--gate-gamma","2",
                 "--input","spec","--norm","window",
                 "--mode","generic","--contrast","plain","--vmd-bank","none","--known-guard","0.4","--guard-group-key","train_installation")
    Write-Host "===== $TAG =====" -ForegroundColor Cyan
    & $PY m2_paderborn_kr.py --data $DATA @ARGS_E0 --eval-part all --tag $TAG --seeds 0 *> "log_$TAG.txt"
    if ($LASTEXITCODE -ne 0) { Write-Host "run FAILED (exit $LASTEXITCODE), see log_$TAG.txt" -ForegroundColor Red; exit 1 }
}
& $PY check_regression.py --root runs_paderborn --tag $TAG --data $DATA --out regression_result.txt
if ($LASTEXITCODE -eq 0) {
    Write-Host "REGRESSION PASS: you can start run_batch_da1.ps1" -ForegroundColor Green
    Copy-Item regression_result.txt regression_PASS.txt -Force
    exit 0
}
if (Test-Path regression_PASS.txt) { Remove-Item regression_PASS.txt -Force }
Write-Host "REGRESSION FAIL: checking whether splits_ch1245 and splits_da_v2 hold the same arrays ..." -ForegroundColor Red
if (Test-Path $OldSplits) {
    & $PY compare_splits.py $OldSplits $Splits motor2_100_ch1245_C1_fold1 *> compare_splits_log.txt
    Get-Content compare_splits_log.txt
} else {
    Write-Host "old split dir not found: $OldSplits (use -OldSplits <dir>)" -ForegroundColor Yellow
}
Write-Host "Send regression_result.txt, log_$TAG.txt and compare_splits_log.txt back. Do not start the batch." -ForegroundColor Yellow
exit 1
