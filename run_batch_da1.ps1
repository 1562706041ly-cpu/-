# da-v1 baselines: E0 (Bg4, source only) and E0' (Bg4 + T_adapt as unlabelled data, addendum A1-D8).
# 6 primary splits (C1,C2,C3 x fold1,fold3) x seeds 0,1,2 x {E0, E0'} = 36 runs. Evaluation on T_eval only.
# ASCII only, Windows PowerShell 5.1.  Usage:  powershell -ExecutionPolicy Bypass -File .\run_batch_da1.ps1
# Needs regression_PASS.txt (written by run_regression_check.ps1) unless -SkipGate is given.
# A failed or non-finite run is logged to invalid_da1.txt and the batch continues. Finished runs are skipped,
# so the batch can be stopped and restarted. Order: seed by seed, so a partial run still has complete seeds.
param(
    [string]$DataRoot = "D:\DCWRU_Data\Paderborn_Experiment",
    [string]$Splits   = "",
    [string]$PY       = "D:\chuqishiyan1\zhubao\pump_reproduction\.venv_gpu\Scripts\python.exe",
    [int[]]$Seeds     = @(0,1,2),
    [switch]$SkipGate
)
$ErrorActionPreference = "Continue"
if ($Splits -eq "") { $Splits = Join-Path $DataRoot "centrifugal\splits_da_v2" }
Set-Location -Path $PSScriptRoot
$env:PUMP_DEVICE = "cuda"
$env:OMP_NUM_THREADS = "2"; $env:MKL_NUM_THREADS = "2"; $env:OPENBLAS_NUM_THREADS = "2"; $env:NUMEXPR_NUM_THREADS = "2"
(Get-Process -Id $PID).PriorityClass = "BelowNormal"
$SCRIPT = "m2_paderborn_kr.py"
if (-not (Test-Path $PY)) { Write-Host "missing python: $PY" -ForegroundColor Red; exit 1 }
if (-not (Test-Path (Join-Path $Splits "motor2_100_ch1245_C1_fold1.npz"))) { Write-Host "missing split files in $Splits (use -Splits <dir>)" -ForegroundColor Red; exit 1 }
$EXP = @{ "m2_paderborn_kr.py" = "e86d3c0e737616eeb948e59e7941b67c1c8514c6995af7f170fd2bc991383005";
          "kr_modules.py"      = "7dfe0e59aed59963b03386fd78a0d00ef4966d2d776d6846112385f1b88a303f";
          "gpu_runtime.py"     = "ea887065b94e6c07692121bbd249012d36b808ebc914283050afc9d52d075046" }
foreach ($k in $EXP.Keys) {
    $h = (Get-FileHash $k -Algorithm SHA256).Hash.ToLower()
    Write-Host "$k sha256 $h"
    if ($h -ne $EXP[$k]) { Write-Host "STOP: $k is not the da-v1 delivery version (expected $($EXP[$k]))." -ForegroundColor Red; exit 1 }
}
if (-not $SkipGate -and -not (Test-Path "regression_PASS.txt")) {
    Write-Host "STOP: regression_PASS.txt not found. Run run_regression_check.ps1 first (addendum A1 section 7 item 3)." -ForegroundColor Red; exit 1
}
# Bg4 configuration (A1 section 3), unchanged: 600 supervised steps + 1500 contrastive steps (A1-D6)
$COMMON = @("--known","0,1","--unknown","2,3,4","--steps","1500","--warmup","300","--sup-steps","600","--device","cuda",
            "--k-score","denergy","--energy-T","0.05","--unk-init","kr","--known-subcenters","2","--guard-expand","0.2",
            "--label-balance-key","train_installation","--label-record-key","train_measurement",
            "--record-key","train_installation","--refresh-every","300","--gate-gamma","2",
            "--input","spec","--norm","window")
$GEN = @("--mode","generic","--contrast","plain","--vmd-bank","none","--known-guard","0.4","--guard-group-key","train_installation")
$COMBOS = @("C1","C2","C3"); $FOLDS = @(1,3)
function Run-One($tag, $c, $f, $s, $extra) {
    if (Test-Path "runs_paderborn\$tag\results_exp2.json") { Write-Host "skip (done): $tag"; return $true }
    $data = Join-Path $Splits "motor2_100_ch1245_${c}_fold$f.npz"
    Write-Host "===== $tag =====" -ForegroundColor Cyan
    & $PY $SCRIPT --data $data @COMMON @GEN @extra --eval-part T_eval --tag $tag --seeds $s *> "log_$tag.txt"
    if ($LASTEXITCODE -ne 0) { "$tag exit_code_$LASTEXITCODE" | Out-File -Append -Encoding ascii invalid_da1.txt; Write-Host "FAILED: $tag (logged, continuing)" -ForegroundColor Red; return $false }
    if (Select-String -Path "log_$tag.txt" -Pattern "loss=(nan|inf|-inf)" -Quiet) { "$tag nonfinite_loss" | Out-File -Append -Encoding ascii invalid_da1.txt; Write-Host "NON-FINITE LOSS: $tag (logged)" -ForegroundColor Red }
    return $true
}
$t0 = Get-Date
foreach ($s in $Seeds) {
    foreach ($c in $COMBOS) {
        foreach ($f in $FOLDS) {
            $null = Run-One "DA1_E0_${c}_f${f}_seed$s"  $c $f $s @()
            $null = Run-One "DA1_E0p_${c}_f${f}_seed$s" $c $f $s @("--use-adapt")
        }
    }
}
& $PY summarize_da1.py --root runs_paderborn --out summary_da1.csv
Write-Host "done in $([int]((Get-Date) - $t0).TotalMinutes) min: upload summary_da1.csv, invalid_da1.txt (if present), log_DA1_*.txt and the results_exp2.json of every runs_paderborn\DA1_* folder" -ForegroundColor Green
