# v24s2 centrifugal pump (GPU): exploratory dose and allocation screen. ASCII only.
# New arms (tags V20_*, seed 0 only): Rk2, Rk4 (adaptive allocation, gain 2 / 4), Rks2, Rks4 (random allocation of the per-window budget, gain 2 / 4).
# 4 arms x 9 splits x seed 0 = 36 runs. Reuses V20_*_Rk_seed0 (v24) and V20_*_Bg4_seed0 (v20). Rough time estimate 40 to 70 minutes (not measured).
# A failed or non-finite run is logged to invalid_v24s2.txt and the batch continues; do not re-run it by hand.
param([string[]]$Methods = @("Rk2","Rk4","Rks2","Rks4"))
$Methods = @($Methods | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$PY     = "D:\chuqishiyan1\zhubao\pump_reproduction\.venv_gpu\Scripts\python.exe"
$SCRIPT = "m2_paderborn_kr.py"
$SPLITS = "centrifugal\splits_ch1245"
$BANK   = "centrifugal\pool\pool_motor2_100_ch1245_vmd.npy"
$COMBOS = @("C1","C2","C3")
$FOLDS  = 1..3
$env:OMP_NUM_THREADS = "4"; $env:MKL_NUM_THREADS = "4"
(Get-Process -Id $PID).PriorityClass = "BelowNormal"
if (-not (Test-Path "$SPLITS\motor2_100_ch1245_C1_fold1.npz")) { Write-Host "missing $SPLITS" -ForegroundColor Red; exit 1 }
if (-not (Test-Path $BANK)) { Write-Host "missing $BANK" -ForegroundColor Red; exit 1 }
$EXP = @{ "m2_paderborn_kr.py" = "6623ea074c5a8a86b83aa37055e3ec63485479fc0bc5fe7facfa231978ea6f73";
          "kr_modules.py"      = "7dfe0e59aed59963b03386fd78a0d00ef4966d2d776d6846112385f1b88a303f" }
foreach ($k in $EXP.Keys) {
    $h = (Get-FileHash $k -Algorithm SHA256).Hash.ToLower()
    Write-Host "$k sha256 $h"
    if ($h -ne $EXP[$k]) { Write-Host "STOP: $k is not the v24s2 version (expected $($EXP[$k])). Replace it with the file from this delivery." -ForegroundColor Red; exit 1 }
}
$COMMON = @("--known","0,1","--unknown","2,3,4","--steps","1500","--warmup","300","--sup-steps","600","--device","cuda",
            "--k-score","denergy","--energy-T","0.05","--unk-init","kr","--known-subcenters","2","--guard-expand","0.2",
            "--label-balance-key","train_installation","--label-record-key","train_measurement",
            "--record-key","train_installation","--refresh-every","300","--gate-gamma","2",
            "--input","spec","--norm","window")
$KR2 = @("--contrast","plain","--vmd-bank",$BANK,"--k-score-train","knn","--known-guard","0.4","--guard-group-key","train_installation")
$methodDefs = @(
    @{ Name = "Rk2";  Args = @("--mode","adaptive_kr2") + $KR2 + @("--kr2-gain","2") },
    @{ Name = "Rk4";  Args = @("--mode","adaptive_kr2") + $KR2 + @("--kr2-gain","4") },
    @{ Name = "Rks2"; Args = @("--mode","shuffled_kr2","--kr2-shuf-scope","sample") + $KR2 + @("--kr2-gain","2") },
    @{ Name = "Rks4"; Args = @("--mode","shuffled_kr2","--kr2-shuf-scope","sample") + $KR2 + @("--kr2-gain","4") }
)
$runDefs = @($methodDefs | Where-Object { $Methods -contains $_.Name })
if ($runDefs.Count -eq 0) { Write-Host "no valid method in -Methods $Methods (use Rk2, Rk4, Rks2, Rks4)" -ForegroundColor Red; exit 1 }
$t0 = Get-Date
foreach ($c in $COMBOS) { foreach ($f in $FOLDS) { foreach ($m in $runDefs) {
    $s = 0
    $tag  = "V20_${c}_f${f}_$($m.Name)_seed$s"
    if (Test-Path "runs_paderborn\$tag\results_exp2.json") { continue }
    $data = "$SPLITS\motor2_100_ch1245_${c}_fold$f.npz"
    Write-Host "===== $tag (elapsed $([int]((Get-Date) - $t0).TotalMinutes) min) =====" -ForegroundColor Cyan
    & $PY $SCRIPT --data $data @COMMON @($m.Args) --tag $tag --seeds $s *> "log_$tag.txt"
    if ($LASTEXITCODE -ne 0) { "$tag exit_code_$LASTEXITCODE" | Out-File -Append -Encoding utf8 invalid_v24s2.txt; Write-Host "FAILED: $tag (logged, continuing)" -ForegroundColor Red; continue }
    if (Select-String -Path "log_$tag.txt" -Pattern "loss=(nan|inf|-inf)" -Quiet) { "$tag nonfinite_loss" | Out-File -Append -Encoding utf8 invalid_v24s2.txt; Write-Host "NON-FINITE LOSS: $tag (logged)" -ForegroundColor Red }
} } }
& $PY analyze_v24s2.py --root runs_paderborn --out summary_v24s2.txt --csv summary_v24s2.csv
Write-Host "done in $([int]((Get-Date) - $t0).TotalMinutes) min: upload summary_v24s2.txt, summary_v24s2.csv, pairing_report_v24s2.txt, invalid_v24s2.txt (if present) and the log_V20_*_Rk[s]*_seed0.txt files to GitHub" -ForegroundColor Green
