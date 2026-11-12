# v19 centrifugal pump: NEW baseline input (channels 1,2,4,5 + log-spectrum front-end) and Chapter 3 on top of it.
# 3 combos x 3 folds x 3 seeds = 27 pairs per method, 4 methods = 108 runs (GPU). ASCII only.
# Needs: merge_v19.ps1 done (pool + VMD bank re-assembled, splits_ch1245 built). No VMD / heavy CPU work here.
# Usage: run_batch_v19_cf.ps1              -> all 4 methods (needs the VMD bank)
#        run_batch_v19_cf.ps1 -Methods B   -> only the new baseline (no VMD bank needed)
param([string[]]$Methods = @("B","A3g","A3d","A3c"))
$Methods = @($Methods | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })   # "-Methods B,A3g" via -File
$PY     = "D:\chuqishiyan1\zhubao\pump_reproduction\.venv_gpu\Scripts\python.exe"
$SCRIPT = "m2_paderborn_kr.py"
$SPLITS = "centrifugal\splits_ch1245"
$BANK   = "centrifugal\pool\pool_motor2_100_ch1245_vmd.npy"
$SEEDS  = 0..2
$COMBOS = @("C1","C2","C3")
$FOLDS  = 1..3
$env:OMP_NUM_THREADS = "4"; $env:MKL_NUM_THREADS = "4"
(Get-Process -Id $PID).PriorityClass = "BelowNormal"
$needBank = @($Methods | Where-Object { $_ -ne "B" }).Count -gt 0
if ($needBank -and -not (Test-Path $BANK)) {
    Write-Host "missing $BANK : download the VMD parts and run merge_v19.ps1, or run only the baseline with  -Methods B" -ForegroundColor Red; exit 1
}

$COMMON = @("--known","0,1","--unknown","2,3,4","--steps","1500","--warmup","300","--sup-steps","600","--device","cuda",
            "--k-score","denergy","--energy-T","0.05","--unk-init","kr","--known-subcenters","2","--known-guard","0.4","--guard-expand","0.2",
            "--label-balance-key","train_installation","--label-record-key","train_measurement",
            "--record-key","train_installation","--refresh-every","300","--gate-gamma","2",
            "--input","spec","--norm","window")
$G = @("--k-score-train","gated")
$methodDefs = @(
    @{ Name = "B";   Args = @("--mode","generic","--contrast","plain","--vmd-bank","none") },          # new baseline
    @{ Name = "A3g"; Args = @("--mode","adaptive_kr2","--contrast","plain","--vmd-bank",$BANK) + $G }, # Chapter 3 (energy-distance gated K)
    @{ Name = "A3d"; Args = @("--mode","adaptive_kr2","--contrast","plain","--vmd-bank",$BANK) },      # Chapter 3, K without energy gate
    @{ Name = "A3c"; Args = @("--mode","shuffled_kr2","--contrast","plain","--vmd-bank",$BANK) + $G }  # control: budgets shuffled
)
$runDefs = @($methodDefs | Where-Object { $Methods -contains $_.Name })
if ($runDefs.Count -eq 0) { Write-Host "no valid method in -Methods $Methods (use B, A3g, A3d, A3c)" -ForegroundColor Red; exit 1 }
$t0 = Get-Date
foreach ($c in $COMBOS) { foreach ($f in $FOLDS) { foreach ($m in $runDefs) { foreach ($s in $SEEDS) {
    $tag  = "V19_${c}_f${f}_$($m.Name)_seed$s"
    if (Test-Path "runs_paderborn\$tag\results_exp2.json") { continue }
    $data = "$SPLITS\motor2_100_ch1245_${c}_fold$f.npz"
    Write-Host "===== $tag (elapsed $([int]((Get-Date) - $t0).TotalMinutes) min) =====" -ForegroundColor Cyan
    & $PY $SCRIPT --data $data @COMMON @($m.Args) --tag $tag --seeds $s *> "log_$tag.txt"
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $tag (see log_$tag.txt)" -ForegroundColor Red; exit 1 }
} } } }
& $PY summarize_cf.py --prefix V19 --baselines B,A3d,A3c --out summary_v19_cf.txt --csv summary_v19_cf.csv
Write-Host "done in $([int]((Get-Date) - $t0).TotalMinutes) min: upload summary_v19_cf.txt and summary_v19_cf.csv to GitHub" -ForegroundColor Green
