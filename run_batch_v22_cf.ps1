# v22 centrifugal pump (GPU): known guard with distance fusion (D1: deep kNN + hand-crafted-feature kNN, D2: leave-one-installation-out calibration).
# ASCII only. Uses the data already on this PC (v19): centrifugal\splits_ch1245. No VMD needed. Nothing to download.
# New methods are tagged V20_* so they pair with the existing v20 runs (B, Bg4): same data, seeds, settings.
# Gd  = deep kNN only (control)   Gfs = deep + spectral-band hand features   Gfa = deep + all 58 hand features per channel
# 3 combos x 3 folds x 3 seeds = 27 pairs per method, 81 runs, about 1 hour.
# If B / Bg4 are not on this PC yet:  run_batch_v22_cf.ps1 -Methods B,Bg4,Gd,Gfs,Gfa   (v20 runs are skipped when present)
param([string[]]$Methods = @("Gd","Gfs","Gfa"))
$Methods = @($Methods | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$PY     = "D:\chuqishiyan1\zhubao\pump_reproduction\.venv_gpu\Scripts\python.exe"
$SCRIPT = "m2_paderborn_kr.py"
$SPLITS = "centrifugal\splits_ch1245"
$SEEDS  = 0..2
$COMBOS = @("C1","C2","C3")
$FOLDS  = 1..3
$env:OMP_NUM_THREADS = "4"; $env:MKL_NUM_THREADS = "4"
(Get-Process -Id $PID).PriorityClass = "BelowNormal"
if (-not (Test-Path "$SPLITS\motor2_100_ch1245_C1_fold1.npz")) { Write-Host "missing $SPLITS (v19 data)" -ForegroundColor Red; exit 1 }

$COMMON = @("--known","0,1","--unknown","2,3,4","--steps","1500","--warmup","300","--sup-steps","600","--device","cuda",
            "--k-score","denergy","--energy-T","0.05","--unk-init","kr","--known-subcenters","2","--guard-expand","0.2",
            "--label-balance-key","train_installation","--label-record-key","train_measurement",
            "--record-key","train_installation","--refresh-every","300","--gate-gamma","2",
            "--input","spec","--norm","window")
$GEN = @("--mode","generic","--contrast","plain","--vmd-bank","none","--known-guard","0.4")
$GG  = @("--guard-group-key","train_installation")
$methodDefs = @(
    @{ Name = "B";   Args = $GEN },                                                       # v19 baseline (old guard)
    @{ Name = "Bg4"; Args = $GEN + $GG },                                                 # v20 guard: leave-one-installation-out
    @{ Name = "Gd";  Args = $GEN + $GG + @("--guard-score","knn") },                      # control: deep kNN, z-scored, LOIO
    @{ Name = "Gfs"; Args = $GEN + $GG + @("--guard-score","knn+spec") },                 # + spectral-band hand features
    @{ Name = "Gfa"; Args = $GEN + $GG + @("--guard-score","knn+all") }                   # + all 58 hand features per channel
)
# the old B has no --guard-group-key; v20 used "--known-guard 0.4" only -> identical to $GEN
$runDefs = @($methodDefs | Where-Object { $Methods -contains $_.Name })
if ($runDefs.Count -eq 0) { Write-Host "no valid method in -Methods $Methods" -ForegroundColor Red; exit 1 }
$t0 = Get-Date
foreach ($c in $COMBOS) { foreach ($f in $FOLDS) { foreach ($m in $runDefs) { foreach ($s in $SEEDS) {
    $tag  = "V20_${c}_f${f}_$($m.Name)_seed$s"
    if (Test-Path "runs_paderborn\$tag\results_exp2.json") { continue }
    $data = "$SPLITS\motor2_100_ch1245_${c}_fold$f.npz"
    Write-Host "===== $tag (elapsed $([int]((Get-Date) - $t0).TotalMinutes) min) =====" -ForegroundColor Cyan
    & $PY $SCRIPT --data $data @COMMON @($m.Args) --tag $tag --seeds $s *> "log_$tag.txt"
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $tag (see log_$tag.txt)" -ForegroundColor Red; exit 1 }
} } } }
& $PY summarize_cf.py --prefix V20 --baselines B,Bg4,Gd --out summary_v22_cf.txt --csv summary_v22_cf.csv
Write-Host "done in $([int]((Get-Date) - $t0).TotalMinutes) min: upload summary_v22_cf.txt and summary_v22_cf.csv to GitHub" -ForegroundColor Green
