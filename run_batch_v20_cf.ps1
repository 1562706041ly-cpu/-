# v20 centrifugal pump (GPU): known-guard calibration + Chapter 3 energy-directed VMD perturbation. ASCII only.
# Uses the v19 data already on this PC: centrifugal\splits_ch1245 and the LOCALLY built VMD bank. Nothing to download.
# 3 combos x 3 folds x 3 seeds = 27 pairs per method.
# Usage: run_batch_v20_cf.ps1                 -> all methods
#        run_batch_v20_cf.ps1 -Methods B,Bg4  -> a subset
param([string[]]$Methods = @("B","Bg3","Bg4","Bg5","EDdg","EDrg"))
$Methods = @($Methods | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$PY     = "D:\chuqishiyan1\zhubao\pump_reproduction\.venv_gpu\Scripts\python.exe"
$SCRIPT = "m2_paderborn_kr.py"
$SPLITS = "centrifugal\splits_ch1245"
$BANK   = "centrifugal\pool\pool_motor2_100_ch1245_vmd.npy"
$SEEDS  = 0..2
$COMBOS = @("C1","C2","C3")
$FOLDS  = 1..3
$env:OMP_NUM_THREADS = "4"; $env:MKL_NUM_THREADS = "4"
(Get-Process -Id $PID).PriorityClass = "BelowNormal"
if (-not (Test-Path "$SPLITS\motor2_100_ch1245_C1_fold1.npz")) { Write-Host "missing $SPLITS (v19 data)" -ForegroundColor Red; exit 1 }
$needBank = @($Methods | Where-Object { $_ -like "ED*" }).Count -gt 0
if ($needBank -and -not (Test-Path $BANK)) { Write-Host "missing $BANK (run build_vmd_local.ps1)" -ForegroundColor Red; exit 1 }

$COMMON = @("--known","0,1","--unknown","2,3,4","--steps","1500","--warmup","300","--sup-steps","600","--device","cuda",
            "--k-score","denergy","--energy-T","0.05","--unk-init","kr","--known-subcenters","2","--guard-expand","0.2",
            "--label-balance-key","train_installation","--label-record-key","train_measurement",
            "--record-key","train_installation","--refresh-every","300","--gate-gamma","2",
            "--input","spec","--norm","window")
$GG  = @("--guard-group-key","train_installation")
$KR2 = @("--mode","adaptive_kr2","--contrast","plain","--vmd-bank",$BANK,"--k-score-train","gated","--edir-lam","0.5")
$methodDefs = @(
    @{ Name = "B";    Args = @("--mode","generic","--contrast","plain","--vmd-bank","none","--known-guard","0.4") },        # v19 baseline
    @{ Name = "Bg3";  Args = @("--mode","generic","--contrast","plain","--vmd-bank","none","--known-guard","0.3") + $GG }, # guard: leave-one-installation-out
    @{ Name = "Bg4";  Args = @("--mode","generic","--contrast","plain","--vmd-bank","none","--known-guard","0.4") + $GG },
    @{ Name = "Bg5";  Args = @("--mode","generic","--contrast","plain","--vmd-bank","none","--known-guard","0.5") + $GG },
    @{ Name = "EDdg"; Args = $KR2 + @("--edir","dist",  "--known-guard","0.4") + $GG },   # Chapter 3: distance-energy direction
    @{ Name = "EDrg"; Args = $KR2 + @("--edir","random","--known-guard","0.4") + $GG }    # control: random direction, same magnitudes
)
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
& $PY summarize_cf.py --prefix V20 --baselines B,Bg4,EDrg --out summary_v20_cf.txt --csv summary_v20_cf.csv
Write-Host "done in $([int]((Get-Date) - $t0).TotalMinutes) min: upload summary_v20_cf.txt and summary_v20_cf.csv to GitHub" -ForegroundColor Green
