# v23 centrifugal pump (GPU): installation mixing (Mx) and its matched control (Rg). ASCII only.
# Uses the data already on this PC: centrifugal\splits_ch1245 and the LOCALLY built VMD bank. Nothing to download.
# 3 combos x 3 folds x 3 seeds = 27 pairs per method, 54 runs. Results are tagged V20_* so that they pair with the v20 runs (Bg4).
param([string[]]$Methods = @("Rg","Mx"))
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
if (-not (Test-Path $BANK)) { Write-Host "missing $BANK (run build_vmd_local.ps1)" -ForegroundColor Red; exit 1 }

$COMMON = @("--known","0,1","--unknown","2,3,4","--steps","1500","--warmup","300","--sup-steps","600","--device","cuda",
            "--k-score","denergy","--energy-T","0.05","--unk-init","kr","--known-subcenters","2","--guard-expand","0.2",
            "--label-balance-key","train_installation","--label-record-key","train_measurement",
            "--record-key","train_installation","--refresh-every","300","--gate-gamma","2",
            "--input","spec","--norm","window")
$KR2 = @("--mode","adaptive_kr2","--contrast","plain","--vmd-bank",$BANK,"--k-score-train","gated",
         "--known-guard","0.4","--guard-group-key","train_installation")
$methodDefs = @(
    @{ Name = "Rg"; Args = $KR2 },                                  # matched control: random independent mode perturbation (A3g behaviour) + LOIO guard
    @{ Name = "Mx"; Args = $KR2 + @("--mix-aug","install") }        # installation mixing + LOIO guard
)
$runDefs = @($methodDefs | Where-Object { $Methods -contains $_.Name })
if ($runDefs.Count -eq 0) { Write-Host "no valid method in -Methods $Methods (use Rg, Mx)" -ForegroundColor Red; exit 1 }
$t0 = Get-Date
foreach ($c in $COMBOS) { foreach ($f in $FOLDS) { foreach ($m in $runDefs) { foreach ($s in $SEEDS) {
    $tag  = "V20_${c}_f${f}_$($m.Name)_seed$s"
    if (Test-Path "runs_paderborn\$tag\results_exp2.json") { continue }
    $data = "$SPLITS\motor2_100_ch1245_${c}_fold$f.npz"
    Write-Host "===== $tag (elapsed $([int]((Get-Date) - $t0).TotalMinutes) min) =====" -ForegroundColor Cyan
    & $PY $SCRIPT --data $data @COMMON @($m.Args) --tag $tag --seeds $s *> "log_$tag.txt"
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $tag (see log_$tag.txt)" -ForegroundColor Red; exit 1 }
} } } }
& $PY summarize_cf.py --prefix V20 --baselines Bg4,Rg --out summary_v23_cf.txt --csv summary_v23_cf.csv
Write-Host "done in $([int]((Get-Date) - $t0).TotalMinutes) min: upload summary_v23_cf.txt and summary_v23_cf.csv to GitHub" -ForegroundColor Green
