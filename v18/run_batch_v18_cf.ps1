# v18 centrifugal pump (Bruinsma Motor-2 100 %, channel 4): Chapter 3 only. 3 combos x 3 folds x 3 seeds = 27 pairs. ASCII only.
# Methods: B (no Ch3), A3g (Ch3 + energy-distance gated K), A3d (Ch3, K without gate), A3c (Ch3 with shuffled budgets).
# Needs centrifugal\splits from centrifugal_splits_data_v1.zip and: pip install vmdpy. gpu_runtime.py must be the REAL one.
# Can run AFTER or IN PARALLEL with run_batch_v18_paderborn.ps1 (separate PowerShell window; slower if sharing one GPU).
$PY     = "D:\chuqishiyan1\zhubao\pump_reproduction\.venv_gpu\Scripts\python.exe"
$SCRIPT = "m2_paderborn_kr.py"
$SPLITS = "centrifugal\splits"
$SEEDS  = 0..2
$COMBOS = @("C1","C2","C3")
$FOLDS  = 1..3

& $PY build_vmd_bank.py (Get-ChildItem "$SPLITS\motor2_100_ch4_C*_fold*.npz" | Where-Object { $_.Name -notlike "*_vmd*" } | ForEach-Object { $_.FullName })
if ($LASTEXITCODE -ne 0) { Write-Host "VMD bank build FAILED (pip install vmdpy?)" -ForegroundColor Red; exit 1 }

$COMMON = @("--known","0,1","--unknown","2,3,4","--steps","1500","--warmup","300","--sup-steps","600","--device","cuda",
            "--k-score","denergy","--energy-T","0.05","--unk-init","kr","--known-subcenters","2","--known-guard","0.4","--guard-expand","0.2",
            "--label-balance-key","train_installation","--label-record-key","train_measurement",
            "--record-key","train_installation","--refresh-every","300","--gate-gamma","2")
$G = @("--k-score-train","gated")
$methods = @(
    @{ Name = "B";   Args = @("--mode","generic","--contrast","plain") },
    @{ Name = "A3g"; Args = @("--mode","adaptive_kr2","--contrast","plain") + $G },
    @{ Name = "A3d"; Args = @("--mode","adaptive_kr2","--contrast","plain") },
    @{ Name = "A3c"; Args = @("--mode","shuffled_kr2","--contrast","plain") + $G }
)
$t0 = Get-Date
foreach ($c in $COMBOS) { foreach ($f in $FOLDS) { foreach ($m in $methods) { foreach ($s in $SEEDS) {
    $tag  = "V18_${c}_f${f}_$($m.Name)_seed$s"
    if (Test-Path "runs_paderborn\$tag\results_exp2.json") { continue }
    $data = "$SPLITS\motor2_100_ch4_${c}_fold$f.npz"; $vmd = "$SPLITS\motor2_100_ch4_${c}_fold${f}_vmd.npy"
    Write-Host "===== $tag (elapsed $([int]((Get-Date) - $t0).TotalMinutes) min) =====" -ForegroundColor Cyan
    & $PY $SCRIPT --data $data --vmd-bank $vmd @COMMON @($m.Args) --tag $tag --seeds $s *> "log_$tag.txt"
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $tag (see log_$tag.txt)" -ForegroundColor Red; exit 1 }
} } } }
& $PY summarize_cf.py --prefix V18 --baselines B,A3d,A3c --out summary_v18_cf.txt --csv summary_v18_cf.csv
Write-Host "done in $([int]((Get-Date) - $t0).TotalMinutes) min: upload summary_v18_cf.txt and summary_v18_cf.csv to GitHub" -ForegroundColor Green
