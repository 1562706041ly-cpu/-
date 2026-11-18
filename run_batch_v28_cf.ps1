# v28 centrifugal pump (GPU): exploratory screen of band-gain augmentation. ASCII only.
# Pre-registration: v28_endtoend_PREREGISTERED.txt (branch v28-prereg, pushed 2026-11-18T14:33:36Z, before any code or run).
# Step 0: GPU consistency check of the reused Bg4 (new code, default band flags, C1 fold 1, seed 0, tag V28CHK_C1_f1_Bg4_seed0) against V20_C1_f1_Bg4_seed0, tolerance |dH| <= 0.005.
#         PASS -> old Bg4 is reused.  FAIL -> all 9 Bg4 are re-run with the new code (tags V28_Cx_fy_Bg4_seed0) and used for every comparison.
# Step 1: 3 arms x 9 splits x seed 0 = 27 runs: BI65 (independent, gain 0.65), BC2 (coordinated, gain 2.0), BI2 (independent, gain 2.0).
# A failed or non-finite run is logged to invalid_v28.txt and the batch continues. Do not re-run by hand, do not drop a run because of its H.
# Rough time: about 1 to 1.5 hours (estimate, not measured).
$PY     = "D:\chuqishiyan1\zhubao\pump_reproduction\.venv_gpu\Scripts\python.exe"
$SCRIPT = "m2_paderborn_kr.py"
$SPLITS = "centrifugal\splits_ch1245"
$env:OMP_NUM_THREADS = "4"; $env:MKL_NUM_THREADS = "4"
(Get-Process -Id $PID).PriorityClass = "BelowNormal"
if (-not (Test-Path "$SPLITS\motor2_100_ch1245_C1_fold1.npz")) { Write-Host "missing $SPLITS" -ForegroundColor Red; exit 1 }
$EXP = @{ "m2_paderborn_kr.py" = "2e2c9f19a6bbde5cc097205d740b424e419b3010320acd23b693faeb743cb3af";
          "kr_modules.py"      = "7dfe0e59aed59963b03386fd78a0d00ef4966d2d776d6846112385f1b88a303f" }
foreach ($k in $EXP.Keys) {
    $h = (Get-FileHash $k -Algorithm SHA256).Hash.ToLower()
    Write-Host "$k sha256 $h"
    if ($h -ne $EXP[$k]) { Write-Host "STOP: $k is not the v28 version (expected $($EXP[$k])). Replace it with the file from this delivery." -ForegroundColor Red; exit 1 }
}
$COMMON = @("--known","0,1","--unknown","2,3,4","--steps","1500","--warmup","300","--sup-steps","600","--device","cuda",
            "--k-score","denergy","--energy-T","0.05","--unk-init","kr","--known-subcenters","2","--guard-expand","0.2",
            "--label-balance-key","train_installation","--label-record-key","train_measurement",
            "--record-key","train_installation","--refresh-every","300","--gate-gamma","2",
            "--input","spec","--norm","window")
$GEN = @("--mode","generic","--contrast","plain","--vmd-bank","none","--known-guard","0.4","--guard-group-key","train_installation")
$COMBOS = @("C1","C2","C3"); $FOLDS = 1..3; $s = 0
function Run-One($tag, $c, $f, $extra) {
    if (Test-Path "runs_paderborn\$tag\results_exp2.json") { return $true }
    $data = "$SPLITS\motor2_100_ch1245_${c}_fold$f.npz"
    Write-Host "===== $tag =====" -ForegroundColor Cyan
    & $PY $SCRIPT --data $data @COMMON @GEN @extra --tag $tag --seeds $s *> "log_$tag.txt"
    if ($LASTEXITCODE -ne 0) { "$tag exit_code_$LASTEXITCODE" | Out-File -Append -Encoding utf8 invalid_v28.txt; Write-Host "FAILED: $tag (logged, continuing)" -ForegroundColor Red; return $false }
    if (Select-String -Path "log_$tag.txt" -Pattern "loss=(nan|inf|-inf)" -Quiet) { "$tag nonfinite_loss" | Out-File -Append -Encoding utf8 invalid_v28.txt; Write-Host "NON-FINITE LOSS: $tag (logged)" -ForegroundColor Red }
    return $true
}
$t0 = Get-Date
# ---- step 0: GPU consistency check of the reused Bg4
$null = Run-One "V28CHK_C1_f1_Bg4_seed0" "C1" 1 @()
& $PY analyze_v28.py --check-only --root runs_paderborn
$rerunBg4 = ($LASTEXITCODE -ne 0)
if ($rerunBg4) {
    Write-Host "GPU check FAILED or could not be evaluated: re-running all 9 Bg4 with the new code (tags V28_*_Bg4_seed0)" -ForegroundColor Yellow
    foreach ($c in $COMBOS) { foreach ($f in $FOLDS) { $null = Run-One "V28_${c}_f${f}_Bg4_seed0" $c $f @() } }
}
# ---- step 1: the three new arms
$arms = @( @{ Name = "BI65"; Args = @("--aug-band","indep","--band-gain","0.65") },
           @{ Name = "BC2";  Args = @("--aug-band","coord","--band-gain","2.0") },
           @{ Name = "BI2";  Args = @("--aug-band","indep","--band-gain","2.0") } )
foreach ($c in $COMBOS) { foreach ($f in $FOLDS) { foreach ($a in $arms) { $null = Run-One "V28_${c}_f${f}_$($a.Name)_seed0" $c $f $a.Args } } }
& $PY analyze_v28.py --root runs_paderborn --splits $SPLITS --out summary_v28.txt --csv summary_v28.csv
Write-Host "done in $([int]((Get-Date) - $t0).TotalMinutes) min: upload summary_v28.txt, summary_v28.csv, pairing_report_v28.txt, bg4_check_v28.txt, invalid_v28.txt (if present), log_V28*.txt and the feat_gcd_PADERBORN_seed0.npz of every V28* and V20_*_Bg4_seed0 run folder (or at least the V28* ones) to GitHub" -ForegroundColor Green
