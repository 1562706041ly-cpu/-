# v24 centrifugal pump (GPU): knownness K in Chapter-3 training. ASCII only.
# New methods (tags V20_*, so they pair with the existing v23 Rg runs):
#   Rk    = pure nearest-neighbour K        (--k-score-train knn)
#   Rd    = distance-energy K (A3d)         (no --k-score-train)
#   Rgrep = repeat of Rg on 3 splits, seed 0 (GPU noise floor)
# Existing Rg runs (V20_*_Rg) are reused. 27 + 27 + 3 = 57 new runs. Expect roughly 1 to 1.5 hours (estimate from the v22 batch, not measured).
param([string[]]$Methods = @("Rk","Rd","Rgrep"))
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
if (-not (Test-Path "$SPLITS\motor2_100_ch1245_C1_fold1.npz")) { Write-Host "missing $SPLITS" -ForegroundColor Red; exit 1 }
if (-not (Test-Path $BANK)) { Write-Host "missing $BANK (run build_vmd_local.ps1)" -ForegroundColor Red; exit 1 }
# code must be the v23 version, the one that produced the Rg runs
$EXP = @{ "m2_paderborn_kr.py" = "3f5ffb6a13507caabe7e0542f9f8b66f0fd9469a21b4d0de33674ba286443827";
          "kr_modules.py"      = "7dfe0e59aed59963b03386fd78a0d00ef4966d2d776d6846112385f1b88a303f" }
foreach ($k in $EXP.Keys) {
    $h = (Get-FileHash $k -Algorithm SHA256).Hash.ToLower()
    Write-Host "$k sha256 $h"
    if ($h -ne $EXP[$k]) { Write-Host "STOP: $k is not the v23 version (expected $($EXP[$k])). Do not run; tell Kiro." -ForegroundColor Red; exit 1 }
}
$COMMON = @("--known","0,1","--unknown","2,3,4","--steps","1500","--warmup","300","--sup-steps","600","--device","cuda",
            "--k-score","denergy","--energy-T","0.05","--unk-init","kr","--known-subcenters","2","--guard-expand","0.2",
            "--label-balance-key","train_installation","--label-record-key","train_measurement",
            "--record-key","train_installation","--refresh-every","300","--gate-gamma","2",
            "--input","spec","--norm","window")
$KR2 = @("--mode","adaptive_kr2","--contrast","plain","--vmd-bank",$BANK,
         "--known-guard","0.4","--guard-group-key","train_installation")
$methodDefs = @(
    @{ Name = "Rk";    Args = $KR2 + @("--k-score-train","knn") },
    @{ Name = "Rd";    Args = $KR2 },
    @{ Name = "Rgrep"; Args = $KR2 + @("--k-score-train","gated") }
)
$repSplits = @("C1_1","C2_2","C3_3")
$runDefs = @($methodDefs | Where-Object { $Methods -contains $_.Name })
if ($runDefs.Count -eq 0) { Write-Host "no valid method in -Methods $Methods (use Rk, Rd, Rgrep)" -ForegroundColor Red; exit 1 }
$t0 = Get-Date
foreach ($c in $COMBOS) { foreach ($f in $FOLDS) { foreach ($m in $runDefs) { foreach ($s in $SEEDS) {
    if ($m.Name -eq "Rgrep" -and (($repSplits -notcontains "${c}_${f}") -or $s -ne 0)) { continue }
    $tag  = "V20_${c}_f${f}_$($m.Name)_seed$s"
    if (Test-Path "runs_paderborn\$tag\results_exp2.json") { continue }
    $data = "$SPLITS\motor2_100_ch1245_${c}_fold$f.npz"
    Write-Host "===== $tag (elapsed $([int]((Get-Date) - $t0).TotalMinutes) min) =====" -ForegroundColor Cyan
    & $PY $SCRIPT --data $data @COMMON @($m.Args) --tag $tag --seeds $s *> "log_$tag.txt"
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $tag (see log_$tag.txt)" -ForegroundColor Red; exit 1 }
} } } }
& $PY analyze_v24.py --root runs_paderborn --out summary_v24.txt --csv summary_v24.csv
Write-Host "done in $([int]((Get-Date) - $t0).TotalMinutes) min: upload summary_v24.txt and summary_v24.csv (and pairing_report_v24.txt) to GitHub" -ForegroundColor Green
