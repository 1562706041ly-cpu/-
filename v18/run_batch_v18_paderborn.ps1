# v18 Paderborn: add seeds 6..11 to P_B / P_A3g / P_A3d / P_A3c (seeds 0..5 already exist locally from v17). ASCII only.
# Purpose: does the energy-distance gated K (A3g) beat the energy-free K (A3d) stably over 12 seeds?
# Run in D:\DCWRU_Data\Paderborn_Experiment\ ; gpu_runtime.py must be the REAL one.
$PY     = "D:\chuqishiyan1\zhubao\pump_reproduction\.venv_gpu\Scripts\python.exe"
$SCRIPT = "m2_paderborn_kr.py"
$DATA   = "data\paderborn_singlecondition_n15_m07_f10_v1.npz"
$VMD    = "data\paderborn_singlecondition_n15_m07_f10_v1_vmd.npy"
$SEEDS  = 6..11
$BASE = @("--k-score","denergy","--energy-T","0.05","--unk-init","kr","--known-subcenters","2","--n-unknown-clusters","2",
          "--known-guard","0.4","--guard-expand","0.2","--label-balance-key","train_bearing_id","--label-record-key","train_source_file",
          "--record-key","train_source_file","--refresh-every","300","--gate-gamma","2")
$G = @("--k-score-train","gated")
$experiments = @(
    @{ Name = "P_B";   Args = @("--mode","generic","--contrast","plain") },
    @{ Name = "P_A3g"; Args = @("--mode","adaptive_kr2","--contrast","plain") + $G },
    @{ Name = "P_A3d"; Args = @("--mode","adaptive_kr2","--contrast","plain") },
    @{ Name = "P_A3c"; Args = @("--mode","shuffled_kr2","--contrast","plain") + $G }
)
$t0 = Get-Date
foreach ($e in $experiments) { foreach ($s in $SEEDS) {
    $tag = "$($e.Name)_seed$s"
    if (Test-Path "runs_paderborn\$tag\results_exp2.json") { continue }
    Write-Host "===== $tag (elapsed $([int]((Get-Date) - $t0).TotalMinutes) min) =====" -ForegroundColor Cyan
    & $PY $SCRIPT --data $DATA --vmd-bank $VMD @BASE @($e.Args) --tag $tag --seeds $s --steps 1500 --warmup 300 --sup-steps 600 --device cuda *> "log_$tag.txt"
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $tag (see log_$tag.txt)" -ForegroundColor Red; exit 1 }
} }
# summaries use ALL seeds found (0..11)
& $PY summarize_runs.py --exps P_B P_A3d P_A3c P_A3g --baseline P_A3d | Out-File -Encoding utf8 summary_v18_paderborn_vs_A3d.txt
& $PY summarize_runs.py --exps P_B P_A3d P_A3c P_A3g --baseline P_B   | Out-File -Encoding utf8 summary_v18_paderborn_vs_B.txt
& $PY summarize_runs.py --exps P_A3c P_A3g --baseline P_A3c          | Out-File -Encoding utf8 summary_v18_paderborn_vs_A3c.txt
Write-Host "done in $([int]((Get-Date) - $t0).TotalMinutes) min: upload summary_v18_paderborn_*.txt to GitHub" -ForegroundColor Green
