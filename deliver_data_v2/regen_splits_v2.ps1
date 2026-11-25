# Regenerate the 9 splits with rule R1-merged (R1m) and verify them. ASCII only on purpose (Windows PowerShell 5.1).
# Usage (from the deliver_data_v2 folder):
#   powershell -ExecutionPolicy Bypass -File regen_splits_v2.ps1
#   powershell -ExecutionPolicy Bypass -File regen_splits_v2.ps1 -Dest D:\DCWRU_Data\Paderborn_Experiment -Py <python.exe>
param(
    [string]$Dest = "D:\DCWRU_Data\Paderborn_Experiment",
    [string]$Py   = "D:\chuqishiyan1\zhubao\pump_reproduction\.venv_gpu\Scripts\python.exe"
)
$ErrorActionPreference = "Continue"   # native stderr must not abort the script in PS 5.1; failures are checked via exit codes
$root = $PSScriptRoot
$log  = Join-Path $Dest "regen_splits_v2_log.txt"
$script:failed = $false

function Log([string]$msg) {
    Write-Host $msg
    Add-Content -Path $log -Value $msg -Encoding UTF8
}
function Step([string]$name, [bool]$ok, [string]$detail) {
    if ($ok) { Log ("OK    " + $name + " " + $detail) }
    else     { Log ("FAIL  " + $name + " " + $detail); $script:failed = $true }
}

New-Item -ItemType Directory -Force -Path $Dest | Out-Null
Set-Content -Path $log -Value ("regen_splits_v2 started " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss")) -Encoding UTF8
Log ("Dest = " + $Dest)
Log ("Python = " + $Py)
Log ("PSVersion = " + $PSVersionTable.PSVersion.ToString())

$poolDir  = Join-Path $Dest "centrifugal\pool"
$pool     = Join-Path $poolDir "pool_motor2_100_ch1245.npz"
$logspec  = Join-Path $poolDir "pool_motor2_100_ch1245_logspec.npy"
$out      = Join-Path $Dest "centrifugal\splits_da_v2"
$makePy   = Join-Path $root "tools\make_splits_da_v2.py"
$verPy    = Join-Path $root "tools\verify_splits_v2.py"
$pjDir    = Join-Path $root "tools\phase_jump_meta\Motor-2_100"
$tsJson   = Join-Path $root "target_split\target_split_v1.json"
$expSha   = Join-Path $root "target_split\split_assignment_r1m_v1.sha256"

foreach ($f in @($Py, $pool, $logspec, $makePy, $verPy, $tsJson, $expSha)) {
    Step ("exists " + (Split-Path $f -Leaf)) (Test-Path $f) ""
}
if ($script:failed) { Log "FAILED: missing input file(s); run merge_data_v1.ps1 from the v1 delivery first"; exit 1 }

Log "--- generating splits (rule R1m) ---"
$o = & $Py $makePy --pool $pool --logspec $logspec --out $out --phase-jump-dir $pjDir --target-split $tsJson 2>&1
foreach ($line in $o) { Log ("" + $line) }
Step "make_splits_da_v2.py exit code" ($LASTEXITCODE -eq 0) ("= " + $LASTEXITCODE)

if (-not $script:failed) {
    Log "--- verifying splits ---"
    $o = & $Py $verPy --splits $out --pool $pool --phase-jump-dir $pjDir --expected-sha $expSha 2>&1
    foreach ($line in $o) { Log ("" + $line) }
    Step "verify_splits_v2.py exit code" ($LASTEXITCODE -eq 0) ("= " + $LASTEXITCODE)
}

if ($script:failed) { Log "RESULT: FAILED"; exit 1 }
Log "RESULT: ALL OK"
Log ("log written to " + $log)
exit 0
