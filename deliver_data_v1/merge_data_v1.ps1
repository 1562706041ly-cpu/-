# Step 1 data delivery v1: re-assemble the pool (and, if needed, the VMD bank) from <100 MB parts, check sha256,
# copy the log spectrum and the target split files, build the 9 splits with T_adapt / T_eval (rule R1).
# Files come from branch data-da-v1; the VMD parts (centrifugal\pool_vmd) come from branch data-da-v1-vmd
# and are only needed when no correct VMD bank exists under <Dest>\centrifugal\pool.
# Run from the unzipped deliver_data_v1 folder:
#   powershell -ExecutionPolicy Bypass -File merge_data_v1.ps1
#   powershell -ExecutionPolicy Bypass -File merge_data_v1.ps1 -Dest D:\DCWRU_Data\Paderborn_Experiment -Py <python.exe>
param([string]$Dest = "D:\DCWRU_Data\Paderborn_Experiment",
      [string]$Py   = "D:\chuqishiyan1\zhubao\pump_reproduction\.venv_gpu\Scripts\python.exe")
$ErrorActionPreference = "Stop"
$src  = $PSScriptRoot
$pool = Join-Path $Dest "centrifugal\pool"
$tsd  = Join-Path $Dest "centrifugal\target_split_v1"
$spl  = Join-Path $Dest "centrifugal\splits_da_v1"
$log  = New-Object System.Collections.Generic.List[string]
$problems = New-Object System.Collections.Generic.List[string]

function Say([string]$msg, [string]$color = "Gray") { Write-Host $msg -ForegroundColor $color; $log.Add($msg) }
function Sha([string]$path) { return (Get-FileHash $path -Algorithm SHA256).Hash.ToLower() }
function Fail([string]$msg) {
    Say "FAILED: $msg" "Red"
    try { $log | Out-File -FilePath (Join-Path $Dest "merge_data_v1_log.txt") -Encoding utf8 } catch { }
    exit 1
}
function SrcPath([string]$rel) { return (Join-Path $src ($rel -replace '/', '\')) }

# Merge the parts of one entry of parts_sha256.json into $out. Returns
#   "ok"       $out exists with the right sha256 (already present or merged now),
#   "missing"  $out does not exist and no parts were downloaded,
#   "mismatch" $out exists with a WRONG sha256 and no parts were downloaded ($out is left unchanged).
function Assemble($entry, [string]$out) {
    $want = $entry.sha256
    $parts = @($entry.parts)
    $have = @($parts | Where-Object { Test-Path (SrcPath $_.file) })
    if (Test-Path $out) {
        if ((Sha $out) -eq $want) { Say "OK   $out (already present, sha256 matches)" "Green"; return "ok" }
        if ($have.Count -eq 0) { return "mismatch" }
        Say "     $out exists but its sha256 differs: it will be rebuilt from the parts" "Yellow"
    }
    if ($have.Count -eq 0) { return "missing" }
    if ($have.Count -ne $parts.Count) { Fail "only $($have.Count) of $($parts.Count) parts of $out found; download all parts" }
    foreach ($p in $parts) {
        $f = SrcPath $p.file
        if ((Get-Item $f).Length -ne [int64]$p.size_bytes) { Fail "wrong size of $f (download incomplete?)" }
        if ((Sha $f) -ne $p.sha256) { Fail "sha256 mismatch of part $f" }
    }
    $tmp = "$out.tmp"
    $o = [System.IO.File]::Create($tmp)
    try {
        foreach ($p in $parts) {
            $i = [System.IO.File]::OpenRead((SrcPath $p.file))
            try { $i.CopyTo($o) } finally { $i.Close() }
        }
    } finally { $o.Close() }
    if ((Sha $tmp) -ne $want) { Remove-Item $tmp -Force; Fail "sha256 mismatch of merged $out" }
    Move-Item $tmp $out -Force
    Say "OK   $out (merged from $($parts.Count) parts, sha256 matches)" "Green"
    return "ok"
}

try {
    if (-not (Test-Path $Py)) { Fail "python not found: $Py (use -Py <path to python.exe>)" }
    New-Item -ItemType Directory -Force -Path $pool, $tsd | Out-Null
    $man = Get-Content (SrcPath "centrifugal/parts_sha256.json") -Raw | ConvertFrom-Json

    # 1) json manifests
    foreach ($j in @("pool_motor2_100_ch1245.json", "pool_motor2_100_ch1245_vmd.json")) {
        Copy-Item (SrcPath "centrifugal/pool/$j") (Join-Path $pool $j) -Force
    }

    # 2) pool (required)
    $poolOut = Join-Path $pool "pool_motor2_100_ch1245.npz"
    $r = Assemble $man.pool $poolOut
    if ($r -eq "mismatch") { Fail "$poolOut exists with a wrong sha256 (expected $($man.pool.sha256)) and the pool parts centrifugal\pool\pool_motor2_100_ch1245.npz.part* are not here" }
    if ($r -ne "ok") { Fail "pool parts centrifugal\pool\pool_motor2_100_ch1245.npz.part* not found" }

    # 3) VMD bank (optional here: the user may already have it)
    $vmdOut = Join-Path $pool "pool_motor2_100_ch1245_vmd.npy"
    $r = Assemble $man.vmd $vmdOut
    if ($r -eq "missing") {
        Say "WARN VMD bank not present and VMD parts (centrifugal\pool_vmd, branch data-da-v1-vmd) not downloaded: skipped. Not needed for the splits." "Yellow"
    }
    if ($r -eq "mismatch") {
        $msg = "VMD bank $vmdOut has a WRONG sha256 (expected $($man.vmd.sha256)) and the VMD parts (centrifugal\pool_vmd) were not downloaded; the file was left unchanged. Download branch data-da-v1-vmd, copy its deliver_data_v1\centrifugal\pool_vmd folder into this folder's centrifugal folder and run again, or delete the file."
        Say "PROBLEM $msg" "Red"
        $problems.Add($msg)
    }

    # 4) log spectrum
    $lsJson = Get-Content (SrcPath "centrifugal/pool/pool_motor2_100_ch1245_logspec.json") -Raw | ConvertFrom-Json
    $lsSrc = SrcPath "centrifugal/pool/pool_motor2_100_ch1245_logspec.npy"
    $lsDst = Join-Path $pool "pool_motor2_100_ch1245_logspec.npy"
    if ((Test-Path $lsDst) -and ((Sha $lsDst) -eq $lsJson.sha256)) {
        Say "OK   $lsDst (already present, sha256 matches)" "Green"
    } else {
        if (-not (Test-Path $lsSrc)) { Fail "log spectrum file not found: $lsSrc" }
        Copy-Item $lsSrc $lsDst -Force
        if ((Sha $lsDst) -ne $lsJson.sha256) { Fail "sha256 mismatch of $lsDst" }
        Say "OK   $lsDst (copied, sha256 matches)" "Green"
    }
    Copy-Item (SrcPath "centrifugal/pool/pool_motor2_100_ch1245_logspec.json") (Join-Path $pool "pool_motor2_100_ch1245_logspec.json") -Force

    # 5) target split files, tools and README
    foreach ($t in @("target_split_v1.json", "target_split_v1.csv", "check_report_v1.md")) {
        Copy-Item (SrcPath "target_split/$t") (Join-Path $tsd $t) -Force
    }
    Copy-Item (SrcPath "README_data_v1.md") (Join-Path $tsd "README_data_v1.md") -Force
    Say "OK   target split files copied to $tsd" "Green"

    # 6) build the 9 splits
    & $Py (SrcPath "tools/make_splits_da_v1.py") --pool (Join-Path $pool "pool_motor2_100_ch1245.npz") `
        --out $spl --target-split (Join-Path $tsd "target_split_v1.json") --logspec $lsDst
    if ($LASTEXITCODE -ne 0) { Fail "make_splits_da_v1.py returned $LASTEXITCODE" }
    $n = @(Get-ChildItem (Join-Path $spl "motor2_100_ch1245_C?_fold?.npz")).Count
    if ($n -ne 9) { Fail "expected 9 split files in $spl, found $n" }
    Say "OK   9 split files written to $spl" "Green"
} catch {
    Fail $_.Exception.Message
}
if ($problems.Count -gt 0) {
    foreach ($p in $problems) { Say "PROBLEM $p" "Red" }
    Fail "the splits were written, but $($problems.Count) problem(s) remain (see PROBLEM lines above)"
}
Say "all OK. Note: C1_fold2, C2_fold2, C3_fold2 have split_feasible = False (see check_report_v1.md)." "Green"
$log | Out-File -FilePath (Join-Path $Dest "merge_data_v1_log.txt") -Encoding utf8
exit 0
