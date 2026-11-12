# Re-assemble the v19 pool + VMD bank (split into <50 MB parts for GitHub), verify sha256, build the 9 splits.
# Run from the repo's v19 folder:  powershell -ExecutionPolicy Bypass -File merge_v19.ps1 -Dest D:\DCWRU_Data\Paderborn_Experiment
param([string]$Dest = "D:\DCWRU_Data\Paderborn_Experiment")
$PY  = "D:\chuqishiyan1\zhubao\pump_reproduction\.venv_gpu\Scripts\python.exe"
$src = Join-Path $PSScriptRoot "centrifugal\pool"
$dst = Join-Path $Dest "centrifugal\pool"
New-Item -ItemType Directory -Force -Path $dst | Out-Null
$files = @(@{ Name = "pool_motor2_100_ch1245.npz";     Key = "sha256" },
           @{ Name = "pool_motor2_100_ch1245_vmd.npy"; Key = "bank_sha256" })
foreach ($f in $files) {
    $json = ($f.Name -replace '\.(npz|npy)$', '.json')
    Copy-Item (Join-Path $src $json) (Join-Path $dst $json) -Force
    $want = (Get-Content (Join-Path $src $json) -Raw | ConvertFrom-Json).($f.Key)
    $out  = Join-Path $dst $f.Name
    if (Test-Path $out) {
        if ((Get-FileHash $out -Algorithm SHA256).Hash.ToLower() -eq $want) { Write-Host "OK  $out (already assembled)"; continue }
    }
    $parts = @(Get-ChildItem (Join-Path $src "$($f.Name).part*") -ErrorAction SilentlyContinue | Sort-Object Name)
    if ($parts.Count -eq 0) {
        if ($f.Name -like "*_vmd.npy") { Write-Host "VMD bank parts not downloaded yet: skipped (baseline B does not need it)" -ForegroundColor Yellow; continue }
        Write-Host "no parts for $($f.Name)" -ForegroundColor Red; exit 1
    }
    $o = [System.IO.File]::Create($out)
    foreach ($p in $parts) { $i = [System.IO.File]::OpenRead($p.FullName); $i.CopyTo($o); $i.Close() }
    $o.Close()
    $got = (Get-FileHash $out -Algorithm SHA256).Hash.ToLower()
    if ($got -ne $want) { Write-Host "SHA MISMATCH $out" -ForegroundColor Red; exit 1 }
    Write-Host "OK  $out"
}
Copy-Item (Join-Path $PSScriptRoot "*.py")  $Dest -Force
Copy-Item (Join-Path $PSScriptRoot "*.ps1") $Dest -Force
Copy-Item (Join-Path $PSScriptRoot "*.md")  $Dest -Force
Copy-Item (Join-Path $PSScriptRoot "*.txt") $Dest -Force
Push-Location (Join-Path $Dest "centrifugal")
Copy-Item (Join-Path $PSScriptRoot "make_splits_from_pool.py") . -Force
& $PY make_splits_from_pool.py --pool pool\pool_motor2_100_ch1245.npz --out splits_ch1245
$rc = $LASTEXITCODE
Pop-Location
if ($rc -ne 0) { Write-Host "building splits FAILED" -ForegroundColor Red; exit 1 }
Write-Host "all OK: now run  powershell -ExecutionPolicy Bypass -File run_batch_v19_cf.ps1  in $Dest" -ForegroundColor Green
