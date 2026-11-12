# Re-assemble the VMD banks (precomputed in the sandbox, split into <50 MB parts for GitHub) and verify sha256.
# Run from the repo's v18 folder:  powershell -ExecutionPolicy Bypass -File merge_vmd_banks.ps1
# Then copy the whole v18\centrifugal folder to D:\DCWRU_Data\Paderborn_Experiment\centrifugal  (see README_v18.md)
$dir = Join-Path $PSScriptRoot "centrifugal\splits"
$bad = 0
foreach ($j in Get-ChildItem "$dir\*_vmd.json") {
    $npy = $j.FullName -replace '\.json$', '.npy'
    $parts = Get-ChildItem "$npy.part*" | Sort-Object Name
    if ($parts.Count -eq 0) { Write-Host "no parts for $npy" -ForegroundColor Red; $bad++; continue }
    $out = [System.IO.File]::Create($npy)
    foreach ($p in $parts) { $in = [System.IO.File]::OpenRead($p.FullName); $in.CopyTo($out); $in.Close() }
    $out.Close()
    $want = (Get-Content $j.FullName -Raw | ConvertFrom-Json).bank_sha256
    $got  = (Get-FileHash $npy -Algorithm SHA256).Hash.ToLower()
    if ($got -eq $want) { Write-Host "OK  $($j.BaseName).npy"; Remove-Item ($parts | ForEach-Object { $_.FullName }) }
    else { Write-Host "SHA MISMATCH $npy" -ForegroundColor Red; $bad++ }
}
if ($bad -eq 0) { Write-Host "all VMD banks OK" -ForegroundColor Green } else { Write-Host "$bad problem(s)" -ForegroundColor Red; exit 1 }
