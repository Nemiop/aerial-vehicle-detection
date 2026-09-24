$ErrorActionPreference = "Stop"

$CvatDistribution = "CVAT-Docker"
$CvatDirectory = "/mnt/d/Ruslan/Test_Project/tools/cvat"
$KeepAliveProcessIdPath = Join-Path $PSScriptRoot "..\tmp\cvat_wsl.pid"

wsl.exe -d $CvatDistribution --user root -- bash -lc `
    "cd '$CvatDirectory' && docker compose stop"

if (Test-Path -LiteralPath $KeepAliveProcessIdPath) {
    $keepAliveProcessId = Get-Content -LiteralPath $KeepAliveProcessIdPath -Raw
    $keepAliveProcess = Get-Process -Id $keepAliveProcessId -ErrorAction SilentlyContinue
    if ($keepAliveProcess -and $keepAliveProcess.ProcessName -eq "wsl") {
        Stop-Process -Id $keepAliveProcessId
    }
    Remove-Item -LiteralPath $KeepAliveProcessIdPath -Force
}

Write-Host "CVAT containers are stopped. Persistent data is preserved."
