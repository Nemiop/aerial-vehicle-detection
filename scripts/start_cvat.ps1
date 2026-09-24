$ErrorActionPreference = "Stop"

$CvatDistribution = "CVAT-Docker"
$KeepAliveScript = "/mnt/d/Ruslan/Test_Project/scripts/cvat_keepalive.sh"
$KeepAliveProcessIdPath = Join-Path $PSScriptRoot "..\tmp\cvat_wsl.pid"
$CvatUrl = "http://localhost:8080"
$StartupTimeoutSeconds = 120
$HealthCheckIntervalSeconds = 2

$keepAliveIsRunning = $false
if (Test-Path -LiteralPath $KeepAliveProcessIdPath) {
    $existingProcessId = Get-Content -LiteralPath $KeepAliveProcessIdPath -Raw
    $existingProcess = Get-Process -Id $existingProcessId -ErrorAction SilentlyContinue
    $keepAliveIsRunning = $existingProcess -and $existingProcess.ProcessName -eq "wsl"
}

if (-not $keepAliveIsRunning) {
    $wslArguments = @(
        "-d", $CvatDistribution,
        "--user", "root",
        "--", "bash", $KeepAliveScript
    )
    $keepAliveProcess = Start-Process `
        -FilePath "wsl.exe" `
        -ArgumentList $wslArguments `
        -WindowStyle Hidden `
        -PassThru
    Set-Content -LiteralPath $KeepAliveProcessIdPath -Value $keepAliveProcess.Id
}

$deadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
do {
    try {
        $response = Invoke-WebRequest -Uri $CvatUrl -UseBasicParsing -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            Write-Host "CVAT is ready: $CvatUrl"
            exit 0
        }
    }
    catch {
        Start-Sleep -Seconds $HealthCheckIntervalSeconds
    }
} while ((Get-Date) -lt $deadline)

throw "CVAT did not become ready within $StartupTimeoutSeconds seconds."
