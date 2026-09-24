param(
    [int]$Port = 5000
)

$projectDirectory = Split-Path -Parent $PSScriptRoot
$pythonExecutable = 'C:\Users\Ruslan\anaconda3\envs\yolo-gpu\python.exe'
$mlflowDirectory = Join-Path $projectDirectory 'My Artifacts\mlflow'
$databasePath = Join-Path $mlflowDirectory 'tracking.db'
$artifactDirectory = Join-Path $mlflowDirectory 'artifacts'
$stdoutPath = Join-Path $mlflowDirectory 'server.stdout.log'
$stderrPath = Join-Path $mlflowDirectory 'server.stderr.log'

New-Item -ItemType Directory -Force -Path $mlflowDirectory, $artifactDirectory | Out-Null
$activeListener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($activeListener) {
    Write-Output "MLflow is already listening on http://127.0.0.1:$Port"
    exit 0
}

$trackingUri = 'sqlite:///' + ($databasePath -replace '\\', '/')
$artifactUri = [System.Uri]::new($artifactDirectory).AbsoluteUri
$arguments = @(
    '-m', 'mlflow', 'server',
    '--host', '127.0.0.1',
    '--port', "$Port",
    '--workers', '1',
    '--backend-store-uri', ('"{0}"' -f $trackingUri),
    '--default-artifact-root', ('"{0}"' -f $artifactUri)
) -join ' '
Start-Process -FilePath $pythonExecutable -ArgumentList $arguments -WorkingDirectory $projectDirectory -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
Write-Output "MLflow is starting at http://127.0.0.1:$Port"
