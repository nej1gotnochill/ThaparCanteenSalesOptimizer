$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$frontendDir = Join-Path (Split-Path -Parent $root) 'canteen-vision'
$pythonExe = Join-Path $root '.venv\Scripts\python.exe'

if (-not (Test-Path $pythonExe)) {
    throw "Python not found at $pythonExe. Create the virtual environment first."
}

if (-not (Test-Path $frontendDir)) {
    throw "Frontend folder not found at $frontendDir. Clone canteen-vision beside this repo."
}

if (-not (Test-Path (Join-Path $frontendDir 'node_modules'))) {
    Write-Host 'Installing frontend dependencies...'
    Push-Location $frontendDir
    npm install
    Pop-Location
}

Write-Host 'Starting backend API on http://127.0.0.1:8000 ...'
Start-Process -FilePath $pythonExe -ArgumentList 'api_server.py' -WorkingDirectory $root | Out-Null

Write-Host 'Starting frontend on http://localhost:8080 ...'
Start-Process -FilePath 'cmd.exe' -ArgumentList '/c', 'npm', 'run', 'dev', '--', '--host', '0.0.0.0', '--port', '8080' -WorkingDirectory $frontendDir | Out-Null

Write-Host ''
Write-Host 'Open the frontend at: http://localhost:8080'
Write-Host 'Backend health check: http://127.0.0.1:8000/api/health'
Write-Host 'Backend dashboard: http://127.0.0.1:8000/api/dashboard'