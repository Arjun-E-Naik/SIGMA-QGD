# Single command to execute all experiment configurations and regenerate results
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$configDir = Join-Path $projectRoot "configs"
$driverScript = Join-Path $projectRoot "scripts\run_experiment.py"

Get-ChildItem -Path $configDir -Filter "*.yaml" -Recurse | ForEach-Object {
    $configPath = $_.FullName
    Write-Host "[Experiment] Executing config: $configPath"
    python $driverScript --config $configPath
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Experiment failed for $configPath (exit code $LASTEXITCODE)"
        exit $LASTEXITCODE
    }
}

Write-Host "[Experiment] All configurations completed successfully."
