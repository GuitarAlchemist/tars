$folders = @(
    "TarsEvolutionTrigger",
    "Legacy_CSharp_Projects",
    "TarsAutoImprovementRunner",
    "TarsCSharpDemo",
    "TarsSupervisor",
    "TarsIntegratedDashboard",
    "Experiments",
    "BlueGreenDemo",
    "RealBlueGreen",
    "TarsEngine.FSharp.Adapters",
    "FinalRealBlueGreen",
    "TarsEngine.CSharp.Adapters",
    "TarsEngine.Services.AI",
    "BlueGreenAI",
    "RealTarsEvolution"
)

$dest = "v1/parked_csharp"
if (-not (Test-Path $dest)) {
    New-Item -ItemType Directory -Path $dest | Out-Null
}

foreach ($folder in $folders) {
    $path = "v1/src/$folder"
    if (Test-Path $path) {
        Write-Host "Moving $folder..."
        Move-Item -Path $path -Destination $dest -Force
    }
    else {
        Write-Host "Folder not found: $folder"
    }
}
