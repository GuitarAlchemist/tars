# Remove copied files from subdirectories
$directories = @("Core", "Analysis", "Services", "DSL", "Agents")
foreach ($dir in $directories) {
    $path = "TarsEngineFSharp\$dir"
    if (Test-Path $path) {
        # Don't remove the original Agents directory files
        if ($dir -eq "Agents") {
            $originalFiles = @("AgentTypes.fs", "AnalysisAgent.fs", "ValidationAgent.fs", "TransformationAgent.fs")
            $filesToRemove = Get-ChildItem -Path $path -Filter "*.fs" | Where-Object { $originalFiles -notcontains $_.Name }
            foreach ($file in $filesToRemove) {
                Remove-Item -Path $file.FullName -Force
                Write-Host "Removed $($file.FullName)"
            }
        } else {
            # Remove all .fs files in other directories
            Get-ChildItem -Path $path -Filter "*.fs" | ForEach-Object {
                Remove-Item -Path $_.FullName -Force
                Write-Host "Removed $($_.FullName)"
            }
        }
    }
}

Write-Host "Cleanup completed."
