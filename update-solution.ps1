# Script to update project references in the solution file
$solutionPath = "C:\Users\spare\source\repos\tars\tars.sln"
$backupDir = "C:\Users\spare\source\repos\tars\Backups\$(Get-Date -Format 'yyyy-MM-dd_HH-mm-ss')"

# Create backup of solution file
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
Copy-Item -Path $solutionPath -Destination "$backupDir\tars.sln" -Force

# Read the solution file
$solutionContent = Get-Content -Path $solutionPath -Raw

# Define the projects we want to ensure are in the solution
$projectsToInclude = @(
    @{Name = "TarsEngine"; Path = "TarsEngine\TarsEngine.csproj"; Type = "FAE04EC0-301F-11D3-BF4B-00C04F79EFBC"},
    @{Name = "TarsEngineFSharp"; Path = "TarsEngineFSharp\TarsEngineFSharp.fsproj"; Type = "F2A71F9B-5D33-465A-A702-920D77279786"},
    @{Name = "TarsEngine.Interfaces"; Path = "TarsEngine.Interfaces\TarsEngine.Interfaces.csproj"; Type = "FAE04EC0-301F-11D3-BF4B-00C04F79EFBC"},
    @{Name = "TarsEngine.SelfImprovement"; Path = "TarsEngine.SelfImprovement\TarsEngine.SelfImprovement.fsproj"; Type = "F2A71F9B-5D33-465A-A702-920D77279786"},
    @{Name = "TarsEngine.DSL"; Path = "TarsEngine.DSL\TarsEngine.DSL.fsproj"; Type = "F2A71F9B-5D33-465A-A702-920D77279786"},
    @{Name = "TarsEngine.DSL.Tests"; Path = "TarsEngine.DSL.Tests\TarsEngine.DSL.Tests.fsproj"; Type = "F2A71F9B-5D33-465A-A702-920D77279786"},
    @{Name = "TarsEngine.Tests"; Path = "TarsEngine.Tests\TarsEngine.Tests.csproj"; Type = "FAE04EC0-301F-11D3-BF4B-00C04F79EFBC"},
    @{Name = "TarsEngineFSharp.Core"; Path = "TarsEngineFSharp.Core\TarsEngineFSharp.Core.fsproj"; Type = "F2A71F9B-5D33-465A-A702-920D77279786"},
    @{Name = "TarsEngine.Unified"; Path = "TarsEngine.Unified\TarsEngine.Unified.csproj"; Type = "FAE04EC0-301F-11D3-BF4B-00C04F79EFBC"},
    @{Name = "TarsCli"; Path = "TarsCli\TarsCli.csproj"; Type = "FAE04EC0-301F-11D3-BF4B-00C04F79EFBC"},
    @{Name = "TarsApp"; Path = "TarsApp\TarsApp.csproj"; Type = "FAE04EC0-301F-11D3-BF4B-00C04F79EFBC"},
    @{Name = "ChatbotExample1"; Path = "Experiments\ChatbotExample1\ChatbotExample1.csproj"; Type = "FAE04EC0-301F-11D3-BF4B-00C04F79EFBC"}
)

# Check if each project is already in the solution
foreach ($project in $projectsToInclude) {
    $projectPattern = "Project\(\`"\{$($project.Type)\}\`"\) = \`"$($project.Name)\`", \`"$($project.Path)\`""
    if (-not ($solutionContent -match $projectPattern)) {
        Write-Host "Project $($project.Name) not found in solution. Adding it..."
        
        # Generate a new GUID for the project
        $projectGuid = [Guid]::NewGuid().ToString().ToUpper()
        
        # Create project entry
        $projectEntry = @"
Project("{$($project.Type)}") = "$($project.Name)", "$($project.Path)", "{$projectGuid}"
EndProject

"@
        
        # Add project entry before the GlobalSection
        $solutionContent = $solutionContent -replace "Global", "$projectEntry`Global"
        
        # Add project configuration
        $configEntry = @"
		{$projectGuid}.Debug|Any CPU.ActiveCfg = Debug|Any CPU
		{$projectGuid}.Debug|Any CPU.Build.0 = Debug|Any CPU
		{$projectGuid}.Release|Any CPU.ActiveCfg = Release|Any CPU
		{$projectGuid}.Release|Any CPU.Build.0 = Release|Any CPU

"@
        
        # Add configuration entry to GlobalSection(ProjectConfigurationPlatforms)
        $solutionContent = $solutionContent -replace "GlobalSection\(ProjectConfigurationPlatforms\) = postSolution", "GlobalSection(ProjectConfigurationPlatforms) = postSolution`n$configEntry"
    }
}

# Save the updated solution file
Set-Content -Path $solutionPath -Value $solutionContent

Write-Host "Solution file updated successfully."
