# Script to update the TarsEngine project file

Write-Host "Updating TarsEngine project file..."

# Read the project file
$projectFile = Get-Content -Path "TarsEngine\TarsEngine.csproj" -Raw

# Replace the F# files section
$updatedProjectFile = $projectFile -replace '    <ItemGroup>\s*<Compile Include="FSharp\\BasicTreeOfThought.fs" />\s*<Compile Include="FSharp\\MetascriptExecution.fs" />\s*<Compile Include="FSharp\\MetascriptGeneration.fs" />\s*<Compile Include="FSharp\\MetascriptResultAnalysis.fs" />\s*<Compile Include="FSharp\\MetascriptToT.fs" />\s*<Compile Include="FSharp\\MetascriptTreeOfThought.fs" />\s*<Compile Include="FSharp\\MetascriptValidation.fs" />\s*<Compile Include="FSharp\\SimpleTreeOfThought.fs" />\s*</ItemGroup>', '    <!-- F# files have been moved to TarsEngine.TreeOfThought project -->'

# Write the updated project file
Set-Content -Path "TarsEngine\TarsEngine.csproj" -Value $updatedProjectFile

Write-Host "TarsEngine project file updated successfully!"
