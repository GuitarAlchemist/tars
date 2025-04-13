$filePath = "TarsCli/Commands/TestingFrameworkCommand.cs"
$content = Get-Content $filePath -Raw

# Replace _consoleService._consoleService with _consoleService
$content = $content -replace "_consoleService\._consoleService\.", "_consoleService."

# Write the updated content back to the file
Set-Content $filePath $content
