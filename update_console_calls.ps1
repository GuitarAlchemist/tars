$filePath = "TarsCli/Commands/TestingFrameworkCommand.cs"
$content = Get-Content $filePath -Raw

# Replace WriteHeader calls
$content = $content -replace "WriteHeader\(", "_consoleService.WriteHeader("

# Replace WriteColorLine calls
$content = $content -replace "WriteColorLine\(", "_consoleService.WriteColorLine("

# Write the updated content back to the file
Set-Content $filePath $content
