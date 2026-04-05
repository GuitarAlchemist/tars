# Script to update the EnhancedTreeOfThoughtService.cs file

$content = Get-Content -Path "TarsEngine\Services\TreeOfThought\EnhancedTreeOfThoughtService.cs" -Raw

# Replace ExecuteAsync with ExecuteMetascriptAsync
$updatedContent = $content -replace "await _metascriptExecutor\.ExecuteAsync\(metascriptPath, variables\)", "await _metascriptExecutor.ExecuteMetascriptAsync(metascriptPath, variables)"

# Write the updated content back to the file
Set-Content -Path "TarsEngine\Services\TreeOfThought\EnhancedTreeOfThoughtService.cs" -Value $updatedContent
