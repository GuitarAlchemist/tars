# Script to update the CliSupport.cs file to add the Tree-of-Thought command

# Read the file
$content = Get-Content -Path "TarsCli\CliSupport.cs" -Raw

# Find the position to insert our command
$insertPosition = $content.IndexOf("        // Add the Improvement Metascript command")
$insertPosition = $content.IndexOf(";", $insertPosition) + 1

# Create the text to insert
$insertText = @"

        // Add the Tree-of-Thought Auto-Improvement command
        var treeOfThoughtAutoImprovementCommand = serviceProvider.GetRequiredService<TreeOfThoughtAutoImprovementCommand>();
        rootCommand.AddCommand(treeOfThoughtAutoImprovementCommand);
"@

# Insert the text
$updatedContent = $content.Insert($insertPosition, $insertText)

# Write the updated content back to the file
Set-Content -Path "TarsCli\CliSupport.cs" -Value $updatedContent
