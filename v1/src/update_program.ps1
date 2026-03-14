# Script to update the Program.cs file to register the Tree-of-Thought services

# Read the file
$content = Get-Content -Path "TarsCli\Program.cs" -Raw

# Find the position to insert our services
$insertPosition = $content.IndexOf(".AddTarsServices()")
$insertPosition = $content.IndexOf(";", $insertPosition) + 1

# Create the text to insert
$insertText = @"

                // Add Tree-of-Thought services
                .AddTreeOfThoughtServices()
"@

# Insert the text
$updatedContent = $content.Insert($insertPosition, $insertText)

# Find the position to insert our using statement
$usingPosition = $content.IndexOf("using TarsEngine.Extensions;")
$usingPosition = $content.IndexOf(";", $usingPosition) + 1

# Create the using statement to insert
$usingText = @"

using TarsCli.Extensions;
"@

# Insert the using statement
$updatedContent = $updatedContent.Insert($usingPosition, $usingText)

# Write the updated content back to the file
Set-Content -Path "TarsCli\Program.cs" -Value $updatedContent
