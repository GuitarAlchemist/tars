# Script to update the TarsEngineFSharp.fsproj file

$content = Get-Content -Path "TarsEngineFSharp\TarsEngineFSharp.fsproj" -Raw

# Find the position to insert our files
$insertPosition = $content.IndexOf("<ItemGroup>") + 10

# Create the text to insert
$insertText = @"

        <Compile Include="TreeOfThought\ThoughtNode.fs" />
        <Compile Include="TreeOfThought\ThoughtTree.fs" />
        <Compile Include="TreeOfThought\Evaluation.fs" />
"@

# Insert the text
$updatedContent = $content.Insert($insertPosition, $insertText)

# Write the updated content back to the file
Set-Content -Path "TarsEngineFSharp\TarsEngineFSharp.fsproj" -Value $updatedContent
