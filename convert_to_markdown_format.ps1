# PowerShell script to convert TARS metascripts from DSL format to Markdown format
# Converts .trsx files with DESCRIBE{} blocks to .meta files with YAML frontmatter

param(
    [string]$SourcePath = ".tars",
    [switch]$DryRun = $false
)

Write-Host "🔄 TARS Metascript DSL to Markdown Converter" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

function Convert-DslToMarkdown {
    param(
        [string]$Content,
        [string]$FileName
    )
    
    $markdown = @()
    $markdown += "# $FileName"
    $markdown += ""
    
    # Extract DESCRIBE block
    if ($Content -match "DESCRIBE\s*\{([^}]*)\}") {
        $describeContent = $matches[1]
        
        # Parse DESCRIBE properties
        $name = if ($describeContent -match 'name:\s*"([^"]*)"') { $matches[1] } else { $FileName }
        $version = if ($describeContent -match 'version:\s*"([^"]*)"') { $matches[1] } else { "1.0" }
        $description = if ($describeContent -match 'description:\s*"([^"]*)"') { $matches[1] } else { "TARS Metascript" }
        $author = if ($describeContent -match 'author:\s*"([^"]*)"') { $matches[1] } else { "TARS" }
        
        $markdown += "```yaml"
        $markdown += "name: $name"
        $markdown += "version: $version"
        $markdown += "description: $description"
        $markdown += "author: $author"
        $markdown += "```"
        $markdown += ""
    }
    
    # Extract CONFIG block
    if ($Content -match "CONFIG\s*\{([^}]*)\}") {
        $configContent = $matches[1]
        $markdown += "## Configuration"
        $markdown += ""
        $markdown += "```yaml"
        
        # Parse CONFIG properties
        $configLines = $configContent -split "`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ -and $_ -notmatch "^\s*$" }
        foreach ($line in $configLines) {
            if ($line -match '(\w+):\s*"?([^"]*)"?') {
                $key = $matches[1]
                $value = $matches[2]
                $markdown += "$key`: $value"
            }
        }
        
        $markdown += "```"
        $markdown += ""
    }
    
    # Extract VARIABLE blocks
    $variableMatches = [regex]::Matches($Content, "VARIABLE\s+(\w+)\s*\{([^}]*)\}", [System.Text.RegularExpressions.RegexOptions]::Singleline)
    if ($variableMatches.Count -gt 0) {
        $markdown += "## Variables"
        $markdown += ""
        
        foreach ($match in $variableMatches) {
            $varName = $match.Groups[1].Value
            $varContent = $match.Groups[2].Value
            
            if ($varContent -match 'value:\s*"([^"]*)"') {
                $varValue = $matches[1]
                $markdown += "- **$varName**: `$varValue`"
            }
        }
        $markdown += ""
    }
    
    # Extract FSHARP blocks
    $fsharpMatches = [regex]::Matches($Content, "FSHARP\s*\{([^}]*)\}", [System.Text.RegularExpressions.RegexOptions]::Singleline)
    if ($fsharpMatches.Count -gt 0) {
        $blockCount = 1
        foreach ($match in $fsharpMatches) {
            $fsharpCode = $match.Groups[1].Value.Trim()
            
            if ($fsharpMatches.Count -eq 1) {
                $markdown += "## F# Implementation"
            } else {
                $markdown += "## F# Block $blockCount"
            }
            $markdown += ""
            $markdown += "```fsharp"
            $markdown += $fsharpCode
            $markdown += "```"
            $markdown += ""
            $blockCount++
        }
    }
    
    # Extract ACTION blocks
    $actionMatches = [regex]::Matches($Content, "ACTION\s*\{([^}]*)\}", [System.Text.RegularExpressions.RegexOptions]::Singleline)
    if ($actionMatches.Count -gt 0) {
        $markdown += "## Actions"
        $markdown += ""
        
        foreach ($match in $actionMatches) {
            $actionContent = $match.Groups[1].Value
            
            if ($actionContent -match 'type:\s*"([^"]*)"') {
                $actionType = $matches[1]
                $markdown += "- **Action Type**: $actionType"
                
                if ($actionContent -match 'message:\s*"([^"]*)"') {
                    $message = $matches[1]
                    $markdown += "  - **Message**: $message"
                }
            }
        }
        $markdown += ""
    }
    
    # Add conclusion
    $markdown += "## Execution"
    $markdown += ""
    $markdown += "This metascript can be executed using the TARS CLI:"
    $markdown += ""
    $markdown += "```bash"
    $markdown += "dotnet run --project TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj -- metascript path/to/script.meta"
    $markdown += "```"
    
    return $markdown -join "`n"
}

# Find all .trsx files
$trsxFiles = Get-ChildItem -Path $SourcePath -Recurse -Filter "*.trsx" | Where-Object { $_.Name -notlike "*template*" }

Write-Host "📁 Found $($trsxFiles.Count) .trsx files to convert" -ForegroundColor Yellow

foreach ($file in $trsxFiles) {
    Write-Host "`n🔍 Processing: $($file.FullName)" -ForegroundColor Green
    
    try {
        # Read the file content
        $content = Get-Content -Path $file.FullName -Raw
        
        # Check if it uses DSL format (has DESCRIBE blocks)
        if ($content -match "DESCRIBE\s*\{") {
            Write-Host "  🔄 Converting DSL format to Markdown format" -ForegroundColor Blue
            
            # Convert to markdown
            $fileName = [System.IO.Path]::GetFileNameWithoutExtension($file.Name)
            $markdownContent = Convert-DslToMarkdown -Content $content -FileName $fileName
            
            if (-not $DryRun) {
                # Save as .meta file
                $newPath = $file.FullName -replace "\.trsx$", ".meta"
                Set-Content -Path $newPath -Value $markdownContent -Encoding UTF8
                
                # Remove old .trsx file
                Remove-Item -Path $file.FullName
                
                Write-Host "  ✅ Converted to: $newPath" -ForegroundColor Green
            } else {
                Write-Host "  📋 Would convert to: $($file.FullName -replace '\.trsx$', '.meta')" -ForegroundColor Yellow
            }
        } else {
            Write-Host "  ⚠️  No DESCRIBE block found, skipping" -ForegroundColor Yellow
        }
        
    } catch {
        Write-Host "  ❌ Error processing file: $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host "`n✅ Metascript conversion completed!" -ForegroundColor Green
Write-Host "📊 Processed $($trsxFiles.Count) files" -ForegroundColor Cyan

if ($DryRun) {
    Write-Host "`n🔍 This was a dry run. Use -DryRun:`$false to actually convert files." -ForegroundColor Yellow
}

Write-Host "`n📋 New Format Benefits:" -ForegroundColor Cyan
Write-Host "• YAML frontmatter for metadata" -ForegroundColor White
Write-Host "• Markdown formatting for documentation" -ForegroundColor White
Write-Host "• Code blocks with syntax highlighting" -ForegroundColor White
Write-Host "• Better readability and maintainability" -ForegroundColor White
Write-Host "• Compatible with modern documentation tools" -ForegroundColor White
