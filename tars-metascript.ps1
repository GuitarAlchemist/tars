param (
    [Parameter(Mandatory=$false)]
    [string]$MetascriptFile,
    
    [Parameter(Mandatory=$false)]
    [string]$MetascriptContent,
    
    [Parameter(Mandatory=$false)]
    [string]$TargetFile,
    
    [Parameter(Mandatory=$false)]
    [switch]$Preview = $false,
    
    [Parameter(Mandatory=$false)]
    [switch]$Verbose = $false
)

# Function to display colored text
function Write-ColoredText {
    param (
        [string]$Text,
        [string]$Color = "White"
    )
    
    Write-Host $Text -ForegroundColor $Color
}

# Function to log verbose information
function Write-VerboseLog {
    param (
        [string]$Text
    )
    
    if ($Verbose) {
        Write-ColoredText "VERBOSE: $Text" "DarkGray"
    }
}

# Check if we have either a metascript file or content
if (-not $MetascriptFile -and -not $MetascriptContent) {
    Write-ColoredText "Error: Either MetascriptFile or MetascriptContent must be provided" "Red"
    exit 1
}

# Load metascript content
if ($MetascriptFile) {
    if (-not (Test-Path $MetascriptFile)) {
        Write-ColoredText "Error: Metascript file not found: $MetascriptFile" "Red"
        exit 1
    }
    
    $MetascriptContent = Get-Content -Path $MetascriptFile -Raw
    Write-VerboseLog "Loaded metascript from file: $MetascriptFile"
}

# Check if target file exists
if ($TargetFile -and -not (Test-Path $TargetFile)) {
    Write-ColoredText "Error: Target file not found: $TargetFile" "Red"
    exit 1
}

# Parse metascript
function Parse-Metascript {
    param (
        [string]$Content
    )
    
    Write-VerboseLog "Parsing metascript..."
    
    $rules = @()
    
    # Regular expression to match rule definitions
    # Format: rule Name { match: "pattern" replace: "replacement" requires: "namespace1; namespace2" description: "description" }
    $rulePattern = '(?ms)rule\s+(\w+)\s*\{\s*match:\s*"(.*?)"\s*replace:\s*"(.*?)"\s*(?:requires:\s*"(.*?)"\s*)?(?:description:\s*"(.*?)"\s*)?(?:language:\s*"(.*?)"\s*)?(?:confidence:\s*(\d+(?:\.\d+)?)\s*)?}'
    
    $matches = [regex]::Matches($Content, $rulePattern)
    
    foreach ($match in $matches) {
        $rule = @{
            Name = $match.Groups[1].Value
            Pattern = $match.Groups[2].Value
            Replacement = $match.Groups[3].Value
            Requires = @()
            Description = ""
            Language = "any"
            Confidence = 1.0
        }
        
        # Parse required namespaces
        if ($match.Groups[4].Success) {
            $rule.Requires = $match.Groups[4].Value -split ";\s*" | Where-Object { $_ }
        }
        
        # Parse description
        if ($match.Groups[5].Success) {
            $rule.Description = $match.Groups[5].Value
        } else {
            $rule.Description = "Apply $($rule.Name) transformation"
        }
        
        # Parse language
        if ($match.Groups[6].Success) {
            $rule.Language = $match.Groups[6].Value
        }
        
        # Parse confidence
        if ($match.Groups[7].Success) {
            $rule.Confidence = [double]$match.Groups[7].Value
        }
        
        $rules += $rule
    }
    
    Write-VerboseLog "Parsed $($rules.Count) rules"
    return $rules
}

# Apply metascript rules to code
function Apply-MetascriptRules {
    param (
        [array]$Rules,
        [string]$Code,
        [string]$Language = "any"
    )
    
    Write-VerboseLog "Applying metascript rules to code..."
    
    $transformedCode = $Code
    $appliedRules = @()
    
    foreach ($rule in $Rules) {
        # Skip rules for other languages
        if ($rule.Language -ne "any" -and $rule.Language -ne $Language) {
            Write-VerboseLog "Skipping rule $($rule.Name) (language: $($rule.Language))"
            continue
        }
        
        Write-VerboseLog "Applying rule: $($rule.Name)"
        
        # Convert the pattern to a regex pattern
        $regexPattern = $rule.Pattern
        
        # Replace variables with capture groups
        $regexPattern = $regexPattern -replace '\$(\w+)', '(?<$1>.*?)'
        
        # Create the regex
        $regex = [regex]::new($regexPattern, [System.Text.RegularExpressions.RegexOptions]::Singleline)
        
        # Find matches
        $matches = $regex.Matches($transformedCode)
        
        if ($matches.Count -gt 0) {
            Write-VerboseLog "Found $($matches.Count) matches for rule $($rule.Name)"
            
            # Apply the replacement
            $replacement = $rule.Replacement
            
            # Replace variables with their captured values
            foreach ($match in $matches) {
                $matchValue = $match.Value
                $replacementValue = $replacement
                
                # Replace variables with their captured values
                foreach ($groupName in $regex.GetGroupNames()) {
                    if ($groupName -ne "0" -and $match.Groups[$groupName].Success) {
                        $replacementValue = $replacementValue -replace "\`$$groupName", $match.Groups[$groupName].Value
                    }
                }
                
                # Apply the replacement
                $transformedCode = $transformedCode.Replace($matchValue, $replacementValue)
            }
            
            $appliedRules += $rule
        } else {
            Write-VerboseLog "No matches found for rule $($rule.Name)"
        }
    }
    
    return @{
        Code = $transformedCode
        AppliedRules = $appliedRules
    }
}

# Main execution
$rules = Parse-Metascript -Content $MetascriptContent

Write-ColoredText "Parsed $($rules.Count) metascript rules:" "Cyan"
foreach ($rule in $rules) {
    Write-ColoredText "  - $($rule.Name): $($rule.Description)" "White"
}

if ($TargetFile) {
    $code = Get-Content -Path $TargetFile -Raw
    
    # Determine language from file extension
    $extension = [System.IO.Path]::GetExtension($TargetFile).ToLower()
    $language = switch ($extension) {
        ".cs" { "csharp" }
        ".fs" { "fsharp" }
        ".js" { "javascript" }
        ".ts" { "typescript" }
        ".py" { "python" }
        default { "any" }
    }
    
    Write-ColoredText "Applying metascript rules to $TargetFile (language: $language)..." "Cyan"
    
    $result = Apply-MetascriptRules -Rules $rules -Code $code -Language $language
    
    if ($result.AppliedRules.Count -gt 0) {
        Write-ColoredText "Applied $($result.AppliedRules.Count) rules:" "Green"
        foreach ($rule in $result.AppliedRules) {
            Write-ColoredText "  - $($rule.Name): $($rule.Description)" "Green"
        }
        
        if ($Preview) {
            Write-ColoredText "`nPreview of changes:" "Cyan"
            
            # Show a simple diff
            $originalLines = $code -split "`n"
            $transformedLines = $result.Code -split "`n"
            
            for ($i = 0; $i -lt [Math]::Max($originalLines.Length, $transformedLines.Length); $i++) {
                $originalLine = if ($i -lt $originalLines.Length) { $originalLines[$i] } else { "" }
                $transformedLine = if ($i -lt $transformedLines.Length) { $transformedLines[$i] } else { "" }
                
                if ($originalLine -ne $transformedLine) {
                    Write-ColoredText "Line $($i + 1):" "Yellow"
                    Write-ColoredText "  - $originalLine" "Red"
                    Write-ColoredText "  + $transformedLine" "Green"
                }
            }
        } else {
            # Save the transformed code
            $result.Code | Set-Content -Path $TargetFile
            Write-ColoredText "Transformed code saved to $TargetFile" "Green"
        }
    } else {
        Write-ColoredText "No rules were applied to the code" "Yellow"
    }
} else {
    Write-ColoredText "No target file specified. Use -TargetFile to apply the metascript to a file." "Yellow"
}
