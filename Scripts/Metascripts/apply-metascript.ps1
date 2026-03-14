param (
    [Parameter(Mandatory=$true)]
    [string]$MetascriptFile,
    
    [Parameter(Mandatory=$true)]
    [string]$TargetFile,
    
    [Parameter(Mandatory=$false)]
    [switch]$Preview = $false
)

# Function to display colored text
function Write-ColoredText {
    param (
        [string]$Text,
        [string]$Color = "White"
    )
    
    Write-Host $Text -ForegroundColor $Color
}

# Check if files exist
if (-not (Test-Path $MetascriptFile)) {
    Write-ColoredText "Error: Metascript file not found: $MetascriptFile" "Red"
    exit 1
}

if (-not (Test-Path $TargetFile)) {
    Write-ColoredText "Error: Target file not found: $TargetFile" "Red"
    exit 1
}

# Load metascript content
$metascriptContent = Get-Content -Path $MetascriptFile -Raw
Write-ColoredText "Loaded metascript from file: $MetascriptFile" "Green"

# Load target file content
$targetContent = Get-Content -Path $TargetFile -Raw
Write-ColoredText "Loaded target file: $TargetFile" "Green"

# Parse metascript
function Parse-Metascript {
    param (
        [string]$Content
    )
    
    $rules = @()
    
    # Regular expression to match rule definitions
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
    
    return $rules
}

# Apply metascript rules to code
function Apply-MetascriptRules {
    param (
        [array]$Rules,
        [string]$Code,
        [string]$Language = "any"
    )
    
    $transformedCode = $Code
    $appliedRules = @()
    
    foreach ($rule in $Rules) {
        # Skip rules for other languages
        if ($rule.Language -ne "any" -and $rule.Language -ne $Language) {
            continue
        }
        
        # Convert the pattern to a regex pattern
        $regexPattern = $rule.Pattern
        
        # Escape special regex characters
        $regexPattern = $regexPattern -replace '\.', '\.'
        $regexPattern = $regexPattern -replace '\(', '\('
        $regexPattern = $regexPattern -replace '\)', '\)'
        $regexPattern = $regexPattern -replace '\[', '\['
        $regexPattern = $regexPattern -replace '\]', '\]'
        $regexPattern = $regexPattern -replace '\+', '\+'
        $regexPattern = $regexPattern -replace '\*', '\*'
        $regexPattern = $regexPattern -replace '\?', '\?'
        
        # Replace variables with capture groups
        $regexPattern = $regexPattern -replace '\$(\w+)', '(?<$1>.*?)'
        
        # Create the regex
        $regex = [regex]::new($regexPattern, [System.Text.RegularExpressions.RegexOptions]::Singleline)
        
        # Find matches
        $matches = $regex.Matches($transformedCode)
        
        if ($matches.Count -gt 0) {
            Write-ColoredText "Found $($matches.Count) matches for rule $($rule.Name)" "Green"
            
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
        }
    }
    
    return @{
        Code = $transformedCode
        AppliedRules = $appliedRules
    }
}

# Parse metascript
$rules = Parse-Metascript -Content $metascriptContent

Write-ColoredText "Parsed $($rules.Count) metascript rules:" "Cyan"
foreach ($rule in $rules) {
    Write-ColoredText "  - $($rule.Name): $($rule.Description)" "White"
}

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

# Apply rules
$result = Apply-MetascriptRules -Rules $rules -Code $targetContent -Language $language

if ($result.AppliedRules.Count -gt 0) {
    Write-ColoredText "Applied $($result.AppliedRules.Count) rules:" "Green"
    foreach ($rule in $result.AppliedRules) {
        Write-ColoredText "  - $($rule.Name): $($rule.Description)" "Green"
    }
    
    if ($Preview) {
        Write-ColoredText "`nPreview of changes:" "Cyan"
        
        # Show a simple diff
        $originalLines = $targetContent -split "`n"
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
