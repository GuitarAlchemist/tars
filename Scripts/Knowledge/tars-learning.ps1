param (
    [Parameter(Mandatory=$false)]
    [string]$DataFile = "tars-learning-data.json",
    
    [Parameter(Mandatory=$false)]
    [switch]$Reset = $false
)

# Function to display colored text
function Write-ColoredText {
    param (
        [string]$Text,
        [string]$Color = "White"
    )
    
    Write-Host $Text -ForegroundColor $Color
}

# Initialize or load learning data
function Initialize-LearningData {
    if ($Reset -or -not (Test-Path $DataFile)) {
        $learningData = @{
            Transformations = @{}
            SuccessRates = @{}
            Feedback = @()
            LastUpdate = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        }
        
        $learningData | ConvertTo-Json -Depth 10 | Set-Content -Path $DataFile
        Write-ColoredText "Learning data initialized" "Green"
        return $learningData
    } else {
        $learningData = Get-Content -Path $DataFile -Raw | ConvertFrom-Json
        Write-ColoredText "Learning data loaded from $DataFile" "Green"
        return $learningData
    }
}

# Record a transformation
function Record-Transformation {
    param (
        [string]$TransformationType,
        [string]$OriginalCode,
        [string]$TransformedCode,
        [bool]$WasAccepted = $true
    )
    
    $learningData = Initialize-LearningData
    
    # Update transformation count
    if (-not $learningData.Transformations.$TransformationType) {
        $learningData.Transformations | Add-Member -NotePropertyName $TransformationType -NotePropertyValue @{
            Total = 0
            Accepted = 0
        }
    }
    
    $learningData.Transformations.$TransformationType.Total++
    if ($WasAccepted) {
        $learningData.Transformations.$TransformationType.Accepted++
    }
    
    # Calculate success rate
    $total = $learningData.Transformations.$TransformationType.Total
    $accepted = $learningData.Transformations.$TransformationType.Accepted
    $successRate = [math]::Round(($accepted / $total) * 100, 2)
    
    $learningData.SuccessRates | Add-Member -NotePropertyName $TransformationType -NotePropertyValue $successRate -Force
    
    # Add to feedback
    $feedback = @{
        Type = $TransformationType
        OriginalCode = $OriginalCode
        TransformedCode = $TransformedCode
        WasAccepted = $WasAccepted
        Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    }
    
    $learningData.Feedback += $feedback
    
    # Update last update timestamp
    $learningData.LastUpdate = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    
    # Save learning data
    $learningData | ConvertTo-Json -Depth 10 | Set-Content -Path $DataFile
    
    Write-ColoredText "Transformation recorded: $TransformationType (Success Rate: $successRate%)" "Green"
}

# Get success rate for a transformation type
function Get-TransformationSuccessRate {
    param (
        [string]$TransformationType
    )
    
    $learningData = Initialize-LearningData
    
    if ($learningData.SuccessRates.$TransformationType) {
        return $learningData.SuccessRates.$TransformationType
    } else {
        return 0
    }
}

# Get all transformation success rates
function Get-AllTransformationSuccessRates {
    $learningData = Initialize-LearningData
    return $learningData.SuccessRates
}

# Get transformation recommendations based on success rates
function Get-TransformationRecommendations {
    $learningData = Initialize-LearningData
    
    $recommendations = @()
    
    foreach ($type in $learningData.SuccessRates.PSObject.Properties.Name) {
        $successRate = $learningData.SuccessRates.$type
        
        if ($successRate -ge 80) {
            $confidence = "High"
        } elseif ($successRate -ge 50) {
            $confidence = "Medium"
        } else {
            $confidence = "Low"
        }
        
        $recommendations += @{
            Type = $type
            SuccessRate = $successRate
            Confidence = $confidence
        }
    }
    
    return $recommendations | Sort-Object -Property SuccessRate -Descending
}

# Export functions
Export-ModuleMember -Function Initialize-LearningData, Record-Transformation, Get-TransformationSuccessRate, Get-AllTransformationSuccessRates, Get-TransformationRecommendations
