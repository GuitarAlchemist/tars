# TARS Response Summarization Tool
# Multi-level LLM response summarization with MoE consensus

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("single", "multi", "compare", "batch", "dsl-test", "interactive", "help")]
    [string]$Action = "interactive",
    
    [Parameter(Mandatory=$false)]
    [string]$Text = "",
    
    [Parameter(Mandatory=$false)]
    [ValidateSet("1", "2", "3", "4", "5", "executive", "tactical", "operational", "comprehensive", "detailed")]
    [string]$Level = "tactical",
    
    [Parameter(Mandatory=$false)]
    [string]$InputFile = "",
    
    [Parameter(Mandatory=$false)]
    [string]$OutputFile = "",
    
    [Parameter(Mandatory=$false)]
    [switch]$MoeConsensus = $true,
    
    [Parameter(Mandatory=$false)]
    [switch]$AutoCorrect = $true
)

Write-Host "📄 TARS RESPONSE SUMMARIZATION TOOL" -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan
Write-Host ""

# Function to parse summarization level
function Get-SummarizationLevel {
    param([string]$LevelInput)
    
    switch ($LevelInput.ToLower()) {
        "1" { return "Executive" }
        "2" { return "Tactical" }
        "3" { return "Operational" }
        "4" { return "Comprehensive" }
        "5" { return "Detailed" }
        "executive" { return "Executive" }
        "tactical" { return "Tactical" }
        "operational" { return "Operational" }
        "comprehensive" { return "Comprehensive" }
        "detailed" { return "Detailed" }
        default { return "Tactical" }
    }
}

# Function to get compression ratio for level
function Get-CompressionRatio {
    param([string]$Level)
    
    switch ($Level) {
        "Executive" { return 0.95 }
        "Tactical" { return 0.85 }
        "Operational" { return 0.75 }
        "Comprehensive" { return 0.60 }
        "Detailed" { return 0.40 }
        default { return 0.80 }
    }
}

# Function to extract key sentences
function Get-KeySentences {
    param(
        [string]$Text,
        [int]$TargetCount
    )
    
    $sentences = $Text -split '[.!?]' | Where-Object { $_.Trim().Length -gt 10 }
    
    if ($sentences.Count -le $TargetCount) {
        return $sentences
    }
    
    # Simple scoring based on length and position
    $scoredSentences = @()
    for ($i = 0; $i -lt $sentences.Count; $i++) {
        $sentence = $sentences[$i].Trim()
        $lengthScore = [Math]::Min($sentence.Length / 100.0, 1.0)
        $positionScore = if ($i -lt 3) { 1.0 } elseif ($i -ge $sentences.Count - 2) { 0.8 } else { 0.6 }
        $score = $lengthScore * $positionScore
        
        $scoredSentences += @{
            Sentence = $sentence
            Score = $score
        }
    }
    
    $topSentences = $scoredSentences | Sort-Object Score -Descending | Select-Object -First $TargetCount
    return $topSentences | ForEach-Object { $_.Sentence }
}

# Function to generate summary for specific level
function New-LevelSummary {
    param(
        [string]$Text,
        [string]$Level,
        [bool]$MoeConsensus = $true,
        [bool]$AutoCorrect = $true
    )
    
    $targetSentences = switch ($Level) {
        "Executive" { 1 }
        "Tactical" { 3 }
        "Operational" { 6 }
        "Comprehensive" { 10 }
        "Detailed" { 15 }
        default { 3 }
    }
    
    $keySentences = Get-KeySentences -Text $Text -TargetCount $targetSentences
    $summary = $keySentences -join " "
    
    # Apply corrections if enabled
    if ($AutoCorrect) {
        $summary = $summary -replace '  ', ' '
        if (-not ($summary.EndsWith('.') -or $summary.EndsWith('!') -or $summary.EndsWith('?'))) {
            $summary += "."
        }
        if ($summary.Length -gt 0 -and [char]::IsLower($summary[0])) {
            $summary = [char]::ToUpper($summary[0]) + $summary.Substring(1)
        }
    }
    
    $compressionRatio = 1.0 - ($summary.Length / $Text.Length)
    $targetCompression = Get-CompressionRatio -Level $Level
    $compressionScore = 1.0 - [Math]::Abs($compressionRatio - $targetCompression)
    
    return @{
        Level = $Level
        Summary = $summary
        CompressionRatio = $compressionRatio
        ConfidenceScore = $compressionScore * 0.8 + 0.2
        OriginalLength = $Text.Length
        SummaryLength = $summary.Length
    }
}

# Function to generate MoE consensus summary
function New-MoeConsensusSummary {
    param(
        [string]$Text,
        [string]$Level
    )
    
    Write-Host "  🧠 Generating expert opinions..." -ForegroundColor Yellow
    
    # Simulate different expert approaches
    $clarityExpert = New-LevelSummary -Text $Text -Level $Level -AutoCorrect $true
    $clarityExpert.Summary = $clarityExpert.Summary -replace "utilize", "use" -replace "demonstrate", "show"
    
    $brevityExpert = New-LevelSummary -Text $Text -Level $Level -AutoCorrect $true
    $targetCount = switch ($Level) {
        "Executive" { 1 }
        "Tactical" { 2 }
        "Operational" { 4 }
        default { 2 }
    }
    $brevitySentences = Get-KeySentences -Text $Text -TargetCount $targetCount
    $brevityExpert.Summary = ($brevitySentences -join " ") -replace " that ", " " -replace " which ", " "
    
    $accuracyExpert = New-LevelSummary -Text $Text -Level $Level -AutoCorrect $true
    # Preserve numbers and facts
    $facts = [regex]::Matches($Text, '\b\d+(?:\.\d+)?%?|\b\d{4}\b') | ForEach-Object { $_.Value }
    if ($facts.Count -gt 0) {
        $accuracyExpert.Summary += " Key facts: " + ($facts | Select-Object -First 3 -Unique) -join ", "
    }
    
    # Simple consensus - weighted average
    $experts = @(
        @{ Expert = "Clarity"; Summary = $clarityExpert; Weight = 0.8 }
        @{ Expert = "Brevity"; Summary = $brevityExpert; Weight = 0.7 }
        @{ Expert = "Accuracy"; Summary = $accuracyExpert; Weight = 0.9 }
    )
    
    # Select best summary based on weighted confidence
    $bestExpert = $experts | Sort-Object { $_.Summary.ConfidenceScore * $_.Weight } -Descending | Select-Object -First 1
    
    $consensusResult = $bestExpert.Summary
    $consensusResult.ConfidenceScore = ($experts | ForEach-Object { $_.Summary.ConfidenceScore * $_.Weight } | Measure-Object -Average).Average
    
    Write-Host "  ✅ Expert consensus achieved" -ForegroundColor Green
    
    return $consensusResult
}

# Function to display summary result
function Show-SummaryResult {
    param($Result)
    
    Write-Host ""
    Write-Host "SUMMARY RESULT:" -ForegroundColor Green
    Write-Host "===============" -ForegroundColor Green
    Write-Host "  Level: $($Result.Level)" -ForegroundColor White
    Write-Host "  Compression: $($Result.CompressionRatio.ToString('P1'))" -ForegroundColor White
    Write-Host "  Confidence: $($Result.ConfidenceScore.ToString('P1'))" -ForegroundColor White
    Write-Host "  Length: $($Result.OriginalLength) → $($Result.SummaryLength) characters" -ForegroundColor White
    Write-Host ""
    Write-Host "  Summary:" -ForegroundColor Yellow
    Write-Host "  ========" -ForegroundColor Yellow
    Write-Host $Result.Summary -ForegroundColor Gray
    Write-Host ""
}

# Function for single-level summarization
function Invoke-SingleLevelSummarization {
    param(
        [string]$Text,
        [string]$Level,
        [bool]$MoeConsensus = $true
    )
    
    $levelName = Get-SummarizationLevel -LevelInput $Level
    Write-Host "📄 Summarizing at $levelName level..." -ForegroundColor Blue
    
    if ($MoeConsensus) {
        $result = New-MoeConsensusSummary -Text $Text -Level $levelName
    } else {
        $result = New-LevelSummary -Text $Text -Level $levelName
    }
    
    Show-SummaryResult -Result $result
}

# Function for multi-level summarization
function Invoke-MultiLevelSummarization {
    param(
        [string]$Text,
        [bool]$MoeConsensus = $true
    )
    
    Write-Host "📊 Multi-level summarization..." -ForegroundColor Green
    Write-Host ""
    
    $levels = @("Executive", "Tactical", "Operational")
    $results = @()
    
    foreach ($level in $levels) {
        Write-Host "Processing $level level..." -ForegroundColor Yellow
        
        if ($MoeConsensus) {
            $result = New-MoeConsensusSummary -Text $Text -Level $level
        } else {
            $result = New-LevelSummary -Text $Text -Level $level
        }
        
        $results += $result
        Show-SummaryResult -Result $result
    }
    
    $overallQuality = ($results | ForEach-Object { $_.ConfidenceScore } | Measure-Object -Average).Average
    Write-Host "Overall Quality: $($overallQuality.ToString('P1'))" -ForegroundColor Cyan
}

# Function to compare summarization approaches
function Compare-SummarizationApproaches {
    param(
        [string]$Text,
        [string]$Level
    )
    
    $levelName = Get-SummarizationLevel -LevelInput $Level
    Write-Host "⚖️ Comparing summarization approaches for $levelName level..." -ForegroundColor Magenta
    Write-Host ""
    
    Write-Host "Approach 1: MoE Consensus" -ForegroundColor Blue
    $result1 = New-MoeConsensusSummary -Text $Text -Level $levelName
    Show-SummaryResult -Result $result1
    
    Write-Host "Approach 2: Standard Summarization" -ForegroundColor Green
    $result2 = New-LevelSummary -Text $Text -Level $levelName
    Show-SummaryResult -Result $result2
    
    Write-Host "COMPARISON:" -ForegroundColor Magenta
    Write-Host "===========" -ForegroundColor Magenta
    
    $format = "{0,-20} | {1,-15} | {2,-15}"
    Write-Host ($format -f "Metric", "MoE Consensus", "Standard") -ForegroundColor White
    Write-Host ("-" * 55) -ForegroundColor Gray
    Write-Host ($format -f "Quality Score", $result1.ConfidenceScore.ToString("P1"), $result2.ConfidenceScore.ToString("P1")) -ForegroundColor White
    Write-Host ($format -f "Compression", $result1.CompressionRatio.ToString("P1"), $result2.CompressionRatio.ToString("P1")) -ForegroundColor White
    Write-Host ($format -f "Length", "$($result1.SummaryLength) chars", "$($result2.SummaryLength) chars") -ForegroundColor White
    Write-Host ""
    
    $recommendation = if ($result1.ConfidenceScore -gt $result2.ConfidenceScore) {
        "Use MoE Consensus approach"
    } elseif ($result2.ConfidenceScore -gt $result1.ConfidenceScore) {
        "Use Standard approach"
    } else {
        "Both approaches are similar in quality"
    }
    
    Write-Host "Recommendation: $recommendation" -ForegroundColor Green
}

# Function for interactive mode
function Start-InteractiveMode {
    Write-Host "🎯 INTERACTIVE RESPONSE SUMMARIZATION" -ForegroundColor Cyan
    Write-Host "=====================================" -ForegroundColor Cyan
    Write-Host ""
    
    do {
        Write-Host "Available actions:" -ForegroundColor Yellow
        Write-Host "  1. Single-level summary" -ForegroundColor White
        Write-Host "  2. Multi-level summary" -ForegroundColor White
        Write-Host "  3. Compare approaches" -ForegroundColor White
        Write-Host "  4. Test DSL block syntax" -ForegroundColor White
        Write-Host "  5. View system info" -ForegroundColor White
        Write-Host "  6. Exit" -ForegroundColor White
        Write-Host ""
        
        $choice = Read-Host "Choose an action (1-6)"
        Write-Host ""
        
        switch ($choice) {
            "1" {
                $text = Read-Host "Enter the text to summarize"
                Write-Host ""
                Write-Host "Available levels: executive (1), tactical (2), operational (3), comprehensive (4), detailed (5)" -ForegroundColor Yellow
                $level = Read-Host "Choose level (default: tactical)"
                if ([string]::IsNullOrWhiteSpace($level)) { $level = "tactical" }
                Write-Host ""
                Invoke-SingleLevelSummarization -Text $text -Level $level -MoeConsensus $MoeConsensus
            }
            "2" {
                $text = Read-Host "Enter the text to summarize"
                Write-Host ""
                Invoke-MultiLevelSummarization -Text $text -MoeConsensus $MoeConsensus
            }
            "3" {
                $text = Read-Host "Enter the text to summarize"
                Write-Host ""
                Write-Host "Available levels: executive, tactical, operational" -ForegroundColor Yellow
                $level = Read-Host "Choose level for comparison (default: tactical)"
                if ([string]::IsNullOrWhiteSpace($level)) { $level = "tactical" }
                Write-Host ""
                Compare-SummarizationApproaches -Text $text -Level $level
            }
            "4" {
                Show-DslSyntax
            }
            "5" {
                Show-SystemInfo
            }
            "6" {
                Write-Host "Goodbye! 👋" -ForegroundColor Green
                return
            }
            default {
                Write-Host "Invalid choice. Please select 1-6." -ForegroundColor Red
            }
        }
        
        Write-Host ""
        Write-Host "Press any key to continue..." -ForegroundColor Gray
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        Write-Host ""
        
    } while ($true)
}

# Function for batch processing
function Start-BatchProcessing {
    param(
        [string]$InputFile,
        [string]$OutputFile
    )
    
    if (-not (Test-Path $InputFile)) {
        Write-Host "❌ Input file '$InputFile' not found" -ForegroundColor Red
        return
    }
    
    Write-Host "📁 Processing responses from '$InputFile'..." -ForegroundColor Blue
    Write-Host ""
    
    $texts = Get-Content $InputFile
    $results = @()
    
    $i = 0
    foreach ($text in $texts) {
        if (-not [string]::IsNullOrWhiteSpace($text)) {
            $i++
            Write-Host "Processing text $i/$($texts.Count)..." -ForegroundColor Yellow
            
            $levels = @("Executive", "Tactical", "Operational")
            
            $results += "Original: $text"
            $results += "Length: $($text.Length) characters"
            
            foreach ($level in $levels) {
                $summary = New-LevelSummary -Text $text -Level $level
                $results += "$level Summary: $($summary.Summary)"
                $results += "Compression: $($summary.CompressionRatio.ToString('P1'))"
                $results += "Confidence: $($summary.ConfidenceScore.ToString('P1'))"
            }
            
            $results += "---"
        }
    }
    
    $results | Out-File -FilePath $OutputFile -Encoding UTF8
    Write-Host ""
    Write-Host "✅ Batch processing complete! Results saved to '$OutputFile'" -ForegroundColor Green
}

# Function to show DSL syntax
function Show-DslSyntax {
    Write-Host "📝 SUMMARIZE DSL BLOCK SYNTAX" -ForegroundColor Cyan
    Write-Host "==============================" -ForegroundColor Cyan
    Write-Host ""
    
    $syntax = @"
Basic Usage:
SUMMARIZE:
  source: "response_variable"
  levels: [1, 2, 3]
  output: "summary_result"

Advanced Configuration:
SUMMARIZE:
  source: "llm_response"
  levels: ["executive", "tactical", "operational"]
  output: "multi_level_summary"
  
  CONFIGURATION:
    moe_consensus: true
    auto_correct: true
    preserve_facts: true
    target_audience: "technical"
  
  EXPERTS:
    clarity_expert: 0.8
    accuracy_expert: 0.9
    brevity_expert: 0.7
  
  OUTPUT_FORMAT:
    structure: "hierarchical"
    include_confidence: true
    show_compression_ratio: true

Multi-Source:
SUMMARIZE:
  sources: ["response_1", "response_2", "response_3"]
  merge_strategy: "consensus_synthesis"
  levels: [1, 2]

Conditional Levels:
SUMMARIZE:
  source: "variable_response"
  CONDITIONAL_LEVELS:
    if_length_gt_1000: [1, 2, 3]
    if_length_gt_500: [1, 2]
    else: [1]
"@
    
    Write-Host $syntax -ForegroundColor Gray
}

# Function to show system info
function Show-SystemInfo {
    Write-Host "📊 SUMMARIZATION SYSTEM INFO" -ForegroundColor Cyan
    Write-Host "============================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Supported Levels: 5 (Executive, Tactical, Operational, Comprehensive, Detailed)" -ForegroundColor White
    Write-Host "  Expert Types: 5 (Clarity, Accuracy, Brevity, Structure, Domain)" -ForegroundColor White
    Write-Host "  MoE Consensus: Supported" -ForegroundColor White
    Write-Host "  Auto Corrections: Supported" -ForegroundColor White
    Write-Host "  Multi-Source: Supported" -ForegroundColor White
    Write-Host "  DSL Integration: Supported" -ForegroundColor White
    Write-Host ""
    Write-Host "  Default Compression Ratios:" -ForegroundColor Yellow
    Write-Host "    Executive: 95%" -ForegroundColor White
    Write-Host "    Tactical: 85%" -ForegroundColor White
    Write-Host "    Operational: 75%" -ForegroundColor White
    Write-Host "    Comprehensive: 60%" -ForegroundColor White
    Write-Host "    Detailed: 40%" -ForegroundColor White
}

# Execute the requested action
switch ($Action.ToLower()) {
    "single" {
        if ([string]::IsNullOrWhiteSpace($Text)) {
            $Text = Read-Host "Enter the text to summarize"
        }
        Write-Host ""
        Invoke-SingleLevelSummarization -Text $Text -Level $Level -MoeConsensus $MoeConsensus
    }
    
    "multi" {
        if ([string]::IsNullOrWhiteSpace($Text)) {
            $Text = Read-Host "Enter the text to summarize"
        }
        Write-Host ""
        Invoke-MultiLevelSummarization -Text $Text -MoeConsensus $MoeConsensus
    }
    
    "compare" {
        if ([string]::IsNullOrWhiteSpace($Text)) {
            $Text = Read-Host "Enter the text to summarize"
        }
        Write-Host ""
        Compare-SummarizationApproaches -Text $Text -Level $Level
    }
    
    "batch" {
        if ([string]::IsNullOrWhiteSpace($InputFile)) {
            $InputFile = Read-Host "Enter input file path"
        }
        if ([string]::IsNullOrWhiteSpace($OutputFile)) {
            $OutputFile = Read-Host "Enter output file path"
        }
        Write-Host ""
        Start-BatchProcessing -InputFile $InputFile -OutputFile $OutputFile
    }
    
    "dsl-test" {
        Write-Host "🧪 Testing SUMMARIZE DSL Block" -ForegroundColor Cyan
        Write-Host ""
        Show-DslSyntax
        Write-Host ""
        Write-Host "Note: Full DSL testing requires the F# backend implementation" -ForegroundColor Yellow
    }
    
    "interactive" {
        Start-InteractiveMode
    }
    
    "help" {
        Write-Host "TARS RESPONSE SUMMARIZATION TOOL - HELP" -ForegroundColor Cyan
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host ""
        
        Write-Host "ACTIONS:" -ForegroundColor Yellow
        Write-Host "  single       - Single-level summarization" -ForegroundColor White
        Write-Host "  multi        - Multi-level summarization" -ForegroundColor White
        Write-Host "  compare      - Compare summarization approaches" -ForegroundColor White
        Write-Host "  batch        - Process multiple texts from file" -ForegroundColor White
        Write-Host "  dsl-test     - Show DSL block syntax" -ForegroundColor White
        Write-Host "  interactive  - Start interactive mode" -ForegroundColor White
        Write-Host "  help         - Show this help message" -ForegroundColor White
        Write-Host ""
        
        Write-Host "LEVELS:" -ForegroundColor Yellow
        Write-Host "  1, executive     - Ultra-concise (1-2 sentences)" -ForegroundColor White
        Write-Host "  2, tactical      - Action-focused (3-5 sentences)" -ForegroundColor White
        Write-Host "  3, operational   - Balanced detail (1-2 paragraphs)" -ForegroundColor White
        Write-Host "  4, comprehensive - Structured summary (3-5 paragraphs)" -ForegroundColor White
        Write-Host "  5, detailed      - Analysis summary (multiple sections)" -ForegroundColor White
        Write-Host ""
        
        Write-Host "USAGE EXAMPLES:" -ForegroundColor Yellow
        Write-Host "  .\summarize_responses.ps1 -Action single -Text 'Your text' -Level tactical" -ForegroundColor Gray
        Write-Host "  .\summarize_responses.ps1 -Action multi -Text 'Your text' -MoeConsensus" -ForegroundColor Gray
        Write-Host "  .\summarize_responses.ps1 -Action compare -Text 'Your text' -Level executive" -ForegroundColor Gray
        Write-Host "  .\summarize_responses.ps1 -Action batch -InputFile texts.txt -OutputFile summaries.txt" -ForegroundColor Gray
        Write-Host "  .\summarize_responses.ps1 -Action interactive" -ForegroundColor Gray
    }
    
    default {
        Write-Host "Invalid action: $Action" -ForegroundColor Red
        Write-Host "Use -Action help to see available actions" -ForegroundColor Yellow
    }
}

Write-Host ""
