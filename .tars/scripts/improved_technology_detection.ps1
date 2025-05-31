# Improved Technology Detection - Fixes the JavaScript/Python detection bug
# This script provides accurate technology detection based on file analysis

param(
    [Parameter(Mandatory=$true)]
    [string]$ProjectPath
)

function Get-ImprovedTechnologyDetection {
    param([string]$ProjectPath)
    
    Write-Host "🔍 IMPROVED TECHNOLOGY DETECTION" -ForegroundColor Yellow
    Write-Host "===============================" -ForegroundColor Yellow
    
    # Get all project files
    $projectFiles = Get-ChildItem $ProjectPath -File | Where-Object { 
        -not $_.Name.StartsWith(".") -and 
        $_.Name -ne "tars.log" 
    }
    
    # Initialize scoring system
    $techScores = @{
        "Python" = 0
        "JavaScript/Node.js" = 0
        "Java" = 0
        "C#" = 0
        "Go" = 0
        "Rust" = 0
        "PHP" = 0
        "Ruby" = 0
    }
    
    # File extension scoring
    foreach ($file in $projectFiles) {
        switch ($file.Extension.ToLower()) {
            ".py" { 
                $techScores["Python"] += 10
                Write-Host "  📄 $($file.Name): Python (+10)" -ForegroundColor Cyan
            }
            ".js" { 
                $techScores["JavaScript/Node.js"] += 8
                Write-Host "  📄 $($file.Name): JavaScript (+8)" -ForegroundColor Cyan
            }
            ".ts" { 
                $techScores["JavaScript/Node.js"] += 9
                Write-Host "  📄 $($file.Name): TypeScript (+9)" -ForegroundColor Cyan
            }
            ".java" { 
                $techScores["Java"] += 10
                Write-Host "  📄 $($file.Name): Java (+10)" -ForegroundColor Cyan
            }
            ".cs" { 
                $techScores["C#"] += 10
                Write-Host "  📄 $($file.Name): C# (+10)" -ForegroundColor Cyan
            }
            ".go" { 
                $techScores["Go"] += 10
                Write-Host "  📄 $($file.Name): Go (+10)" -ForegroundColor Cyan
            }
            ".rs" { 
                $techScores["Rust"] += 10
                Write-Host "  📄 $($file.Name): Rust (+10)" -ForegroundColor Cyan
            }
            ".php" { 
                $techScores["PHP"] += 10
                Write-Host "  📄 $($file.Name): PHP (+10)" -ForegroundColor Cyan
            }
            ".rb" { 
                $techScores["Ruby"] += 10
                Write-Host "  📄 $($file.Name): Ruby (+10)" -ForegroundColor Cyan
            }
        }
    }
    
    # Main file detection (higher weight)
    $mainFiles = @("main.py", "app.py", "server.py", "index.js", "app.js", "server.js", "Main.java", "Program.cs", "main.go", "main.rs")
    foreach ($mainFile in $mainFiles) {
        if ($projectFiles | Where-Object { $_.Name -eq $mainFile }) {
            switch ($mainFile) {
                { $_ -match "\.py$" } { 
                    $techScores["Python"] += 20
                    Write-Host "  🎯 Main file detected: $mainFile (Python +20)" -ForegroundColor Green
                }
                { $_ -match "\.js$" } { 
                    $techScores["JavaScript/Node.js"] += 20
                    Write-Host "  🎯 Main file detected: $mainFile (JavaScript +20)" -ForegroundColor Green
                }
                { $_ -match "\.java$" } { 
                    $techScores["Java"] += 20
                    Write-Host "  🎯 Main file detected: $mainFile (Java +20)" -ForegroundColor Green
                }
                { $_ -match "\.cs$" } { 
                    $techScores["C#"] += 20
                    Write-Host "  🎯 Main file detected: $mainFile (C# +20)" -ForegroundColor Green
                }
                { $_ -match "\.go$" } { 
                    $techScores["Go"] += 20
                    Write-Host "  🎯 Main file detected: $mainFile (Go +20)" -ForegroundColor Green
                }
                { $_ -match "\.rs$" } { 
                    $techScores["Rust"] += 20
                    Write-Host "  🎯 Main file detected: $mainFile (Rust +20)" -ForegroundColor Green
                }
            }
        }
    }
    
    # Configuration file detection
    $configFiles = @{
        "requirements.txt" = "Python"
        "setup.py" = "Python"
        "Pipfile" = "Python"
        "package.json" = "JavaScript/Node.js"
        "yarn.lock" = "JavaScript/Node.js"
        "pom.xml" = "Java"
        "build.gradle" = "Java"
        "*.csproj" = "C#"
        "*.fsproj" = "C#"
        "go.mod" = "Go"
        "Cargo.toml" = "Rust"
        "composer.json" = "PHP"
        "Gemfile" = "Ruby"
    }
    
    foreach ($configPattern in $configFiles.Keys) {
        $tech = $configFiles[$configPattern]
        if ($configPattern.Contains("*")) {
            $extension = $configPattern.Replace("*", "")
            if ($projectFiles | Where-Object { $_.Name.EndsWith($extension) }) {
                $techScores[$tech] += 15
                Write-Host "  ⚙️ Config file detected: *$extension ($tech +15)" -ForegroundColor Magenta
            }
        } else {
            if ($projectFiles | Where-Object { $_.Name -eq $configPattern }) {
                $techScores[$tech] += 15
                Write-Host "  ⚙️ Config file detected: $configPattern ($tech +15)" -ForegroundColor Magenta
            }
        }
    }
    
    # File count weighting (more files = higher confidence)
    foreach ($tech in $techScores.Keys) {
        $fileCount = switch ($tech) {
            "Python" { ($projectFiles | Where-Object { $_.Extension -eq ".py" }).Count }
            "JavaScript/Node.js" { ($projectFiles | Where-Object { $_.Extension -in @(".js", ".ts") }).Count }
            "Java" { ($projectFiles | Where-Object { $_.Extension -eq ".java" }).Count }
            "C#" { ($projectFiles | Where-Object { $_.Extension -eq ".cs" }).Count }
            "Go" { ($projectFiles | Where-Object { $_.Extension -eq ".go" }).Count }
            "Rust" { ($projectFiles | Where-Object { $_.Extension -eq ".rs" }).Count }
            "PHP" { ($projectFiles | Where-Object { $_.Extension -eq ".php" }).Count }
            "Ruby" { ($projectFiles | Where-Object { $_.Extension -eq ".rb" }).Count }
            default { 0 }
        }
        
        if ($fileCount -gt 1) {
            $bonus = ($fileCount - 1) * 3
            $techScores[$tech] += $bonus
            Write-Host "  📊 File count bonus: $tech has $fileCount files (+$bonus)" -ForegroundColor Blue
        }
    }
    
    # Find the winning technology
    $sortedTechs = $techScores.GetEnumerator() | Sort-Object Value -Descending
    $winningTech = $sortedTechs[0]
    $secondTech = $sortedTechs[1]
    
    # Calculate confidence
    $totalScore = ($techScores.Values | Measure-Object -Sum).Sum
    $confidence = if ($totalScore -gt 0) { $winningTech.Value / $totalScore } else { 0.5 }
    
    Write-Host ""
    Write-Host "📊 TECHNOLOGY SCORES:" -ForegroundColor Yellow
    foreach ($tech in $sortedTechs) {
        if ($tech.Value -gt 0) {
            $percentage = if ($totalScore -gt 0) { ($tech.Value / $totalScore * 100) } else { 0 }
            $indicator = if ($tech.Name -eq $winningTech.Name) { "🏆" } else { "  " }
            Write-Host "  $indicator $($tech.Name): $($tech.Value) points ($($percentage.ToString('F1'))%)" -ForegroundColor $(if ($tech.Name -eq $winningTech.Name) { "Green" } else { "White" })
        }
    }
    
    Write-Host ""
    Write-Host "🎯 DETECTION RESULT:" -ForegroundColor Green
    Write-Host "  Primary Technology: $($winningTech.Name)" -ForegroundColor Green
    Write-Host "  Confidence: $($confidence.ToString('P1'))" -ForegroundColor Green
    Write-Host "  Score: $($winningTech.Value)/$totalScore points" -ForegroundColor Green
    
    # Determine if detection is reliable
    $isReliable = $confidence -gt 0.6 -and $winningTech.Value -gt 10
    Write-Host "  Reliability: $(if ($isReliable) { '✅ HIGH' } else { '⚠️ LOW' })" -ForegroundColor $(if ($isReliable) { "Green" } else { "Yellow" })
    
    return @{
        Technology = $winningTech.Name
        Confidence = $confidence
        Score = $winningTech.Value
        TotalScore = $totalScore
        IsReliable = $isReliable
        AllScores = $techScores
        SecondaryTechnology = $secondTech.Name
    }
}

# Test the improved detection
if ($ProjectPath) {
    $result = Get-ImprovedTechnologyDetection -ProjectPath $ProjectPath
    
    Write-Host ""
    Write-Host "🎉 IMPROVED TECHNOLOGY DETECTION COMPLETE" -ForegroundColor Green
    Write-Host "=========================================" -ForegroundColor Green
    Write-Host "🔧 Technology: $($result.Technology)" -ForegroundColor White
    Write-Host "📊 Confidence: $($result.Confidence.ToString('P1'))" -ForegroundColor White
    Write-Host "🎯 Reliability: $(if ($result.IsReliable) { 'HIGH' } else { 'LOW' })" -ForegroundColor White
    
    return $result
}
