# Script to analyze the TarsCli monolith structure
$sourceTarsCliPath = "C:\Users\spare\source\repos\tars\Rescue\tars\TarsCli"
$targetRoot = "C:\Users\spare\source\repos\tars"

# Create backup directory
$backupDir = "$targetRoot\Backups\$(Get-Date -Format 'yyyy-MM-dd_HH-mm-ss')"
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null

# Backup the source TarsCli
Copy-Item -Path $sourceTarsCliPath -Destination "$backupDir\TarsCli" -Recurse -Force

# Analyze the directory structure
Write-Host "TarsCli Directory Structure:" -ForegroundColor Green
Get-ChildItem -Path $sourceTarsCliPath -Directory | ForEach-Object {
    Write-Host "- $($_.Name)"
    Get-ChildItem -Path $_.FullName -File -Filter "*.cs" | ForEach-Object {
        Write-Host "  - $($_.Name)"
    }
}

# Analyze the namespaces used in the code
Write-Host "`nTarsCli Namespaces:" -ForegroundColor Green
$namespaces = @{}
Get-ChildItem -Path $sourceTarsCliPath -Recurse -Filter "*.cs" | ForEach-Object {
    $content = Get-Content -Path $_.FullName -Raw
    if ($content -match "namespace\s+([a-zA-Z0-9_.]+)") {
        $namespace = $matches[1]
        if (-not $namespaces.ContainsKey($namespace)) {
            $namespaces[$namespace] = 0
        }
        $namespaces[$namespace]++
    }
}

$namespaces.GetEnumerator() | Sort-Object Name | ForEach-Object {
    Write-Host "- $($_.Key): $($_.Value) files"
}

# Analyze the references to key features
Write-Host "`nKey Feature References:" -ForegroundColor Green
$features = @{
    "Intelligence" = 0
    "ML" = 0
    "DSL" = 0
    "MCP" = 0
    "CodeAnalysis" = 0
    "Docker" = 0
    "WebUI" = 0
    "Commands" = 0
    "Services" = 0
    "Models" = 0
}

Get-ChildItem -Path $sourceTarsCliPath -Recurse -Filter "*.cs" | ForEach-Object {
    $content = Get-Content -Path $_.FullName -Raw
    foreach ($feature in $features.Keys) {
        if ($content -match $feature) {
            $features[$feature]++
        }
    }
}

$features.GetEnumerator() | Sort-Object Name | ForEach-Object {
    Write-Host "- $($_.Key): $($_.Value) files"
}

# Output a summary of findings
Write-Host "`nSummary:" -ForegroundColor Green
Write-Host "The TarsCli monolith can be broken down into the following components:"
foreach ($feature in $features.Keys) {
    if ($features[$feature] -gt 0) {
        Write-Host "- TarsCli.$feature"
    }
}

Write-Host "`nAnalysis complete. Use this information to guide the migration of code to feature-based projects."
