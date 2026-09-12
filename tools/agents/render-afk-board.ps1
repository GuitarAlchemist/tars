# tools/agents/render-afk-board.ps1 - Generate the human-readable HTML and Markdown boards from JSON artifacts in PowerShell.

$ErrorActionPreference = "Stop"

# Locate Repo Root
$gitRoot = git rev-parse --show-toplevel 2>$null
if ($null -eq $gitRoot) {
    $RepoRoot = Get-Location
} else {
    $RepoRoot = $gitRoot
}

Write-Host "=== Rendering AFK Live Agent Board (PowerShell) ==="

$GovRunsDir = Join-Path $RepoRoot "governance/agents/live/runs"
$HtmlOutDir = Join-Path $RepoRoot "docs/agents/live/html"
$HtmlOutFile = Join-Path $HtmlOutDir "afk-runs.json"
$MdOutFile = Join-Path $RepoRoot "docs/agents/live/afk-board.md"

# Ensure output directories exist
if (-not (Test-Path $HtmlOutDir)) {
    New-Item -ItemType Directory -Force -Path $HtmlOutDir | Out-Null
}

$compiledRuns = @()

# 1. Gather runs from governance/agents/live/runs/*.json
if (Test-Path $GovRunsDir) {
    $jsonFiles = Get-ChildItem -Path $GovRunsDir -Filter "*.json" | Sort-Object Name
    foreach ($file in $jsonFiles) {
        try {
            $content = Get-Content -Raw -Path $file.FullName -ErrorAction Stop
            $runData = ConvertFrom-Json $content

            $state = if ($runData.state) { $runData.state } else { "queued" }

            # Synthesize flags based on state vocabulary
            $isStale = ($state -eq "stale") -or ($runData.is_stale -eq $true)
            $isBlocked = ($state -eq "blocked") -or ($runData.is_blocked -eq $true)
            $isDuplicate = ($state -in @("duplicate", "duplicate-agent-pr")) -or ($runData.is_duplicate -eq $true)
            $ciFailing = ($state -eq "ci-failing") -or ($runData.ci_failing -eq $true)
            $needsReview = ($state -in @("needs-human-review", "human-review")) -or ($runData.needs_human_review -eq $true)

            $evidence = @()
            if ($runData.evidence) {
                foreach ($e in $runData.evidence) {
                    if ($e -is [string]) {
                        $evidence += $e
                    } elseif ($null -ne $e.type) {
                        $finding = if ($e.finding) { $e.finding } else { "Reference" }
                        $evidence += "$($e.type): $finding"
                    }
                }
            }

            $issue = if ($runData.issue) {
                if ($runData.issue.ToString().StartsWith("#")) { $runData.issue.ToString() } else { "#" + $runData.issue.ToString() }
            } else {
                ""
            }

            $pr = if ($runData.pr) {
                if ($runData.pr.ToString().StartsWith("#")) { $runData.pr.ToString() } else { "#" + $runData.pr.ToString() }
            } else {
                ""
            }

            $agent = if ($runData.agent) { $runData.agent.ToString() } else { "Unknown" }
            # Capitalize agent name
            if ($agent.Length -gt 0) {
                $agent = $agent.Substring(0,1).ToUpper() + $agent.Substring(1)
            }

            $normalizedRun = [PSCustomObject]@{
                issue              = $issue
                title              = if ($runData.summary) { $runData.summary } else { "Untitled Agent Run" }
                agent              = $agent
                pr                 = $pr
                state              = $state
                risk               = if ($runData.risk) { $runData.risk } else { "low" }
                last_signal        = if ($runData.last_signal_at) { $runData.last_signal_at } else { if ($runData.last_signal) { $runData.last_signal } else { "" } }
                next_action        = if ($runData.next_action) { $runData.next_action } else { "None specified" }
                evidence           = $evidence
                is_stale           = [bool]$isStale
                is_blocked         = [bool]$isBlocked
                is_duplicate       = [bool]$isDuplicate
                ci_failing         = [bool]$ciFailing
                needs_human_review = [bool]$needsReview
            }
            $compiledRuns += $normalizedRun
        } catch {
            Write-Warning "Failed to parse $($file.Name): $_"
        }
    }
}

# 2. Write compiled runs array into HTML folder
try {
    $jsonOut = ConvertTo-Json -InputObject $compiledRuns -Depth 100
    [System.IO.File]::WriteAllText($HtmlOutFile, $jsonOut)
    Write-Host "Successfully compiled $($compiledRuns.Count) runs into $HtmlOutFile"
} catch {
    Write-Error "Failed to write compiled runs to $HtmlOutFile: $_"
}

# 3. Generate Markdown Board
try {
    $timestamp = [DateTime]::UtcNow.ToString("yyyy-MM-dd HH:mm:ss UTC")

    $totalCount = $compiledRuns.Count
    $reviewCount = ($compiledRuns | Where-Object { $_.needs_human_review -or $_.state -eq "needs-human-review" }).Count
    $blockedCount = ($compiledRuns | Where-Object { $_.is_blocked -or $_.state -eq "blocked" }).Count
    $failingCount = ($compiledRuns | Where-Object { $_.ci_failing -or $_.state -eq "ci-failing" }).Count
    $staleCount = ($compiledRuns | Where-Object { $_.is_stale -or $_.state -eq "stale" }).Count
    $doneCount = ($compiledRuns | Where-Object { $_.state -eq "done" }).Count

    $mdContent = @"
# AFK Live Agent Board

*Last Updated:* $timestamp

This is the human-readable Markdown view of the parallel cloud-agent tracking system. The static [HTML Dashboard](./html/index.html) is also available for live scanning.

## Summary Counts

| Metric | Count |
|--------|-------|
| **Total Active Runs** | $totalCount |
| **Needs Human Review** | $reviewCount |
| **Blocked Runs** | $blockedCount |
| **CI Failing Runs** | $failingCount |
| **Stale Runs** | $staleCount |
| **Completed (Done) Runs** | $doneCount |

## Active Runs

"@

    if ($compiledRuns.Count -gt 0) {
        $mdContent += "`n| Issue / Task | Agent | PR | State | Risk | Last Signal | Next Action |`n"
        $mdContent += "|---|---|---|---|---|---|---|`n"
        foreach ($run in $compiledRuns) {
            $issueStr = if ($run.issue) { $run.issue } else { "N/A" }
            $titleStr = $run.title
            $agentStr = $run.agent
            $prStr = if ($run.pr) { $run.pr } else { "None" }
            $stateStr = $run.state
            $riskStr = $run.risk.ToUpper()
            $signalStr = if ($run.last_signal) { $run.last_signal } else { "N/A" }
            $actionStr = $run.next_action

            $stateLabel = $stateStr
            if ($run.is_stale) { $stateLabel += " ⚠️ STALE" }
            if ($run.is_blocked) { $stateLabel += " 🚫 BLOCKED" }
            if ($run.is_duplicate) { $stateLabel += " 📋 DUPLICATE" }
            if ($run.ci_failing) { $stateLabel += " ❌ FAILING" }
            if ($run.needs_human_review) { $stateLabel += " 👀 NEEDS REVIEW" }

            $mdContent += "| $issueStr - $titleStr | $agentStr | $prStr | `$stateLabel` | $riskStr | $signalStr | $actionStr |`n"
        }
    } else {
        $mdContent += "*No active cloud agent work is currently registered.*\n"
    }

    [System.IO.File]::WriteAllText($MdOutFile, $mdContent)
    Write-Host "Successfully generated Markdown board in $MdOutFile"
} catch {
    Write-Error "Failed to write Markdown board: $_"
}

Write-Host "=== Rendering Completed ==="
