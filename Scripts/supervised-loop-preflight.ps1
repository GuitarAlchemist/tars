#requires -Version 7
<#
.SYNOPSIS
    Preflight gate for the tars supervised autonomous loop.
.DESCRIPTION
    Runs before any /supervised-loop cycle. Reads agent-blackbox.loop-policy.json,
    confirms the working tree is clean of out-of-scope edits (any file changed
    that is NOT inside allow_edit, OR any file changed that matches a
    protected_paths glob), and emits a JSON status to stdout. Exits non-zero on
    any block.

    Honors:
      - $env:USERPROFILE/.demerzel/HALT-ALL  (global halt marker)
      - .STOP at repo root                   (per-repo halt marker)

    This is intentionally minimal — the heavy lifting lives in the
    agent-blackbox CLI (`harness-audit`, `pr-preflight`). This script is the
    repo-local guard that gives the loop a fast first-fence check.
.NOTES
    Companion to agent-blackbox.loop-policy.json and
    .claude/skills/supervised-loop/SKILL.md.
#>
[CmdletBinding()]
param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
    [switch]$Json
)

$ErrorActionPreference = 'Stop'

function Write-Status {
    param([hashtable]$Payload, [int]$ExitCode)
    if ($Json) {
        $Payload | ConvertTo-Json -Depth 8
    }
    else {
        Write-Host ("status   : {0}" -f $Payload.status)
        Write-Host ("verdict  : {0}" -f $Payload.verdict)
        if ($Payload.findings) {
            Write-Host "findings :"
            foreach ($f in $Payload.findings) { Write-Host "  - $f" }
        }
    }
    exit $ExitCode
}

$findings = New-Object System.Collections.Generic.List[string]

# Halt markers
$haltAll = Join-Path $env:USERPROFILE '.demerzel/HALT-ALL'
if (Test-Path -LiteralPath $haltAll) {
    $findings.Add("HALT-ALL marker present at $haltAll")
}
$stopMarker = Join-Path $RepoRoot '.STOP'
if (Test-Path -LiteralPath $stopMarker) {
    $findings.Add("Repo .STOP marker present at $stopMarker")
}

# Load loop policy
$policyPath = Join-Path $RepoRoot 'agent-blackbox.loop-policy.json'
if (-not (Test-Path -LiteralPath $policyPath)) {
    Write-Status -Payload @{
        status   = 'block'
        verdict  = 'missing-loop-policy'
        findings = @("agent-blackbox.loop-policy.json missing at $policyPath")
    } -ExitCode 2
}
$policy = Get-Content -LiteralPath $policyPath -Raw | ConvertFrom-Json
$allowEdit = $policy.allow_edit
$protected = $policy.protected_paths

# Get the dirty file set (porcelain, NUL-safe enough for our needs)
Push-Location $RepoRoot
try {
    $dirtyRaw = git status --porcelain 2>$null
}
finally {
    Pop-Location
}
$dirty = @()
if ($dirtyRaw) {
    $dirty = $dirtyRaw -split "`n" | Where-Object { $_ } | ForEach-Object {
        # porcelain: XY <path>  (paths use forward slashes)
        ($_ -replace '^..\s+', '').Trim('"')
    }
}

function Test-Glob {
    param([string]$Path, [string[]]$Patterns)
    if (-not $Patterns) { return $false }
    foreach ($p in $Patterns) {
        # Convert glob to regex (very minimal: ** -> .*, * -> [^/]*)
        $regex = '^' + [Regex]::Escape($p).Replace('\*\*', '.*').Replace('\*', '[^/]*') + '$'
        if ($Path -match $regex) { return $true }
    }
    return $false
}

$outOfScope = @()
$protectedHit = @()
foreach ($file in $dirty) {
    if (Test-Glob -Path $file -Patterns $protected) { $protectedHit += $file; continue }
    if (-not (Test-Glob -Path $file -Patterns $allowEdit)) { $outOfScope += $file }
}

if ($protectedHit.Count -gt 0) {
    $findings.Add("Protected paths dirty (loop must not edit): $($protectedHit -join ', ')")
}
if ($outOfScope.Count -gt 0) {
    $findings.Add("Out-of-scope dirty (not in allow_edit): $($outOfScope -join ', ')")
}

$exitCode = 0
$status = 'ok'
$verdict = 'loop-eligible'
if ($findings.Count -gt 0) {
    $status = 'block'
    $verdict = 'loop-blocked'
    $exitCode = 1
}

Write-Status -Payload @{
    status         = $status
    verdict        = $verdict
    findings       = $findings.ToArray()
    dirtyFileCount = $dirty.Count
    repoRoot       = $RepoRoot
} -ExitCode $exitCode
