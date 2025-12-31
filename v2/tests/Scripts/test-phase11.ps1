# Test Phase 11: Plan Integration
# This script verifies that agents can interact with the planning system.

$ErrorActionPreference = "Stop"

Write-Host "Starting Phase 11 Test Protocol" -ForegroundColor Cyan

# 1. Create a new plan
Write-Host "1. Creating Test Plan..." -ForegroundColor Yellow
$cmd = "dotnet run --project src/Tars.Interface.Cli/Tars.Interface.Cli.fsproj -- plan new 'Phase 11 Verification Protocol'"
Write-Host "> $cmd" -ForegroundColor Gray
$planOutput = Invoke-Expression $cmd
$planOutput | Out-Host

# Extract Plan ID (Format: Plan created: p:GUID - Name)
$planId = $null
foreach ($line in $planOutput) {
    if ($line -match "p:([a-f0-9]+)") {
        $planId = "p:" + $matches[1]
        break
    }
}

if ($planId) {
    Write-Host "Plan Created: $planId" -ForegroundColor Green
} else {
    Write-Error "Failed to extract Plan ID from output."
}

# 2. Verify Plan List
Write-Host "2. Verifying Plan List..." -ForegroundColor Yellow
$cmd = "dotnet run --project src/Tars.Interface.Cli/Tars.Interface.Cli.fsproj -- plan list"
Write-Host "> $cmd" -ForegroundColor Gray
$listOutput = Invoke-Expression $cmd
$listOutput | Out-Host

if ($listOutput -match $planId) {
    Write-Host "Plan found in list." -ForegroundColor Green
} else {
    Write-Error "Plan not found in list."
}

# 3. Activate the plan (by starting first step via Agent)
Write-Host "3. Running Agent with Plan Context..." -ForegroundColor Yellow
# Using an extremely direct prompt and a stronger model
$prompt = "TASK: CALL THE 'update_plan_step' TOOL WITH INPUT '1' IMMEDIATELY. DO NOT THINK. DO NOT SEARCH. JUST CALL THE TOOL. INPUT IS JUST THE NUMBER '1'."
$cmd = "dotnet run --project src/Tars.Interface.Cli/Tars.Interface.Cli.fsproj -- agent react '$prompt' --plan $planId --max-steps 5 --model deepseek-r1:8b"
Write-Host "> $cmd" -ForegroundColor Gray
$agentOutput = Invoke-Expression $cmd
$agentOutput | Out-Host

# 4. Verify Plan Status (Check step status)
Write-Host "4. Verifying Plan Execution..." -ForegroundColor Yellow
$cmd = "dotnet run --project src/Tars.Interface.Cli/Tars.Interface.Cli.fsproj -- plan show $planId"
Write-Host "> $cmd" -ForegroundColor Gray
$showOutput = Invoke-Expression $cmd
$showOutput | Out-Host

if ($showOutput -match "Completed") {
    Write-Host "✅ SUCCESS: Step 1 is marked Completed!" -ForegroundColor Green
} else {
    Write-Host "❌ FAILURE: Step 1 is NOT Completed." -ForegroundColor Red
    exit 1
}

Write-Host "Finished Test." -ForegroundColor Cyan