$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Write-Step {
    param($Message)
    Write-Host ""
    Write-Host "👉 $Message" -ForegroundColor Cyan
}

function Write-Success {
    param($Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

function Write-Failure {
    param($Message)
    Write-Host "❌ $Message" -ForegroundColor Red
    exit 1
}

$Project = "src/Tars.Interface.Cli/Tars.Interface.Cli.fsproj"

# 1. Build
Write-Step "Building TARS CLI..."
dotnet build $Project -c Debug -v q
if ($LASTEXITCODE -ne 0) { Write-Failure "Build failed" }

# 1.5. Verify Diagnostics
Write-Step "Verifying Diagnostics..."
$diagOutput = dotnet run --project $Project --no-build -- diag --verbose 2>&1
if ($diagOutput -match "FAIL") {
    Write-Host $diagOutput
    Write-Failure "Diagnostics Failed (Found 'FAIL' in output)."
}
else {
    Write-Success "Diagnostics Verified (No failures detected)"
}

# 2. Verify Chain of Thought
Write-Step "Verifying Chain of Thought (CoT)..."
$cotOutput = dotnet run --project $Project --no-build -- agent cot "What is 5 plus 3?" --verbose 2>&1
if ($cotOutput -match "Success") {
    Write-Success "CoT Verified"
}
else {
    Write-Failure "CoT Failed. Output:`n$cotOutput"
}

# 3. Verify Workflow of Thought (WoT)
Write-Step "Verifying Workflow of Thought (WoT)..."
$wotOutput = dotnet run --project $Project --no-build -- agent wot "Name three colors" --verbose 2>&1
if ($wotOutput -match "Success") {
    Write-Success "WoT Verified"
}
else {
    Write-Failure "WoT Failed. Output:`n$wotOutput"
}

# 4. Verify Knowledge Ledger
Write-Step "Verifying Knowledge Ledger..."
$knowOutput = dotnet run --project $Project --no-build -- know status --pg 2>&1
if ($knowOutput -match "Valid Beliefs") {
    Write-Success "Knowledge Ledger Verified"
}
else {
    Write-Failure "Knowledge Ledger Failed. Output:`n$knowOutput"
}

# 5. Verify RAG Demo (Quick Mode)
Write-Step "Verifying RAG Demo..."
$ragOutput = dotnet run --project $Project --no-build -- demo-rag --quick --scenario 1 2>&1
if ($ragOutput -match "scenarios completed successfully") {
    Write-Success "RAG Demo Verified"
}
else {
    Write-Failure "RAG Demo Failed. Output:`n$ragOutput"
}

Write-Host ""
Write-Host "🎉 ALL CAPABILITIES VERIFIED" -ForegroundColor Green
exit 0
