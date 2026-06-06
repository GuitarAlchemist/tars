# Validate all TARS tools via MCP
$ErrorActionPreference = "Stop"

$tools = @(
    "agent_status", "analyze_code", "analyze_file_complexity", "augment_codebase_search",
    "augment_disconnect"
)

Write-Host "Starting TARS Tool Validation (5 tools)" -ForegroundColor Cyan

$success = 0
$fail = 0

foreach ($tool in $tools) {
    Write-Host "Validating $tool..." -NoNewline
    
    $input = "{"toolName": "$tool"}"
    $cmd = "dotnet run --project src/Tars.Interface.Cli/Tars.Interface.Cli.fsproj -- mcp call introspect_tool dotnet 'run --project src/Tars.Interface.Cli/Tars.Interface.Cli.fsproj -- mcp server' --input '$input'"
    
    try {
        $output = Invoke-Expression $cmd
        $joined = $output -join " "
        if ($joined -match "Tool: $tool") {
            Write-Host " [OK]" -ForegroundColor Green
            $success++
        } else {
            Write-Host " [FAIL]" -ForegroundColor Red
            $fail++
        }
    } catch {
        Write-Host " [ERROR]" -ForegroundColor Red
        $fail++
    }
}

Write-Host "Summary:" -ForegroundColor Cyan
Write-Host "Passed: $success" -ForegroundColor Green
Write-Host "Failed: $fail" -ForegroundColor Red
