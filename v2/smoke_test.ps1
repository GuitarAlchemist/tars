# TARS Smoke Test Script
$ErrorActionPreference = "Stop"

$port = 8080
$url = "http://localhost:$port"

function Test-Endpoint($path) {
    Write-Host "Testing $path..." -ForegroundColor Cyan
    try {
        $resp = Invoke-RestMethod -Uri "$url$path" -Method Get
        Write-Host "✅ OK: $(ConvertTo-Json $resp -Compress)" -ForegroundColor Green
        return $resp
    }
    catch {
        Write-Error "❌ FAILED: $_"
        exit 1
    }
}

function Call-Tool($name, $args = "{}") {
    Write-Host "Calling tool '$name'..." -ForegroundColor Cyan
    $payload = @{
        jsonrpc = "2.0"
        id      = [guid]::NewGuid().ToString()
        method  = "tools/call"
        params  = @{
            name      = $name
            arguments = $args
        }
    }
    
    try {
        $json = $payload | ConvertTo-Json -Depth 10
        $resp = Invoke-RestMethod -Uri "$url/message" -Method Post -Body $json -ContentType "application/json"
        
        if ($resp.result.isError) {
            Write-Host "⚠️ Tool returned error (Expected in some cases): $($resp.result.content[0].text)" -ForegroundColor Yellow
        }
        else {
            Write-Host "✅ OK: Result received" -ForegroundColor Green
        }
        
        # Check for traceability IDs
        if ($resp.result._meta.instance_id -and $resp.result._meta.correlation_id) {
            Write-Host "✅ Traceability IDs present: $($resp.result._meta.instance_id) / $($resp.result._meta.correlation_id)" -ForegroundColor Green
        }
        else {
            Write-Error "❌ Missing traceability metadata in response!"
            exit 1
        }
        
        return $resp
    }
    catch {
        Write-Error "❌ FAILED to call tool: $_"
        exit 1
    }
}

# 1. Check Health/About
$info = Test-Endpoint "/about"
if ($info.tool_count -eq 0) {
    Write-Error "❌ Tool registry reported EMPTY!"
    exit 1
}

# 2. List all tools
Call-Tool "list_all_tools"

# 3. Report Status
Call-Tool "report_status"

# 4. Forced Error (invalid tool)
Write-Host "Testing forced error..." -ForegroundColor Cyan
Call-Tool "http_get" '{"url": "http://invalid.domain.that.does.not.exist"}'

# 5. List Tool Errors
Call-Tool "list_tool_errors"

Write-Host "`n🚀 SMOKE TEST PASSED!" -ForegroundColor Green -BackgroundColor Black
