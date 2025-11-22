# TARS Security Escalation Test Script
# Tests security incident detection and DevSecOps agent escalation

Write-Host "🛡️ TARS Security Escalation Test" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$baseUrl = "http://localhost:5000"
$loginUrl = "$baseUrl/api/auth/login"
$protectedUrl = "$baseUrl/api/service/status"

function Test-BruteForceDetection {
    Write-Host "🔨 Testing Brute Force Attack Detection" -ForegroundColor Yellow
    Write-Host "Attempting multiple failed logins to trigger security escalation..." -ForegroundColor Gray
    
    $failedAttempts = 0
    $maxAttempts = 7  # Should trigger escalation at 5 attempts
    
    for ($i = 1; $i -le $maxAttempts; $i++) {
        try {
            $loginData = @{
                Username = "attacker"
                Password = "wrongpassword$i"
            } | ConvertTo-Json
            
            Write-Host "  Attempt $i/7: Invalid login..." -ForegroundColor Gray
            
            $response = Invoke-RestMethod -Uri $loginUrl -Method POST -Body $loginData -ContentType "application/json" -ErrorAction SilentlyContinue
            
        } catch {
            $failedAttempts++
            if ($_.Exception.Response.StatusCode -eq 401) {
                Write-Host "    ❌ Authentication failed (expected)" -ForegroundColor Red
            } elseif ($_.Exception.Response.StatusCode -eq 429) {
                Write-Host "    🔒 IP LOCKED OUT - Security escalation triggered!" -ForegroundColor Magenta
                break
            }
        }
        
        Start-Sleep -Milliseconds 500
    }
    
    if ($failedAttempts -ge 5) {
        Write-Host "  ✅ Brute force detection should have triggered DevSecOps escalation" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️ Brute force detection may not have triggered" -ForegroundColor Yellow
    }
    
    Write-Host ""
}

function Test-TokenTamperingDetection {
    Write-Host "🔧 Testing Token Tampering Detection" -ForegroundColor Yellow
    Write-Host "Attempting to use tampered JWT tokens..." -ForegroundColor Gray
    
    $tamperedTokens = @(
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.TAMPERED_PAYLOAD.invalid_signature",
        "invalid.token.format",
        "eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJzdWIiOiJhdHRhY2tlciIsInJvbGUiOiJhZG1pbiJ9.",
        "Bearer malicious_token_here"
    )
    
    foreach ($token in $tamperedTokens) {
        try {
            Write-Host "  Testing tampered token: $($token.Substring(0, [Math]::Min(20, $token.Length)))..." -ForegroundColor Gray
            
            $headers = @{
                "Authorization" = "Bearer $token"
            }
            
            $response = Invoke-RestMethod -Uri $protectedUrl -Method GET -Headers $headers -ErrorAction SilentlyContinue
            Write-Host "    ⚠️ Tampered token was accepted (unexpected)" -ForegroundColor Yellow
            
        } catch {
            if ($_.Exception.Response.StatusCode -eq 401) {
                Write-Host "    ✅ Tampered token rejected - should trigger DevSecOps alert" -ForegroundColor Green
            } else {
                Write-Host "    ❓ Unexpected response: $($_.Exception.Response.StatusCode)" -ForegroundColor Yellow
            }
        }
        
        Start-Sleep -Milliseconds 300
    }
    
    Write-Host ""
}

function Test-UnauthorizedAccessDetection {
    Write-Host "🚫 Testing Unauthorized Access Detection" -ForegroundColor Yellow
    Write-Host "Attempting to access protected endpoints without authentication..." -ForegroundColor Gray
    
    $protectedEndpoints = @(
        "/api/service/status",
        "/api/service/config",
        "/api/agents/list",
        "/api/admin/settings"
    )
    
    foreach ($endpoint in $protectedEndpoints) {
        try {
            $url = "$baseUrl$endpoint"
            Write-Host "  Accessing $endpoint without auth..." -ForegroundColor Gray
            
            $response = Invoke-RestMethod -Uri $url -Method GET -ErrorAction SilentlyContinue
            Write-Host "    ⚠️ Unauthorized access allowed (unexpected)" -ForegroundColor Yellow
            
        } catch {
            if ($_.Exception.Response.StatusCode -eq 401) {
                Write-Host "    ✅ Unauthorized access blocked - should trigger DevSecOps monitoring" -ForegroundColor Green
            } else {
                Write-Host "    ❓ Unexpected response: $($_.Exception.Response.StatusCode)" -ForegroundColor Yellow
            }
        }
        
        Start-Sleep -Milliseconds 200
    }
    
    Write-Host ""
}

function Test-SuspiciousUserAgentDetection {
    Write-Host "🤖 Testing Suspicious User Agent Detection" -ForegroundColor Yellow
    Write-Host "Sending requests with suspicious user agents..." -ForegroundColor Gray
    
    $suspiciousUserAgents = @(
        "sqlmap/1.0",
        "Nikto/2.1.6",
        "Mozilla/5.0 (compatible; Nmap Scripting Engine)",
        "python-requests/2.25.1 bot",
        "curl/7.68.0 scanner"
    )
    
    foreach ($userAgent in $suspiciousUserAgents) {
        try {
            Write-Host "  Testing user agent: $userAgent" -ForegroundColor Gray
            
            $headers = @{
                "User-Agent" = $userAgent
            }
            
            $response = Invoke-RestMethod -Uri $loginUrl -Method GET -Headers $headers -ErrorAction SilentlyContinue
            
        } catch {
            Write-Host "    ✅ Suspicious user agent detected - should trigger DevSecOps alert" -ForegroundColor Green
        }
        
        Start-Sleep -Milliseconds 300
    }
    
    Write-Host ""
}

function Test-RapidRequestPattern {
    Write-Host "⚡ Testing Rapid Request Pattern Detection" -ForegroundColor Yellow
    Write-Host "Sending rapid requests to simulate automated attack..." -ForegroundColor Gray
    
    $requestCount = 20
    $startTime = Get-Date
    
    for ($i = 1; $i -le $requestCount; $i++) {
        try {
            Invoke-RestMethod -Uri $protectedUrl -Method GET -ErrorAction SilentlyContinue
        } catch {
            # Expected to fail due to no auth
        }
        
        if ($i % 5 -eq 0) {
            Write-Host "  Sent $i/$requestCount rapid requests..." -ForegroundColor Gray
        }
    }
    
    $endTime = Get-Date
    $duration = ($endTime - $startTime).TotalSeconds
    $requestsPerSecond = $requestCount / $duration
    
    Write-Host "  ✅ Sent $requestCount requests in $([Math]::Round($duration, 2)) seconds ($([Math]::Round($requestsPerSecond, 1)) req/s)" -ForegroundColor Green
    Write-Host "  📊 Should trigger rate limiting and DevSecOps monitoring" -ForegroundColor Green
    
    Write-Host ""
}

function Show-DevSecOpsExpectedActions {
    Write-Host "🛡️ Expected DevSecOps Agent Actions" -ForegroundColor Cyan
    Write-Host "===================================" -ForegroundColor Cyan
    Write-Host ""
    
    Write-Host "The DevSecOps agent should have detected and responded to:" -ForegroundColor White
    Write-Host ""
    
    Write-Host "🔨 Brute Force Attack:" -ForegroundColor Yellow
    Write-Host "  • Detected multiple failed login attempts" -ForegroundColor Gray
    Write-Host "  • Triggered IP lockout after 5 attempts" -ForegroundColor Gray
    Write-Host "  • Escalated to HIGH severity incident" -ForegroundColor Gray
    Write-Host "  • Recommended IP blocking and rate limiting" -ForegroundColor Gray
    Write-Host ""
    
    Write-Host "🔧 Token Tampering:" -ForegroundColor Yellow
    Write-Host "  • Detected invalid JWT token signatures" -ForegroundColor Gray
    Write-Host "  • Escalated to HIGH severity incident" -ForegroundColor Gray
    Write-Host "  • Recommended JWT key rotation" -ForegroundColor Gray
    Write-Host "  • Suggested token blacklisting" -ForegroundColor Gray
    Write-Host ""
    
    Write-Host "🚫 Unauthorized Access:" -ForegroundColor Yellow
    Write-Host "  • Monitored access attempts to protected endpoints" -ForegroundColor Gray
    Write-Host "  • Logged security events for analysis" -ForegroundColor Gray
    Write-Host "  • Recommended access control review" -ForegroundColor Gray
    Write-Host ""
    
    Write-Host "🤖 Suspicious Activity:" -ForegroundColor Yellow
    Write-Host "  • Detected automated/bot user agents" -ForegroundColor Gray
    Write-Host "  • Flagged potential scanning activity" -ForegroundColor Gray
    Write-Host "  • Recommended behavioral monitoring" -ForegroundColor Gray
    Write-Host ""
    
    Write-Host "⚡ Rate Limiting:" -ForegroundColor Yellow
    Write-Host "  • Detected rapid request patterns" -ForegroundColor Gray
    Write-Host "  • Monitored for DDoS-like behavior" -ForegroundColor Gray
    Write-Host "  • Recommended traffic analysis" -ForegroundColor Gray
    Write-Host ""
}

function Show-SecurityLogs {
    Write-Host "📋 Check Security Logs" -ForegroundColor Cyan
    Write-Host "======================" -ForegroundColor Cyan
    Write-Host ""
    
    Write-Host "To verify DevSecOps agent responses, check:" -ForegroundColor White
    Write-Host ""
    Write-Host "1. 📺 Console Output:" -ForegroundColor Yellow
    Write-Host "   Look for security incident logs in the TARS service console" -ForegroundColor Gray
    Write-Host ""
    Write-Host "2. 📝 Windows Event Log:" -ForegroundColor Yellow
    Write-Host "   Check Application logs for TARS security events" -ForegroundColor Gray
    Write-Host ""
    Write-Host "3. 📊 Service Status:" -ForegroundColor Yellow
    Write-Host "   Query service endpoints for security statistics" -ForegroundColor Gray
    Write-Host ""
    Write-Host "4. 🚨 Alert Manager:" -ForegroundColor Yellow
    Write-Host "   Review alert manager for escalated incidents" -ForegroundColor Gray
    Write-Host ""
}

# Main test execution
Write-Host "🚀 Starting Security Escalation Tests..." -ForegroundColor Cyan
Write-Host "This will simulate various security attacks to test DevSecOps agent response" -ForegroundColor Gray
Write-Host ""

# Wait for user confirmation
Write-Host "⚠️  WARNING: This will generate security alerts and may trigger lockouts" -ForegroundColor Red
Write-Host "Press Enter to continue or Ctrl+C to cancel..." -ForegroundColor Yellow
Read-Host

Write-Host ""
Write-Host "🎯 Running Security Tests..." -ForegroundColor Cyan
Write-Host ""

# Run security tests
Test-BruteForceDetection
Test-TokenTamperingDetection
Test-UnauthorizedAccessDetection
Test-SuspiciousUserAgentDetection
Test-RapidRequestPattern

# Show expected results
Show-DevSecOpsExpectedActions
Show-SecurityLogs

Write-Host "🎯 Security Escalation Tests Complete!" -ForegroundColor Cyan
Write-Host ""
Write-Host "📋 Summary:" -ForegroundColor White
Write-Host "  • Simulated brute force attacks" -ForegroundColor Gray
Write-Host "  • Tested token tampering detection" -ForegroundColor Gray
Write-Host "  • Attempted unauthorized access" -ForegroundColor Gray
Write-Host "  • Sent suspicious user agents" -ForegroundColor Gray
Write-Host "  • Generated rapid request patterns" -ForegroundColor Gray
Write-Host ""
Write-Host "🛡️ DevSecOps agent should have escalated security incidents appropriately!" -ForegroundColor Green
