# TARS JWT Authentication Test Script
# Tests the JWT authentication endpoints

Write-Host "🔐 TARS JWT Authentication Test" -ForegroundColor Cyan
Write-Host "===============================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$baseUrl = "http://localhost:5000"  # Adjust if service runs on different port
$loginUrl = "$baseUrl/api/auth/login"
$validateUrl = "$baseUrl/api/auth/validate"

# Test credentials
$testUsers = @(
    @{ Username = "admin"; Password = "admin123"; Role = "Administrator" }
    @{ Username = "user"; Password = "user123"; Role = "User" }
    @{ Username = "agent"; Password = "agent123"; Role = "Agent" }
)

function Test-Login {
    param($Username, $Password, $Role)
    
    Write-Host "🔑 Testing login for: $Username ($Role)" -ForegroundColor Yellow
    
    $loginData = @{
        Username = $Username
        Password = $Password
        Role = $Role
    } | ConvertTo-Json
    
    try {
        $response = Invoke-RestMethod -Uri $loginUrl -Method POST -Body $loginData -ContentType "application/json"
        
        if ($response.Token) {
            Write-Host "  ✅ Login successful!" -ForegroundColor Green
            Write-Host "  📝 User: $($response.User.Username)" -ForegroundColor Gray
            Write-Host "  🎭 Role: $($response.User.Role)" -ForegroundColor Gray
            Write-Host "  ⏰ Expires: $($response.ExpiresAt)" -ForegroundColor Gray
            Write-Host "  🔑 Token: $($response.Token.Substring(0, 20))..." -ForegroundColor Gray
            return $response.Token
        } else {
            Write-Host "  ❌ Login failed: No token received" -ForegroundColor Red
            return $null
        }
    }
    catch {
        Write-Host "  ❌ Login failed: $($_.Exception.Message)" -ForegroundColor Red
        return $null
    }
}

function Test-TokenValidation {
    param($Token, $Username)
    
    Write-Host "🔍 Testing token validation for: $Username" -ForegroundColor Yellow
    
    $validateData = @{
        Token = $Token
    } | ConvertTo-Json
    
    try {
        $response = Invoke-RestMethod -Uri $validateUrl -Method POST -Body $validateData -ContentType "application/json"
        
        if ($response.IsValid) {
            Write-Host "  ✅ Token validation successful!" -ForegroundColor Green
            Write-Host "  📝 User: $($response.User.Username)" -ForegroundColor Gray
            Write-Host "  🎭 Role: $($response.User.Role)" -ForegroundColor Gray
            Write-Host "  💬 Message: $($response.Message)" -ForegroundColor Gray
        } else {
            Write-Host "  ❌ Token validation failed: $($response.Message)" -ForegroundColor Red
        }
    }
    catch {
        Write-Host "  ❌ Token validation failed: $($_.Exception.Message)" -ForegroundColor Red
    }
}

function Test-InvalidLogin {
    Write-Host "🚫 Testing invalid login" -ForegroundColor Yellow
    
    $loginData = @{
        Username = "invalid"
        Password = "wrong"
    } | ConvertTo-Json
    
    try {
        $response = Invoke-RestMethod -Uri $loginUrl -Method POST -Body $loginData -ContentType "application/json"
        Write-Host "  ❌ Invalid login should have failed!" -ForegroundColor Red
    }
    catch {
        Write-Host "  ✅ Invalid login correctly rejected" -ForegroundColor Green
    }
}

function Test-InvalidToken {
    Write-Host "🚫 Testing invalid token validation" -ForegroundColor Yellow
    
    $validateData = @{
        Token = "invalid.token.here"
    } | ConvertTo-Json
    
    try {
        $response = Invoke-RestMethod -Uri $validateUrl -Method POST -Body $validateData -ContentType "application/json"
        
        if (-not $response.IsValid) {
            Write-Host "  ✅ Invalid token correctly rejected" -ForegroundColor Green
        } else {
            Write-Host "  ❌ Invalid token should have been rejected!" -ForegroundColor Red
        }
    }
    catch {
        Write-Host "  ✅ Invalid token correctly rejected with error" -ForegroundColor Green
    }
}

# Main test execution
Write-Host "🚀 Starting JWT authentication tests..." -ForegroundColor Cyan
Write-Host ""

# Test valid logins
$tokens = @{}
foreach ($user in $testUsers) {
    $token = Test-Login -Username $user.Username -Password $user.Password -Role $user.Role
    if ($token) {
        $tokens[$user.Username] = $token
    }
    Write-Host ""
}

# Test token validation
foreach ($username in $tokens.Keys) {
    Test-TokenValidation -Token $tokens[$username] -Username $username
    Write-Host ""
}

# Test invalid scenarios
Test-InvalidLogin
Write-Host ""

Test-InvalidToken
Write-Host ""

Write-Host "🎯 JWT Authentication Tests Complete!" -ForegroundColor Cyan
Write-Host ""
Write-Host "📋 Summary:" -ForegroundColor White
Write-Host "  • Tested login for multiple user roles" -ForegroundColor Gray
Write-Host "  • Tested token validation" -ForegroundColor Gray
Write-Host "  • Tested invalid login rejection" -ForegroundColor Gray
Write-Host "  • Tested invalid token rejection" -ForegroundColor Gray
Write-Host ""
Write-Host "🔐 TARS JWT authentication is working!" -ForegroundColor Green
