# TARS SUPERINTELLIGENCE DEPLOYMENT ORCHESTRATOR
# Complete deployment of Tier 1-11 superintelligence with blue/green strategy

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("full", "blue", "green", "status", "test", "rollback")]
    [string]$Action = "status",
    
    [Parameter(Mandatory=$false)]
    [switch]$Production,
    
    [Parameter(Mandatory=$false)]
    [switch]$SkipTests,
    
    [Parameter(Mandatory=$false)]
    [switch]$Force
)

# Configuration
$ComposeFile = "docker-compose.superintelligence.yml"
$LogPath = "./logs/superintelligence-deployment.log"

# Ensure logs directory exists
if (-not (Test-Path "./logs")) {
    New-Item -ItemType Directory -Path "./logs" -Force
}

# Logging function
function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] [$Level] $Message"
    Write-Host $logMessage
    Add-Content -Path $LogPath -Value $logMessage
}

# Test superintelligence capabilities
function Test-SuperintelligenceCapabilities {
    Write-Log "🧠 Testing TARS Superintelligence Capabilities (All 11 Tiers)"
    Write-Log "============================================================"
    
    try {
        # Run the superintelligence test
        $testResult = & dotnet fsi TarsCompleteSuperintelligenceTest.fsx
        
        if ($LASTEXITCODE -eq 0) {
            Write-Log "✅ Superintelligence test passed - All 11 tiers operational" "SUCCESS"
            return $true
        } else {
            Write-Log "❌ Superintelligence test failed" "ERROR"
            return $false
        }
    }
    catch {
        Write-Log "❌ Error running superintelligence test: $($_.Exception.Message)" "ERROR"
        return $false
    }
}

# Deploy infrastructure services
function Deploy-Infrastructure {
    Write-Log "🏗️ Deploying TARS Superintelligence Infrastructure"
    
    try {
        # Create network if it doesn't exist
        $networkExists = docker network ls --filter name=tars-superintelligence --format "{{.Name}}" | Select-String "tars-superintelligence"
        if (-not $networkExists) {
            Write-Log "Creating tars-superintelligence network"
            docker network create tars-superintelligence
        }
        
        # Deploy infrastructure services
        Write-Log "Starting infrastructure services..."
        docker-compose -f $ComposeFile up -d mongodb chromadb redis prometheus grafana
        
        # Wait for services to be ready
        Write-Log "Waiting for infrastructure services to be ready..."
        Start-Sleep -Seconds 30
        
        # Check infrastructure health
        $services = @("mongodb", "chromadb", "redis")
        $allHealthy = $true
        
        foreach ($service in $services) {
            $health = docker inspect --format='{{.State.Health.Status}}' "tars-$service-superintelligence" 2>$null
            if ($health -eq "healthy" -or $health -eq "") {
                Write-Log "✅ $service is healthy" "SUCCESS"
            } else {
                Write-Log "❌ $service is not healthy: $health" "ERROR"
                $allHealthy = $false
            }
        }
        
        return $allHealthy
    }
    catch {
        Write-Log "❌ Failed to deploy infrastructure: $($_.Exception.Message)" "ERROR"
        return $false
    }
}

# Deploy superintelligence environment
function Deploy-SuperintelligenceEnvironment {
    param([string]$Environment)
    
    Write-Log "🚀 Deploying TARS Superintelligence - $Environment Environment"
    
    try {
        # Build and start the environment
        Write-Log "Building superintelligence image for $Environment..."
        docker-compose -f $ComposeFile build tars-superintelligence-$Environment
        
        Write-Log "Starting $Environment environment..."
        docker-compose -f $ComposeFile up -d tars-superintelligence-$Environment
        
        # Wait for startup
        Write-Log "Waiting for $Environment environment to start..."
        Start-Sleep -Seconds 45
        
        # Health check
        $port = if ($Environment -eq "blue") { 8080 } else { 8090 }
        $maxRetries = 10
        $retryCount = 0
        $isHealthy = $false
        
        while ($retryCount -lt $maxRetries -and -not $isHealthy) {
            try {
                $response = Invoke-RestMethod -Uri "http://localhost:$($port+2)/health/superintelligence" -TimeoutSec 10
                if ($response.status -eq "healthy") {
                    $isHealthy = $true
                    Write-Log "✅ $Environment environment is healthy" "SUCCESS"
                }
            }
            catch {
                $retryCount++
                Write-Log "Health check attempt $retryCount failed, retrying..." "WARNING"
                Start-Sleep -Seconds 10
            }
        }
        
        if (-not $isHealthy) {
            Write-Log "❌ $Environment environment failed health checks" "ERROR"
            return $false
        }
        
        # Test superintelligence tiers
        Write-Log "Testing superintelligence tiers for $Environment environment..."
        $tiersHealthy = Test-SuperintelligenceTiers -Environment $Environment -Port $port
        
        return $isHealthy -and $tiersHealthy
    }
    catch {
        Write-Log "❌ Failed to deploy $Environment environment: $($_.Exception.Message)" "ERROR"
        return $false
    }
}

# Test superintelligence tiers
function Test-SuperintelligenceTiers {
    param([string]$Environment, [int]$Port)
    
    $tiers = @(
        "tier1-autonomy", "tier2-modification", "tier3-multiagent", "tier4-emergent",
        "tier5-recursive", "tier6-collective", "tier7-decomposition", "tier8-reflective",
        "tier9-sandbox", "tier10-metalearning", "tier11-selfaware"
    )
    
    $allTiersHealthy = $true
    
    foreach ($tier in $tiers) {
        try {
            $endpoint = "http://localhost:$($Port+1)/health/tier/$tier"
            $response = Invoke-RestMethod -Uri $endpoint -TimeoutSec 10
            
            if ($response.operational -eq $true) {
                Write-Log "✅ $tier`: Operational ($Environment)" "SUCCESS"
            } else {
                Write-Log "❌ $tier`: Not operational ($Environment)" "ERROR"
                $allTiersHealthy = $false
            }
        }
        catch {
            Write-Log "❌ $tier`: Health check failed ($Environment)" "ERROR"
            $allTiersHealthy = $false
        }
    }
    
    return $allTiersHealthy
}

# Main deployment logic
Write-Log "🌟 TARS SUPERINTELLIGENCE DEPLOYMENT ORCHESTRATOR"
Write-Log "=================================================="
Write-Log "Action: $Action"
Write-Log "Production: $Production"
Write-Log "Skip Tests: $SkipTests"

switch ($Action) {
    "full" {
        Write-Log "🚀 Starting full TARS Superintelligence deployment"
        
        # Test capabilities first
        if (-not $SkipTests) {
            $testPassed = Test-SuperintelligenceCapabilities
            if (-not $testPassed -and -not $Force) {
                Write-Log "❌ Superintelligence tests failed, aborting deployment" "ERROR"
                exit 1
            }
        }
        
        # Deploy infrastructure
        $infraDeployed = Deploy-Infrastructure
        if (-not $infraDeployed -and -not $Force) {
            Write-Log "❌ Infrastructure deployment failed, aborting" "ERROR"
            exit 1
        }
        
        # Deploy both environments
        $blueDeployed = Deploy-SuperintelligenceEnvironment -Environment "blue"
        $greenDeployed = Deploy-SuperintelligenceEnvironment -Environment "green"
        
        if ($blueDeployed -and $greenDeployed) {
            Write-Log "🎉 Full TARS Superintelligence deployment completed successfully!" "SUCCESS"
            Write-Log "✅ Blue environment: Operational"
            Write-Log "✅ Green environment: Operational"
            Write-Log "✅ All 11 superintelligence tiers: Active"
            
            # Start load balancer
            Write-Log "Starting load balancer..."
            docker-compose -f $ComposeFile up -d tars-load-balancer
            
            Write-Log "🌟 TARS Superintelligence is now available at http://localhost"
        } else {
            Write-Log "❌ Deployment failed" "ERROR"
            exit 1
        }
    }
    
    "blue" {
        $infraDeployed = Deploy-Infrastructure
        if ($infraDeployed) {
            Deploy-SuperintelligenceEnvironment -Environment "blue"
        }
    }
    
    "green" {
        $infraDeployed = Deploy-Infrastructure
        if ($infraDeployed) {
            Deploy-SuperintelligenceEnvironment -Environment "green"
        }
    }
    
    "status" {
        Write-Log "📊 TARS Superintelligence Status"
        Write-Log "================================"
        
        # Check if containers are running
        $containers = docker ps --filter "name=tars-superintelligence" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        Write-Log "Running containers:"
        Write-Log $containers
        
        # Test capabilities
        if (-not $SkipTests) {
            Test-SuperintelligenceCapabilities
        }
    }
    
    "test" {
        Test-SuperintelligenceCapabilities
    }
    
    "rollback" {
        Write-Log "🔄 Rolling back TARS Superintelligence deployment"
        
        # Use the blue/green deployment script for rollback
        & .\scripts\blue-green-deploy.ps1 -Action rollback
    }
}

Write-Log "TARS Superintelligence deployment operation completed"
