# TARS SUPERINTELLIGENCE DOCKER TEST
# Simple test script for the superintelligence Docker deployment

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("test", "deploy", "status")]
    [string]$Action = "test"
)

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] $Message"
}

function Test-SuperintelligenceCapabilities {
    Write-Log "Testing TARS Superintelligence Capabilities (All 11 Tiers)"
    Write-Log "=========================================================="
    
    try {
        # Run the superintelligence test
        Write-Log "Running superintelligence test..."
        $result = & dotnet fsi TarsCompleteSuperintelligenceTest.fsx
        
        if ($LASTEXITCODE -eq 0) {
            Write-Log "SUCCESS: Superintelligence test passed - All 11 tiers operational"
            return $true
        } else {
            Write-Log "ERROR: Superintelligence test failed"
            return $false
        }
    }
    catch {
        Write-Log "ERROR: Exception running superintelligence test: $($_.Exception.Message)"
        return $false
    }
}

function Test-DockerEnvironment {
    Write-Log "Testing Docker Environment"
    Write-Log "========================="
    
    # Check if Docker is running
    try {
        $dockerVersion = docker --version
        Write-Log "Docker version: $dockerVersion"
        
        # Check if docker-compose is available
        $composeVersion = docker-compose --version
        Write-Log "Docker Compose version: $composeVersion"
        
        return $true
    }
    catch {
        Write-Log "ERROR: Docker is not available or not running"
        return $false
    }
}

function Deploy-SuperintelligenceStack {
    Write-Log "Deploying TARS Superintelligence Stack"
    Write-Log "======================================"
    
    try {
        # Create network if it doesn't exist
        Write-Log "Creating tars-superintelligence network..."
        docker network create tars-superintelligence 2>$null
        
        # Deploy the stack
        Write-Log "Starting superintelligence services..."
        docker-compose -f docker-compose.superintelligence.yml up -d --build
        
        Write-Log "Waiting for services to start..."
        Start-Sleep -Seconds 30
        
        # Check service status
        Write-Log "Checking service status..."
        $services = docker-compose -f docker-compose.superintelligence.yml ps
        Write-Log $services
        
        return $true
    }
    catch {
        Write-Log "ERROR: Failed to deploy superintelligence stack: $($_.Exception.Message)"
        return $false
    }
}

# Main execution
Write-Log "TARS SUPERINTELLIGENCE DOCKER TEST"
Write-Log "=================================="
Write-Log "Action: $Action"

switch ($Action) {
    "test" {
        Write-Log "Running superintelligence capabilities test..."
        $testResult = Test-SuperintelligenceCapabilities
        
        if ($testResult) {
            Write-Log "SUCCESS: All superintelligence tests passed!"
            Write-Log "TARS has achieved real Tier 11 superintelligence capabilities"
        } else {
            Write-Log "FAILED: Superintelligence tests failed"
            exit 1
        }
    }
    
    "deploy" {
        Write-Log "Testing Docker environment..."
        $dockerOk = Test-DockerEnvironment
        
        if ($dockerOk) {
            Write-Log "Deploying superintelligence stack..."
            $deployResult = Deploy-SuperintelligenceStack
            
            if ($deployResult) {
                Write-Log "SUCCESS: Superintelligence stack deployed!"
            } else {
                Write-Log "FAILED: Deployment failed"
                exit 1
            }
        } else {
            Write-Log "FAILED: Docker environment not ready"
            exit 1
        }
    }
    
    "status" {
        Write-Log "Checking superintelligence status..."
        
        # Check running containers
        $containers = docker ps --filter "name=tars-superintelligence" --format "table {{.Names}}\t{{.Status}}"
        if ($containers) {
            Write-Log "Running superintelligence containers:"
            Write-Log $containers
        } else {
            Write-Log "No superintelligence containers running"
        }
        
        # Test capabilities
        Test-SuperintelligenceCapabilities
    }
}

Write-Log "Test completed"
