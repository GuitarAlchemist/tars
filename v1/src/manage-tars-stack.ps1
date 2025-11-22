# TARS Unified Stack Management Script
# Simplified management for the consolidated TARS infrastructure

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("start", "stop", "restart", "status", "logs", "health", "clean", "backup")]
    [string]$Action,
    
    [string]$Service = "",
    [switch]$Follow = $false,
    [switch]$Force = $false
)

$ComposeFile = "docker-compose.unified.yml"

Write-Host "🎯 TARS Unified Stack Manager" -ForegroundColor Cyan
Write-Host "=============================" -ForegroundColor Cyan

# Check if compose file exists
if (-not (Test-Path $ComposeFile)) {
    Write-Error "Compose file not found: $ComposeFile"
    Write-Host "💡 Run the consolidation script first: .\scripts\consolidate-tars-stack.ps1" -ForegroundColor Yellow
    exit 1
}

switch ($Action) {
    "start" {
        Write-Host "🚀 Starting TARS unified stack..." -ForegroundColor Green
        if ($Service) {
            docker-compose -f $ComposeFile up -d $Service
            Write-Host "✅ Service '$Service' started" -ForegroundColor Green
        }
        else {
            docker-compose -f $ComposeFile up -d
            Write-Host "✅ All TARS services started" -ForegroundColor Green
        }
    }
    
    "stop" {
        Write-Host "🛑 Stopping TARS unified stack..." -ForegroundColor Yellow
        if ($Service) {
            docker-compose -f $ComposeFile stop $Service
            Write-Host "✅ Service '$Service' stopped" -ForegroundColor Green
        }
        else {
            docker-compose -f $ComposeFile stop
            Write-Host "✅ All TARS services stopped" -ForegroundColor Green
        }
    }
    
    "restart" {
        Write-Host "🔄 Restarting TARS unified stack..." -ForegroundColor Blue
        if ($Service) {
            docker-compose -f $ComposeFile restart $Service
            Write-Host "✅ Service '$Service' restarted" -ForegroundColor Green
        }
        else {
            docker-compose -f $ComposeFile restart
            Write-Host "✅ All TARS services restarted" -ForegroundColor Green
        }
    }
    
    "status" {
        Write-Host "📊 TARS Stack Status:" -ForegroundColor Blue
        docker-compose -f $ComposeFile ps
    }
    
    "logs" {
        Write-Host "📋 TARS Stack Logs:" -ForegroundColor Blue
        if ($Service) {
            if ($Follow) {
                docker-compose -f $ComposeFile logs -f $Service
            }
            else {
                docker-compose -f $ComposeFile logs --tail=50 $Service
            }
        }
        else {
            if ($Follow) {
                docker-compose -f $ComposeFile logs -f
            }
            else {
                docker-compose -f $ComposeFile logs --tail=50
            }
        }
    }
    
    "health" {
        Write-Host "🏥 TARS Health Check:" -ForegroundColor Blue
        
        $services = @(
            @{Name="TARS Main"; Url="http://localhost:8080/health"},
            @{Name="TARS Autonomous"; Url="http://localhost:8088/health"},
            @{Name="MongoDB"; Port=27017},
            @{Name="ChromaDB"; Url="http://localhost:8000/api/v1/heartbeat"},
            @{Name="Redis"; Port=6379},
            @{Name="Fuseki"; Url="http://localhost:3030/$/ping"},
            @{Name="Virtuoso"; Port=8890},
            @{Name="NGINX"; Url="http://localhost/health"}
        )
        
        foreach ($service in $services) {
            try {
                if ($service.Url) {
                    $response = Invoke-WebRequest -Uri $service.Url -TimeoutSec 5 -ErrorAction Stop
                    Write-Host "  ✅ $($service.Name) - Healthy (HTTP $($response.StatusCode))" -ForegroundColor Green
                }
                elseif ($service.Port) {
                    $connection = Test-NetConnection -ComputerName localhost -Port $service.Port -WarningAction SilentlyContinue
                    if ($connection.TcpTestSucceeded) {
                        Write-Host "  ✅ $($service.Name) - Port $($service.Port) accessible" -ForegroundColor Green
                    }
                    else {
                        Write-Host "  ❌ $($service.Name) - Port $($service.Port) not accessible" -ForegroundColor Red
                    }
                }
            }
            catch {
                Write-Host "  ❌ $($service.Name) - Health check failed: $($_.Exception.Message)" -ForegroundColor Red
            }
        }
    }
    
    "clean" {
        if (-not $Force) {
            $confirm = Read-Host "⚠️  This will remove all containers and volumes. Are you sure? (y/N)"
            if ($confirm -ne "y" -and $confirm -ne "Y") {
                Write-Host "❌ Clean operation cancelled" -ForegroundColor Red
                exit 0
            }
        }
        
        Write-Host "🧹 Cleaning TARS stack..." -ForegroundColor Yellow
        docker-compose -f $ComposeFile down -v --remove-orphans
        Write-Host "✅ TARS stack cleaned" -ForegroundColor Green
    }
    
    "backup" {
        Write-Host "📦 Creating TARS data backup..." -ForegroundColor Blue
        
        $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
        $backupDir = "backups/unified-stack-$timestamp"
        
        if (-not (Test-Path "backups")) {
            New-Item -ItemType Directory -Path "backups" -Force | Out-Null
        }
        
        New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
        
        # Backup volumes
        $volumes = @("mongodb_data", "chromadb_data", "redis_data", "fuseki_data", "virtuoso_data", "tars_data")
        
        foreach ($volume in $volumes) {
            try {
                Write-Host "  Backing up $volume..." -ForegroundColor Gray
                docker run --rm -v "${volume}:/data" -v "${PWD}/${backupDir}:/backup" alpine tar czf "/backup/${volume}.tar.gz" -C /data .
                Write-Host "  ✅ $volume backed up" -ForegroundColor Green
            }
            catch {
                Write-Host "  ⚠️  Failed to backup $volume" -ForegroundColor Yellow
            }
        }
        
        Write-Host "✅ Backup completed: $backupDir" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "📋 Quick Access URLs:" -ForegroundColor Cyan
Write-Host "  TARS Main: http://localhost" -ForegroundColor Gray
Write-Host "  TARS Autonomous: http://localhost:8088" -ForegroundColor Gray
Write-Host "  MongoDB Admin: http://localhost:8081" -ForegroundColor Gray
Write-Host "  Redis Admin: http://localhost:8082" -ForegroundColor Gray
Write-Host "  ChromaDB: http://localhost:8000" -ForegroundColor Gray
Write-Host "  Fuseki: http://localhost:3030" -ForegroundColor Gray
