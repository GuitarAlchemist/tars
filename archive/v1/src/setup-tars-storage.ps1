# TARS Optimal Storage Architecture Setup
# Replaces disk-first approach with database-first architecture

param(
    [switch]$Start,
    [switch]$Stop,
    [switch]$Reset,
    [switch]$Status,
    [switch]$Migrate,
    [switch]$Test
)

Write-Host "🎯 TARS Optimal Storage Architecture Manager" -ForegroundColor Blue
Write-Host "=============================================" -ForegroundColor Blue

function Show-Help {
    Write-Host ""
    Write-Host "Usage:" -ForegroundColor Yellow
    Write-Host "  .\setup-tars-storage.ps1 -Start     # Start all storage services"
    Write-Host "  .\setup-tars-storage.ps1 -Stop      # Stop all storage services"
    Write-Host "  .\setup-tars-storage.ps1 -Reset     # Reset and reinitialize"
    Write-Host "  .\setup-tars-storage.ps1 -Status    # Check service status"
    Write-Host "  .\setup-tars-storage.ps1 -Migrate   # Migrate from disk to databases"
    Write-Host "  .\setup-tars-storage.ps1 -Test      # Test all connections"
    Write-Host ""
}

function Start-TarsStorage {
    Write-Host "🚀 Starting TARS Optimal Storage Architecture..." -ForegroundColor Green
    
    # Create network if it doesn't exist
    Write-Host "📡 Creating TARS network..." -ForegroundColor Cyan
    docker network create tars-network 2>$null
    
    # Start storage services
    Write-Host "💾 Starting storage services..." -ForegroundColor Cyan
    docker-compose -f docker-compose.storage.yml up -d
    
    # Wait for services to be healthy
    Write-Host "⏳ Waiting for services to be healthy..." -ForegroundColor Yellow
    Start-Sleep -Seconds 30
    
    # Check service health
    Test-TarsStorage
}

function Stop-TarsStorage {
    Write-Host "🛑 Stopping TARS Storage Architecture..." -ForegroundColor Red
    docker-compose -f docker-compose.storage.yml down
    Write-Host "✅ All storage services stopped." -ForegroundColor Green
}

function Reset-TarsStorage {
    Write-Host "🔄 Resetting TARS Storage Architecture..." -ForegroundColor Yellow
    
    # Stop and remove everything
    docker-compose -f docker-compose.storage.yml down -v
    
    # Remove old containers if they exist
    Write-Host "🧹 Cleaning up old containers..." -ForegroundColor Cyan
    docker rm -f tars-virtuoso tars-fuseki tars-chromadb 2>$null
    
    # Start fresh
    Start-TarsStorage
}

function Get-TarsStorageStatus {
    Write-Host "📊 TARS Storage Architecture Status" -ForegroundColor Cyan
    Write-Host "===================================" -ForegroundColor Cyan
    
    $services = @(
        @{Name="MongoDB"; Container="tars-mongodb"; Port="27017"},
        @{Name="ChromaDB"; Container="tars-chromadb"; Port="8000"},
        @{Name="Fuseki RDF"; Container="tars-fuseki"; Port="3030"},
        @{Name="Redis"; Container="tars-redis"; Port="6379"},
        @{Name="Elasticsearch"; Container="tars-elasticsearch"; Port="9200"},
        @{Name="PostgreSQL"; Container="tars-postgres"; Port="5432"}
    )
    
    foreach ($service in $services) {
        $status = docker ps --filter "name=$($service.Container)" --format "{{.Status}}" 2>$null
        if ($status) {
            Write-Host "✅ $($service.Name): " -NoNewline -ForegroundColor Green
            Write-Host "Running ($status)" -ForegroundColor White
        } else {
            Write-Host "❌ $($service.Name): " -NoNewline -ForegroundColor Red
            Write-Host "Not running" -ForegroundColor White
        }
    }
    
    Write-Host ""
    Write-Host "🌐 Management UIs:" -ForegroundColor Yellow
    Write-Host "  MongoDB Express: http://localhost:8081 (tars/tars_ui_2024)"
    Write-Host "  Redis Commander: http://localhost:8082 (tars/tars_ui_2024)"
    Write-Host "  pgAdmin:         http://localhost:8083 (admin@tars.local/tars_pgadmin_2024)"
    Write-Host "  Fuseki:          http://localhost:3030"
    Write-Host ""
}

function Test-TarsStorage {
    Write-Host "🧪 Testing TARS Storage Connections..." -ForegroundColor Cyan
    
    $tests = @()
    
    # Test MongoDB
    try {
        $mongoResult = docker exec tars-mongodb mongosh --eval "db.adminCommand('ping')" --quiet 2>$null
        if ($mongoResult -match "ok.*1") {
            $tests += @{Service="MongoDB"; Status="✅ Connected"; Details="Ping successful"}
        } else {
            $tests += @{Service="MongoDB"; Status="❌ Failed"; Details="Ping failed"}
        }
    } catch {
        $tests += @{Service="MongoDB"; Status="❌ Error"; Details=$_.Exception.Message}
    }
    
    # Test ChromaDB
    try {
        $chromaResult = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/heartbeat" -TimeoutSec 5 2>$null
        $tests += @{Service="ChromaDB"; Status="✅ Connected"; Details="Heartbeat successful"}
    } catch {
        $tests += @{Service="ChromaDB"; Status="❌ Failed"; Details="Heartbeat failed"}
    }
    
    # Test Redis
    try {
        $redisResult = docker exec tars-redis redis-cli --no-auth-warning -a tars_redis_2024 ping 2>$null
        if ($redisResult -match "PONG") {
            $tests += @{Service="Redis"; Status="✅ Connected"; Details="Ping successful"}
        } else {
            $tests += @{Service="Redis"; Status="❌ Failed"; Details="Ping failed"}
        }
    } catch {
        $tests += @{Service="Redis"; Status="❌ Error"; Details=$_.Exception.Message}
    }
    
    # Test Fuseki
    try {
        $fusekiResult = Invoke-RestMethod -Uri "http://localhost:3030/$/ping" -TimeoutSec 5 2>$null
        $tests += @{Service="Fuseki"; Status="✅ Connected"; Details="Ping successful"}
    } catch {
        $tests += @{Service="Fuseki"; Status="❌ Failed"; Details="Ping failed"}
    }
    
    # Test Elasticsearch
    try {
        $elasticResult = Invoke-RestMethod -Uri "http://localhost:9200/_cluster/health" -TimeoutSec 5 2>$null
        if ($elasticResult.status -eq "green" -or $elasticResult.status -eq "yellow") {
            $tests += @{Service="Elasticsearch"; Status="✅ Connected"; Details="Cluster healthy"}
        } else {
            $tests += @{Service="Elasticsearch"; Status="❌ Failed"; Details="Cluster unhealthy"}
        }
    } catch {
        $tests += @{Service="Elasticsearch"; Status="❌ Failed"; Details="Connection failed"}
    }
    
    # Test PostgreSQL
    try {
        $pgResult = docker exec tars-postgres pg_isready -U tars_audit -d tars_audit 2>$null
        if ($pgResult -match "accepting connections") {
            $tests += @{Service="PostgreSQL"; Status="✅ Connected"; Details="Ready for connections"}
        } else {
            $tests += @{Service="PostgreSQL"; Status="❌ Failed"; Details="Not ready"}
        }
    } catch {
        $tests += @{Service="PostgreSQL"; Status="❌ Error"; Details=$_.Exception.Message}
    }
    
    # Display results
    Write-Host ""
    Write-Host "📊 Connection Test Results:" -ForegroundColor Yellow
    foreach ($test in $tests) {
        Write-Host "  $($test.Status) $($test.Service): $($test.Details)"
    }
    Write-Host ""
}

function Start-Migration {
    Write-Host "🔄 Migrating TARS from Disk-First to Database-First..." -ForegroundColor Yellow
    
    # Check if disk cache exists
    $diskCache = ".tars/knowledge_cache/memory_cache.json"
    if (Test-Path $diskCache) {
        Write-Host "📁 Found existing disk cache: $diskCache" -ForegroundColor Cyan
        
        # This would integrate with TARS CLI to migrate data
        Write-Host "🚀 Migration would involve:" -ForegroundColor Green
        Write-Host "  1. Reading existing JSON knowledge cache"
        Write-Host "  2. Parsing and validating knowledge entries"
        Write-Host "  3. Inserting into MongoDB with proper schema"
        Write-Host "  4. Generating embeddings for ChromaDB"
        Write-Host "  5. Creating RDF triples for Fuseki"
        Write-Host "  6. Updating TARS configuration to use databases"
        Write-Host ""
        Write-Host "💡 Run: dotnet run --project TarsEngine.FSharp.Cli -- migrate-storage" -ForegroundColor Yellow
    } else {
        Write-Host "ℹ️ No existing disk cache found. Starting fresh with database-first approach." -ForegroundColor Blue
    }
}

# Main execution
if ($Start) {
    Start-TarsStorage
} elseif ($Stop) {
    Stop-TarsStorage
} elseif ($Reset) {
    Reset-TarsStorage
} elseif ($Status) {
    Get-TarsStorageStatus
} elseif ($Migrate) {
    Start-Migration
} elseif ($Test) {
    Test-TarsStorage
} else {
    Show-Help
}

Write-Host ""
Write-Host "🎯 TARS Storage Architecture: Database-First Approach" -ForegroundColor Blue
Write-Host "   Primary: MongoDB (documents) + ChromaDB (vectors) + Fuseki (RDF)" -ForegroundColor White
Write-Host "   Caching: Redis (fast access) + Disk (offline fallback)" -ForegroundColor White
Write-Host "   Analytics: Elasticsearch (search) + PostgreSQL (audit)" -ForegroundColor White
