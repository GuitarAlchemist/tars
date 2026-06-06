#!/usr/bin/env pwsh

# TARS GREEN BASELINE REDEPLOYMENT SYSTEM
# Critical: Redeploy tars-green-stable container under tars namespace
# Green nodes MUST be working 24/7 as stable baseline

param(
    [switch]$Force,
    [switch]$Rebuild,
    [string]$Tag = "latest"
)

$ErrorActionPreference = "Stop"

Write-Host "🚨 CRITICAL: TARS GREEN BASELINE REDEPLOYMENT" -ForegroundColor Red
Write-Host "=============================================" -ForegroundColor Red
Write-Host "🔒 Restoring stable green baseline container" -ForegroundColor Yellow
Write-Host ""

# Configuration
$ContainerName = "tars-green-stable"
$ImageName = "tars/green-stable"
$NetworkName = "tars-network"
$SessionId = Get-Date -Format "yyyyMMdd-HHmmss"

Write-Host "📋 Deployment Configuration:" -ForegroundColor Cyan
Write-Host "   Container: $ContainerName" -ForegroundColor Gray
Write-Host "   Image: $ImageName`:$Tag" -ForegroundColor Gray
Write-Host "   Network: $NetworkName" -ForegroundColor Gray
Write-Host "   Session: $SessionId" -ForegroundColor Gray
Write-Host ""

# Step 1: Stop and remove existing container
Write-Host "🛑 Step 1: Cleanup Existing Container" -ForegroundColor Magenta
Write-Host "=====================================" -ForegroundColor Magenta

try {
    $existingContainer = docker ps -a --filter "name=$ContainerName" --format "{{.Names}}" 2>$null
    if ($existingContainer -eq $ContainerName) {
        Write-Host "  🔄 Stopping existing container: $ContainerName" -ForegroundColor Yellow
        docker stop $ContainerName 2>$null | Out-Null
        
        Write-Host "  🗑️  Removing existing container: $ContainerName" -ForegroundColor Yellow
        docker rm $ContainerName 2>$null | Out-Null
        
        Write-Host "  ✅ Existing container cleaned up" -ForegroundColor Green
    } else {
        Write-Host "  ℹ️  No existing container found" -ForegroundColor Gray
    }
} catch {
    Write-Host "  ⚠️  Cleanup warning: $($_.Exception.Message)" -ForegroundColor Yellow
}

# Step 2: Create TARS network if needed
Write-Host ""
Write-Host "🌐 Step 2: Setup TARS Network" -ForegroundColor Magenta
Write-Host "=============================" -ForegroundColor Magenta

try {
    $existingNetwork = docker network ls --filter "name=$NetworkName" --format "{{.Name}}" 2>$null
    if ($existingNetwork -ne $NetworkName) {
        Write-Host "  🔧 Creating TARS network: $NetworkName" -ForegroundColor Yellow
        docker network create $NetworkName 2>$null | Out-Null
        Write-Host "  ✅ TARS network created" -ForegroundColor Green
    } else {
        Write-Host "  ✅ TARS network already exists" -ForegroundColor Green
    }
} catch {
    Write-Host "  ⚠️  Network setup warning: $($_.Exception.Message)" -ForegroundColor Yellow
}

# Step 3: Build/Pull TARS image
Write-Host ""
Write-Host "🏗️  Step 3: Prepare TARS Image" -ForegroundColor Magenta
Write-Host "==============================" -ForegroundColor Magenta

if ($Rebuild -or $Force) {
    Write-Host "  🔨 Building TARS green image..." -ForegroundColor Yellow
    
    # Create Dockerfile for green baseline
    $dockerfileContent = @"
# TARS Green Baseline Container
FROM mcr.microsoft.com/dotnet/aspnet:9.0 AS base
WORKDIR /app
EXPOSE 8080
EXPOSE 8081

# Copy TARS CLI
FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
WORKDIR /src
COPY ["TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj", "TarsEngine.FSharp.Cli/"]
RUN dotnet restore "TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj"
COPY . .
WORKDIR "/src/TarsEngine.FSharp.Cli"
RUN dotnet build "TarsEngine.FSharp.Cli.fsproj" -c Release -o /app/build
RUN dotnet publish "TarsEngine.FSharp.Cli.fsproj" -c Release -o /app/publish

# Runtime image
FROM base AS final
WORKDIR /app
COPY --from=build /app/publish .

# Create TARS directories
RUN mkdir -p /app/.tars/green /app/.tars/shared /app/.tars/logs

# Set environment variables
ENV TARS_ENVIRONMENT=green
ENV TARS_ROLE=baseline
ENV TARS_CONTAINER_NAME=$ContainerName
ENV TARS_SESSION_ID=$SessionId
ENV DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1
ENV ASPNETCORE_URLS=http://+:8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Entry point
ENTRYPOINT ["dotnet", "TarsEngine.FSharp.Cli.dll", "serve", "--port", "8080"]
"@
    
    $dockerfileContent | Out-File -FilePath "Dockerfile.green" -Encoding UTF8
    
    Write-Host "  📝 Dockerfile.green created" -ForegroundColor Gray
    Write-Host "  🔨 Building image: $ImageName`:$Tag" -ForegroundColor Yellow
    
    docker build -f Dockerfile.green -t "$ImageName`:$Tag" . 2>&1 | ForEach-Object {
        if ($_ -match "ERROR|error") {
            Write-Host "    ❌ $_" -ForegroundColor Red
        } elseif ($_ -match "Successfully") {
            Write-Host "    ✅ $_" -ForegroundColor Green
        } else {
            Write-Host "    📦 $_" -ForegroundColor Gray
        }
    }
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ TARS green image built successfully" -ForegroundColor Green
    } else {
        Write-Host "  ❌ Image build failed!" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "  📦 Using existing image: $ImageName`:$Tag" -ForegroundColor Gray
}

# Step 4: Deploy green container
Write-Host ""
Write-Host "🚀 Step 4: Deploy TARS Green Container" -ForegroundColor Magenta
Write-Host "======================================" -ForegroundColor Magenta

Write-Host "  🚀 Starting container: $ContainerName" -ForegroundColor Yellow

$dockerArgs = @(
    "run", "-d"
    "--name", $ContainerName
    "--network", $NetworkName
    "--label", "tars.environment=green"
    "--label", "tars.role=baseline"
    "--label", "tars.namespace=tars"
    "--label", "tars.session=$SessionId"
    "--restart", "always"
    "-p", "8080:8080"
    "-p", "8081:8081"
    "-v", "$(Get-Location)/.tars/green:/app/.tars/green:rw"
    "-v", "$(Get-Location)/.tars/shared:/app/.tars/shared:ro"
    "-e", "TARS_ENVIRONMENT=green"
    "-e", "TARS_ROLE=baseline"
    "-e", "TARS_MONITORING_ENABLED=true"
    "-e", "TARS_SESSION_ID=$SessionId"
    "$ImageName`:$Tag"
)

try {
    $containerId = docker @dockerArgs 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ Container started: $($containerId.Substring(0,12))" -ForegroundColor Green
    } else {
        Write-Host "  ❌ Container start failed: $containerId" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "  ❌ Deployment failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Step 5: Health verification
Write-Host ""
Write-Host "🏥 Step 5: Health Verification" -ForegroundColor Magenta
Write-Host "==============================" -ForegroundColor Magenta

Write-Host "  🔍 Waiting for container to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

for ($i = 1; $i -le 12; $i++) {
    try {
        $containerStatus = docker inspect $ContainerName --format "{{.State.Status}}" 2>$null
        if ($containerStatus -eq "running") {
            Write-Host "  ✅ Container is running (attempt $i/12)" -ForegroundColor Green
            break
        } else {
            Write-Host "  ⏳ Container status: $containerStatus (attempt $i/12)" -ForegroundColor Yellow
            Start-Sleep -Seconds 5
        }
    } catch {
        Write-Host "  ⚠️  Health check failed (attempt $i/12): $($_.Exception.Message)" -ForegroundColor Yellow
        Start-Sleep -Seconds 5
    }
    
    if ($i -eq 12) {
        Write-Host "  ❌ Container failed to start properly!" -ForegroundColor Red
        Write-Host "  📋 Container logs:" -ForegroundColor Yellow
        docker logs $ContainerName --tail 20
        exit 1
    }
}

# Step 6: Final verification
Write-Host ""
Write-Host "✅ Step 6: Final System Verification" -ForegroundColor Magenta
Write-Host "====================================" -ForegroundColor Magenta

Write-Host "  📊 Container Information:" -ForegroundColor Cyan
docker ps --filter "name=$ContainerName" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | ForEach-Object {
    Write-Host "    $_" -ForegroundColor Gray
}

Write-Host ""
Write-Host "  🌐 Network Information:" -ForegroundColor Cyan
docker network inspect $NetworkName --format "{{.Name}}: {{len .Containers}} containers" | ForEach-Object {
    Write-Host "    $_" -ForegroundColor Gray
}

Write-Host ""
Write-Host "🎉 TARS GREEN BASELINE REDEPLOYMENT COMPLETE!" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host "✅ Container: $ContainerName is running" -ForegroundColor Green
Write-Host "✅ Image: $ImageName`:$Tag deployed" -ForegroundColor Green
Write-Host "✅ Network: $NetworkName configured" -ForegroundColor Green
Write-Host "✅ Ports: 8080, 8081 exposed" -ForegroundColor Green
Write-Host "✅ Auto-restart: enabled" -ForegroundColor Green
Write-Host "✅ Health monitoring: active" -ForegroundColor Green
Write-Host ""
Write-Host "🔒 STABLE GREEN BASELINE RESTORED!" -ForegroundColor Green
Write-Host "🎯 Ready for blue node operations" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Management Commands:" -ForegroundColor Cyan
Write-Host "   View logs: docker logs $ContainerName -f" -ForegroundColor Gray
Write-Host "   Stop: docker stop $ContainerName" -ForegroundColor Gray
Write-Host "   Restart: docker restart $ContainerName" -ForegroundColor Gray
Write-Host "   Health: docker inspect $ContainerName --format '{{.State.Health.Status}}'" -ForegroundColor Gray
