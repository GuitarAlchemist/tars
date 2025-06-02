#!/bin/bash 
# Green Environment - Stable Baseline 
echo "🟢 Starting TARS Green Environment (Baseline)" 
 
# Check if container already exists 
if docker ps -a --format "table {{.Names}}" | grep -q "^tars-green-stable$"; then 
    echo "  🔄 Stopping existing green container..." 
    docker stop tars-green-stable 2>/dev/null || true 
    docker rm tars-green-stable 2>/dev/null || true 
fi 
 
# Create network if it doesn't exist 
docker network create tars-evolution 2>/dev/null || true 
 
# Start green container 
echo "  🚀 Starting green container..." 
docker run -d --name tars-green-stable \ 
  --network tars-evolution \ 
  --label tars.environment=green \ 
  --label tars.role=baseline \ 
  --label tars.evolver.session=34744403 \ 
  -p 8080:8080 \ 
  -p 8081:8081 \ 
  -v "$(pwd)/.tars/green:/app/tars:rw" \ 
  -v "$(pwd)/.tars/shared:/app/shared:ro" \ 
  -e TARS_ENVIRONMENT=green \ 
  -e TARS_ROLE=baseline \ 
  -e TARS_MONITORING_ENABLED=true \ 
  -e TARS_SESSION_ID=34744403 \ 
  mcr.microsoft.com/dotnet/aspnet:9.0 
 
echo "  ✅ Green environment ready at http://localhost:8080" 
echo "  📊 Metrics available at http://localhost:8081/metrics" 
