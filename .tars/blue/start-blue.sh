#!/bin/bash 
# Blue Environment - Evolution Experimental 
echo "🔵 Starting TARS Blue Environment (Evolution)" 
 
# Check if container already exists 
if docker ps -a --format "table {{.Names}}" | grep -q "^tars-blue-evolution$"; then 
    echo "  🔄 Stopping existing blue container..." 
    docker stop tars-blue-evolution 2>/dev/null || true 
    docker rm tars-blue-evolution 2>/dev/null || true 
fi 
 
# Create network if it doesn't exist 
docker network create tars-evolution 2>/dev/null || true 
 
# Start blue container 
echo "  🚀 Starting blue container..." 
docker run -d --name tars-blue-evolution \ 
  --network tars-evolution \ 
  --label tars.environment=blue \ 
  --label tars.role=evolution \ 
  --label tars.evolver.session=34744403 \ 
  -p 8082:8080 \ 
  -p 8083:8081 \ 
  -v "$(pwd)/.tars/blue:/app/tars:rw" \ 
  -v "$(pwd)/.tars/shared:/app/shared:ro" \ 
  -v "$(pwd)/.tars/evolution:/app/evolution:rw" \ 
  -e TARS_ENVIRONMENT=blue \ 
  -e TARS_ROLE=evolution \ 
  -e TARS_EVOLUTION_ENABLED=true \ 
  -e TARS_MONITORING_ENABLED=true \ 
  -e TARS_SESSION_ID=34744403 \ 
  mcr.microsoft.com/dotnet/aspnet:9.0 
 
echo "  ✅ Blue environment ready at http://localhost:8082" 
echo "  📊 Metrics available at http://localhost:8083/metrics" 
