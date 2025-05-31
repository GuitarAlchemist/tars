#!/bin/bash

# TARS Docker Container Entrypoint
# Provides stable container operation for swarm deployment

echo "🚀 TARS Container Starting..."
echo "Instance ID: ${TARS_INSTANCE_ID:-UNKNOWN}"
echo "Role: ${TARS_ROLE:-UNKNOWN}"
echo "Environment: ${TARS_ENVIRONMENT:-Docker}"

# Create necessary directories
mkdir -p /app/.tars/logs
mkdir -p /app/.tars/runtime

# Set proper permissions
chmod -R 755 /app/.tars/

# Check if TARS CLI is available
if [ -f "/app/TarsEngine.FSharp.Cli.dll" ]; then
    echo "✅ TARS CLI found"
    
    # Test TARS CLI
    echo "🔍 Testing TARS CLI..."
    dotnet /app/TarsEngine.FSharp.Cli.dll version 2>/dev/null || echo "⚠️  TARS CLI test failed"
else
    echo "❌ TARS CLI not found"
fi

# Check for metascripts
if [ -d "/app/.tars/metascripts" ]; then
    echo "📁 Metascripts directory found"
    ls -la /app/.tars/metascripts/ 2>/dev/null || echo "📁 Metascripts directory empty"
else
    echo "📁 Creating metascripts directory"
    mkdir -p /app/.tars/metascripts
fi

echo "🔄 Container ready - entering daemon mode..."
echo "💡 Use 'docker exec -it <container> /bin/bash' to interact"

# Keep container alive with a simple loop that can be interrupted
while true; do
    echo "$(date): TARS ${TARS_INSTANCE_ID:-UNKNOWN} running..."
    sleep 60
done
