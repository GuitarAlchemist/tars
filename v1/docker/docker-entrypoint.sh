#!/bin/bash
# TARS Unified Architecture - Docker Entrypoint Script
# Initializes the unified system with proper configuration and health checks

set -e

echo "🚀 Starting TARS Unified Architecture..."
echo "   Version: Unified Architecture v2.0"
echo "   Environment: ${TARS_ENVIRONMENT:-Docker}"
echo "   Mode: ${TARS_MODE:-Unified}"

# Function to wait for service
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local max_attempts=30
    local attempt=1
    
    echo "⏳ Waiting for $service_name ($host:$port)..."
    
    while [ $attempt -le $max_attempts ]; do
        if nc -z "$host" "$port" 2>/dev/null; then
            echo "✅ $service_name is ready!"
            return 0
        fi
        
        echo "   Attempt $attempt/$max_attempts - $service_name not ready yet..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo "❌ $service_name failed to become ready after $max_attempts attempts"
    return 1
}

# Function to check CUDA availability
check_cuda() {
    if [ "${TARS_CUDA_ENABLED:-true}" = "true" ]; then
        echo "🔍 Checking CUDA availability..."
        
        if command -v nvidia-smi >/dev/null 2>&1; then
            echo "✅ NVIDIA drivers detected"
            nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null || echo "⚠️  GPU info unavailable"
        else
            echo "⚠️  NVIDIA drivers not found - will use CPU fallback"
        fi
    else
        echo "ℹ️  CUDA disabled by configuration"
    fi
}

# Function to initialize directories
init_directories() {
    echo "📁 Initializing TARS directories..."
    
    # Create required directories
    mkdir -p "${TARS_DATA_PATH:-/app/data}"
    mkdir -p "${TARS_CACHE_PATH:-/app/data/cache}"
    mkdir -p "${TARS_LOGS_PATH:-/app/data/logs}"
    mkdir -p "${TARS_CONFIG_PATH:-/app/data/config}"
    mkdir -p "${TARS_PROOFS_PATH:-/app/data/proofs}"
    mkdir -p "/app/data/monitoring"
    
    # Set permissions
    chmod 755 "${TARS_DATA_PATH:-/app/data}"
    chmod 755 "${TARS_CACHE_PATH:-/app/data/cache}"
    chmod 755 "${TARS_LOGS_PATH:-/app/data/logs}"
    chmod 755 "${TARS_CONFIG_PATH:-/app/data/config}"
    chmod 755 "${TARS_PROOFS_PATH:-/app/data/proofs}"
    chmod 755 "/app/data/monitoring"
    
    echo "✅ Directories initialized"
}

# Function to copy default configuration
init_configuration() {
    echo "⚙️  Initializing configuration..."
    
    local config_file="${TARS_CONFIG_PATH:-/app/data/config}/tars.config.json"
    
    if [ ! -f "$config_file" ]; then
        if [ -f "/app/data/config/tars.config.json" ]; then
            echo "📋 Using provided configuration"
        else
            echo "📋 Creating default configuration"
            cat > "$config_file" << 'EOF'
{
  "tars": {
    "core": {
      "logLevel": "Information",
      "enableProofGeneration": true
    },
    "cache": {
      "enabled": true,
      "diskPath": "/app/data/cache"
    },
    "monitoring": {
      "enabled": true
    }
  }
}
EOF
        fi
    fi
    
    echo "✅ Configuration ready: $config_file"
}

# Function to run system diagnostics
run_diagnostics() {
    echo "🔍 Running system diagnostics..."
    
    # Basic system info
    echo "   OS: $(uname -s) $(uname -r)"
    echo "   Architecture: $(uname -m)"
    echo "   Memory: $(free -h | grep '^Mem:' | awk '{print $2}') total"
    echo "   Disk: $(df -h /app | tail -1 | awk '{print $4}') available"
    
    # .NET info
    if command -v dotnet >/dev/null 2>&1; then
        echo "   .NET: $(dotnet --version)"
    fi
    
    # Check if TARS can start
    echo "🧪 Testing TARS startup..."
    if timeout 30 dotnet TarsEngine.FSharp.Cli.dll --version >/dev/null 2>&1; then
        echo "✅ TARS startup test passed"
    else
        echo "⚠️  TARS startup test failed or timed out"
    fi
}

# Main initialization
main() {
    echo "🔧 Initializing TARS Unified Architecture..."
    
    # Initialize directories
    init_directories
    
    # Initialize configuration
    init_configuration
    
    # Check CUDA
    check_cuda
    
    # Wait for dependencies if specified
    if [ -n "${REDIS_HOST:-}" ]; then
        wait_for_service "${REDIS_HOST}" "${REDIS_PORT:-6379}" "Redis"
    fi
    
    if [ -n "${POSTGRES_HOST:-}" ]; then
        wait_for_service "${POSTGRES_HOST}" "${POSTGRES_PORT:-5432}" "PostgreSQL"
    fi
    
    # Run diagnostics
    run_diagnostics
    
    echo "🎉 TARS Unified Architecture initialization complete!"
    echo ""
    echo "🚀 Starting TARS with unified systems:"
    echo "   ✅ Unified Core Foundation"
    echo "   ✅ Unified Configuration Management"
    echo "   ✅ Unified Proof Generation"
    echo "   ✅ Unified Caching System"
    echo "   ✅ Unified Monitoring System"
    echo "   ✅ Unified CUDA Engine"
    echo "   ✅ Unified Agent Coordination"
    echo ""
    
    # Execute the main command
    exec "$@"
}

# Handle signals gracefully
trap 'echo "🛑 Received shutdown signal, stopping TARS..."; exit 0' SIGTERM SIGINT

# Run main function
main "$@"
