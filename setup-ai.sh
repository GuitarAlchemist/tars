#!/bin/bash
# TARS AI Setup Script - Install and configure Ollama for local LLM integration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
OLLAMA_VERSION="latest"
DEFAULT_MODEL="llama3.2:3b"
RECOMMENDED_MODELS=("llama3.2:3b" "codellama:7b" "mistral:7b" "phi3:mini")

# Functions
print_header() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    TARS AI Setup Script                     ║"
    echo "║              Local LLM Integration with Ollama              ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_step() {
    echo -e "${BLUE}🔧 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check system requirements
check_requirements() {
    print_step "Checking system requirements..."
    
    # Check OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        print_success "Linux detected"
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        print_success "macOS detected"
        OS="macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        print_success "Windows detected"
        OS="windows"
    else
        print_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    
    # Check available memory
    if command -v free &> /dev/null; then
        local mem_gb=$(free -g | awk '/^Mem:/{print $2}')
        if [ "$mem_gb" -lt 8 ]; then
            print_warning "Low memory detected: ${mem_gb}GB (8GB+ recommended for LLM)"
        else
            print_success "Sufficient memory: ${mem_gb}GB"
        fi
    fi
    
    # Check disk space
    local available_space=$(df . | tail -1 | awk '{print $4}')
    if [ "$available_space" -lt 10000000 ]; then # 10GB in KB
        print_warning "Low disk space: $(df -h . | tail -1 | awk '{print $4}') (10GB+ recommended)"
    else
        print_success "Sufficient disk space: $(df -h . | tail -1 | awk '{print $4}')"
    fi
    
    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        local gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "")
        if [ -n "$gpu_info" ]; then
            print_success "NVIDIA GPU detected: $gpu_info"
            GPU_AVAILABLE=true
        else
            print_warning "NVIDIA GPU not available - will use CPU"
            GPU_AVAILABLE=false
        fi
    else
        print_warning "NVIDIA drivers not found - will use CPU"
        GPU_AVAILABLE=false
    fi
}

# Install Ollama
install_ollama() {
    print_step "Installing Ollama..."
    
    if command -v ollama &> /dev/null; then
        print_success "Ollama already installed: $(ollama --version)"
        return 0
    fi
    
    case $OS in
        "linux"|"macos")
            print_step "Downloading and installing Ollama..."
            curl -fsSL https://ollama.ai/install.sh | sh
            ;;
        "windows")
            print_warning "Please download and install Ollama manually from: https://ollama.ai/download"
            print_warning "Then run this script again"
            exit 1
            ;;
        *)
            print_error "Unsupported OS for automatic installation"
            exit 1
            ;;
    esac
    
    if command -v ollama &> /dev/null; then
        print_success "Ollama installed successfully"
    else
        print_error "Ollama installation failed"
        exit 1
    fi
}

# Start Ollama service
start_ollama() {
    print_step "Starting Ollama service..."
    
    # Check if Ollama is already running
    if curl -s http://localhost:11434/api/tags &> /dev/null; then
        print_success "Ollama service is already running"
        return 0
    fi
    
    # Start Ollama in background
    print_step "Starting Ollama server..."
    nohup ollama serve > /dev/null 2>&1 &
    
    # Wait for service to start
    local attempts=0
    local max_attempts=30
    
    while [ $attempts -lt $max_attempts ]; do
        if curl -s http://localhost:11434/api/tags &> /dev/null; then
            print_success "Ollama service started successfully"
            return 0
        fi
        
        sleep 1
        attempts=$((attempts + 1))
    done
    
    print_error "Failed to start Ollama service"
    exit 1
}

# Download recommended models
download_models() {
    print_step "Downloading recommended models..."
    
    for model in "${RECOMMENDED_MODELS[@]}"; do
        print_step "Downloading $model..."
        
        if ollama list | grep -q "$model"; then
            print_success "$model already downloaded"
        else
            print_step "Pulling $model (this may take a while)..."
            if ollama pull "$model"; then
                print_success "$model downloaded successfully"
            else
                print_warning "Failed to download $model - continuing with others"
            fi
        fi
    done
}

# Test AI integration
test_integration() {
    print_step "Testing AI integration..."
    
    # Test basic Ollama functionality
    print_step "Testing Ollama API..."
    if curl -s http://localhost:11434/api/tags | grep -q "models"; then
        print_success "Ollama API is working"
    else
        print_error "Ollama API test failed"
        return 1
    fi
    
    # Test model inference
    print_step "Testing model inference..."
    local test_response=$(curl -s -X POST http://localhost:11434/api/generate \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'$DEFAULT_MODEL'",
            "prompt": "Hello, this is a test. Please respond with just: AI test successful",
            "stream": false
        }' | grep -o '"response":"[^"]*"' | cut -d'"' -f4)
    
    if [ -n "$test_response" ]; then
        print_success "Model inference test passed"
        echo -e "${CYAN}Test response: $test_response${NC}"
    else
        print_warning "Model inference test failed - model may not be ready"
    fi
}

# Configure TARS for AI
configure_tars() {
    print_step "Configuring TARS for AI integration..."
    
    # Create AI configuration
    local config_file="./data/config/tars.ai.config.json"
    mkdir -p "./data/config"
    
    cat > "$config_file" << EOF
{
  "tars": {
    "llm": {
      "ollamaEndpoint": "http://localhost:11434",
      "defaultModel": "$DEFAULT_MODEL",
      "maxConcurrentRequests": 3,
      "requestTimeoutSeconds": 120,
      "enableCaching": true,
      "cacheTtlMinutes": 60,
      "enableCuda": $GPU_AVAILABLE,
      "maxMemoryUsage": 4294967296,
      "defaultTemperature": 0.7,
      "defaultMaxTokens": 2048,
      "enableProofGeneration": true
    }
  }
}
EOF
    
    print_success "TARS AI configuration created: $config_file"
}

# Show usage instructions
show_usage() {
    print_step "AI Integration Complete!"
    echo ""
    
    echo -e "${CYAN}🤖 Available Models:${NC}"
    ollama list 2>/dev/null || echo "  Run 'ollama list' to see available models"
    echo ""
    
    echo -e "${CYAN}🚀 TARS AI Commands:${NC}"
    echo "  tars ai --chat              # Start interactive AI chat"
    echo "  tars ai --status            # Check AI system status"
    echo "  tars ai --models            # List available models"
    echo ""
    
    echo -e "${CYAN}📋 Useful Ollama Commands:${NC}"
    echo "  ollama list                 # List downloaded models"
    echo "  ollama pull <model>         # Download a model"
    echo "  ollama rm <model>           # Remove a model"
    echo "  ollama serve                # Start Ollama server"
    echo ""
    
    echo -e "${CYAN}🔧 Recommended Models:${NC}"
    echo "  llama3.2:3b               # Fast, general purpose (3B parameters)"
    echo "  codellama:7b               # Code generation and analysis"
    echo "  mistral:7b                 # High quality responses"
    echo "  phi3:mini                  # Lightweight, fast responses"
    echo ""
    
    echo -e "${CYAN}💡 Tips:${NC}"
    echo "  • Use smaller models (3B) for faster responses"
    echo "  • Use larger models (7B+) for better quality"
    echo "  • GPU acceleration requires NVIDIA drivers and CUDA"
    echo "  • Models are cached locally for offline use"
    echo ""
    
    if [ "$GPU_AVAILABLE" = true ]; then
        echo -e "${GREEN}🎉 GPU acceleration is available!${NC}"
    else
        echo -e "${YELLOW}💻 Using CPU inference (slower but works)${NC}"
    fi
}

# Main setup function
main() {
    print_header
    
    case "${1:-install}" in
        "install")
            check_requirements
            install_ollama
            start_ollama
            download_models
            test_integration
            configure_tars
            show_usage
            print_success "TARS AI integration setup complete! 🎉"
            ;;
        "start")
            start_ollama
            print_success "Ollama service started"
            ;;
        "test")
            test_integration
            ;;
        "models")
            print_step "Downloading additional models..."
            download_models
            ;;
        "status")
            if curl -s http://localhost:11434/api/tags &> /dev/null; then
                print_success "Ollama service is running"
                echo ""
                echo "Available models:"
                ollama list
            else
                print_warning "Ollama service is not running"
                echo "Start with: $0 start"
            fi
            ;;
        *)
            echo "Usage: $0 {install|start|test|models|status}"
            echo ""
            echo "Commands:"
            echo "  install  - Complete AI setup (default)"
            echo "  start    - Start Ollama service"
            echo "  test     - Test AI integration"
            echo "  models   - Download additional models"
            echo "  status   - Check service status"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
