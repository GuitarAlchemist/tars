#!/bin/bash
# TARS Blue-Green Evolution Dashboard Deployment Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
DASHBOARD_PORT=${DASHBOARD_PORT:-8888}
OLLAMA_PORT=${OLLAMA_PORT:-11434}
COMPOSE_FILE="docker-compose.dashboard.yml"

print_header() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║           TARS Dashboard Docker Deployment                   ║"
    echo "║         Integrated Blue-Green Evolution Interface           ║"
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

print_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_step "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    print_success "Docker found: $(docker --version | head -1)"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    print_success "Docker Compose available"
    
    # Check if Docker is running
    if ! docker ps &> /dev/null; then
        print_error "Docker is not running"
        exit 1
    fi
    print_success "Docker daemon is running"
    
    # Check available space
    local available_space=$(df . | tail -1 | awk '{print $4}')
    if [ "$available_space" -lt 2000000 ]; then # 2GB in KB
        print_warning "Low disk space: $(df -h . | tail -1 | awk '{print $4}')"
    else
        print_success "Sufficient disk space: $(df -h . | tail -1 | awk '{print $4}')"
    fi
}

# Build TARS dashboard image
build_dashboard() {
    print_step "Building TARS dashboard Docker image..."
    
    if docker build -f Dockerfile.dashboard -t tars-dashboard:latest .; then
        print_success "TARS dashboard image built successfully"
    else
        print_error "Failed to build TARS dashboard image"
        exit 1
    fi
}

# Deploy with Docker Compose
deploy_compose() {
    print_step "Deploying TARS dashboard with Docker Compose..."
    
    # Create .env file for configuration
    cat > .env << EOF
DASHBOARD_PORT=${DASHBOARD_PORT}
OLLAMA_PORT=${OLLAMA_PORT}
COMPOSE_PROJECT_NAME=tars
EOF
    
    # Deploy services
    if docker-compose -f ${COMPOSE_FILE} up -d; then
        print_success "TARS dashboard deployed successfully"
    else
        print_error "Failed to deploy TARS dashboard"
        exit 1
    fi
}

# Wait for services to be ready
wait_for_services() {
    print_step "Waiting for services to be ready..."
    
    # Wait for dashboard
    print_info "Waiting for TARS dashboard..."
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s -f "http://localhost:${DASHBOARD_PORT}/api/status" > /dev/null 2>&1; then
            print_success "TARS dashboard is ready"
            break
        fi
        
        attempt=$((attempt + 1))
        echo -n "."
        sleep 2
    done
    
    if [ $attempt -eq $max_attempts ]; then
        print_warning "TARS dashboard took longer than expected to start"
    fi
    
    # Wait for Ollama
    print_info "Waiting for Ollama AI service..."
    attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s -f "http://localhost:${OLLAMA_PORT}/api/tags" > /dev/null 2>&1; then
            print_success "Ollama AI service is ready"
            break
        fi
        
        attempt=$((attempt + 1))
        echo -n "."
        sleep 2
    done
    
    if [ $attempt -eq $max_attempts ]; then
        print_warning "Ollama AI service took longer than expected to start"
    fi
}

# Setup AI models
setup_ai_models() {
    print_step "Setting up AI models..."
    
    # Pull required models
    local models=("llama3.2:3b" "qwen2.5:3b")
    
    for model in "${models[@]}"; do
        print_info "Pulling AI model: $model"
        if docker exec tars-ollama ollama pull "$model"; then
            print_success "Model $model ready"
        else
            print_warning "Failed to pull model $model"
        fi
    done
}

# Show deployment status
show_status() {
    print_step "Deployment Status"
    echo ""
    
    # Show running containers
    print_info "Running containers:"
    docker-compose -f ${COMPOSE_FILE} ps
    echo ""
    
    # Show service URLs
    print_success "🌐 TARS Dashboard: http://localhost:${DASHBOARD_PORT}"
    print_success "🤖 Ollama AI API: http://localhost:${OLLAMA_PORT}"
    print_success "🔄 Traefik Dashboard: http://localhost:8080"
    echo ""
    
    # Show logs command
    print_info "View logs: docker-compose -f ${COMPOSE_FILE} logs -f"
    print_info "Stop services: docker-compose -f ${COMPOSE_FILE} down"
    print_info "Restart services: docker-compose -f ${COMPOSE_FILE} restart"
}

# Cleanup function
cleanup() {
    print_step "Cleaning up..."
    
    if [ -f ".env" ]; then
        rm -f .env
        print_info "Cleaned up .env file"
    fi
}

# Handle errors
handle_error() {
    print_error "Deployment failed"
    cleanup
    exit 1
}

# Main deployment function
main() {
    print_header
    
    # Set up error handling
    trap handle_error ERR
    
    # Parse command line arguments
    case "${1:-deploy}" in
        "deploy")
            check_prerequisites
            build_dashboard
            deploy_compose
            wait_for_services
            setup_ai_models
            show_status
            ;;
        "stop")
            print_step "Stopping TARS dashboard..."
            docker-compose -f ${COMPOSE_FILE} down
            print_success "TARS dashboard stopped"
            ;;
        "restart")
            print_step "Restarting TARS dashboard..."
            docker-compose -f ${COMPOSE_FILE} restart
            print_success "TARS dashboard restarted"
            ;;
        "logs")
            docker-compose -f ${COMPOSE_FILE} logs -f
            ;;
        "status")
            show_status
            ;;
        "clean")
            print_step "Cleaning up TARS dashboard..."
            docker-compose -f ${COMPOSE_FILE} down -v --remove-orphans
            docker rmi tars-dashboard:latest 2>/dev/null || true
            print_success "TARS dashboard cleaned up"
            ;;
        *)
            echo "Usage: $0 {deploy|stop|restart|logs|status|clean}"
            echo ""
            echo "Commands:"
            echo "  deploy   - Deploy TARS dashboard (default)"
            echo "  stop     - Stop all services"
            echo "  restart  - Restart all services"
            echo "  logs     - Show service logs"
            echo "  status   - Show deployment status"
            echo "  clean    - Clean up everything"
            exit 1
            ;;
    esac
    
    cleanup
}

# Run main function
main "$@"
