#!/bin/bash
# TARS Unified Architecture - Production Deployment Script
# Deploys the complete unified system with all components

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
TARS_VERSION="unified-v2.0"
DOCKER_NETWORK="tars-network"
COMPOSE_FILE="docker-compose.yml"

# Functions
print_header() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                 TARS Unified Architecture                    ║"
    echo "║                Production Deployment Script                  ║"
    echo "║                     Version: $TARS_VERSION                        ║"
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

# Check prerequisites
check_prerequisites() {
    print_step "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    print_success "Docker found: $(docker --version)"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    
    if command -v docker-compose &> /dev/null; then
        print_success "Docker Compose found: $(docker-compose --version)"
        COMPOSE_CMD="docker-compose"
    else
        print_success "Docker Compose found: $(docker compose version)"
        COMPOSE_CMD="docker compose"
    fi
    
    # Check .NET SDK (for building)
    if ! command -v dotnet &> /dev/null; then
        print_warning ".NET SDK not found - will build in Docker"
    else
        print_success ".NET SDK found: $(dotnet --version)"
    fi
    
    # Check available disk space
    local available_space=$(df . | tail -1 | awk '{print $4}')
    if [ "$available_space" -lt 5000000 ]; then # 5GB in KB
        print_warning "Low disk space available: $(df -h . | tail -1 | awk '{print $4}')"
    else
        print_success "Sufficient disk space available: $(df -h . | tail -1 | awk '{print $4}')"
    fi
}

# Create Docker network
create_network() {
    print_step "Creating Docker network..."
    
    if docker network ls | grep -q "$DOCKER_NETWORK"; then
        print_success "Network '$DOCKER_NETWORK' already exists"
    else
        docker network create "$DOCKER_NETWORK"
        print_success "Network '$DOCKER_NETWORK' created"
    fi
}

# Build TARS image
build_tars() {
    print_step "Building TARS unified architecture..."
    
    # Run tests first
    print_step "Running tests..."
    if command -v dotnet &> /dev/null; then
        dotnet test TarsEngine.FSharp.Tests/TarsEngine.FSharp.Tests.fsproj --verbosity normal
        print_success "Tests passed"
    else
        print_warning "Skipping local tests - will run in Docker"
    fi
    
    # Build Docker image
    docker build -t tars-unified:$TARS_VERSION .
    print_success "TARS unified image built"
}

# Deploy services
deploy_services() {
    print_step "Deploying TARS unified services..."
    
    # Create required directories
    mkdir -p ./data/{cache,logs,config,proofs,monitoring}
    mkdir -p ./flux
    
    # Copy configuration if it doesn't exist
    if [ ! -f "./data/config/tars.config.json" ]; then
        cp docker/tars.config.json ./data/config/
        print_success "Configuration file copied"
    fi
    
    # Deploy with Docker Compose
    $COMPOSE_CMD -f $COMPOSE_FILE up -d
    print_success "Services deployed"
}

# Wait for services
wait_for_services() {
    print_step "Waiting for services to be ready..."
    
    local services=("redis" "tars-unified")
    local max_attempts=30
    
    for service in "${services[@]}"; do
        local attempt=1
        print_step "Waiting for $service..."
        
        while [ $attempt -le $max_attempts ]; do
            if $COMPOSE_CMD -f $COMPOSE_FILE ps $service | grep -q "Up"; then
                print_success "$service is ready"
                break
            fi
            
            if [ $attempt -eq $max_attempts ]; then
                print_error "$service failed to start"
                return 1
            fi
            
            sleep 2
            attempt=$((attempt + 1))
        done
    done
}

# Run health checks
run_health_checks() {
    print_step "Running health checks..."
    
    # Check TARS unified service
    if $COMPOSE_CMD -f $COMPOSE_FILE exec -T tars-unified dotnet TarsEngine.FSharp.Cli.dll diagnose --status; then
        print_success "TARS unified service health check passed"
    else
        print_warning "TARS unified service health check failed"
    fi
    
    # Check Redis
    if $COMPOSE_CMD -f $COMPOSE_FILE exec -T redis redis-cli ping | grep -q "PONG"; then
        print_success "Redis health check passed"
    else
        print_warning "Redis health check failed"
    fi
}

# Show deployment status
show_status() {
    print_step "Deployment Status:"
    echo ""
    
    # Show running services
    echo -e "${CYAN}🔧 Running Services:${NC}"
    $COMPOSE_CMD -f $COMPOSE_FILE ps
    echo ""
    
    # Show service URLs
    echo -e "${CYAN}🌐 Service URLs:${NC}"
    echo "  TARS Unified:     http://localhost:8080"
    echo "  TARS Core:        http://localhost:8081"
    echo "  TARS Knowledge:   http://localhost:8082"
    echo "  TARS Agents:      http://localhost:8083"
    echo "  Model Runner:     http://localhost:8084"
    echo ""
    
    # Show useful commands
    echo -e "${CYAN}📋 Useful Commands:${NC}"
    echo "  View logs:        $COMPOSE_CMD -f $COMPOSE_FILE logs -f tars-unified"
    echo "  Run diagnostics:  $COMPOSE_CMD -f $COMPOSE_FILE exec tars-unified dotnet TarsEngine.FSharp.Cli.dll diagnose --full"
    echo "  Run tests:        $COMPOSE_CMD -f $COMPOSE_FILE exec tars-unified dotnet TarsEngine.FSharp.Cli.dll test"
    echo "  Interactive chat: $COMPOSE_CMD -f $COMPOSE_FILE exec tars-unified dotnet TarsEngine.FSharp.Cli.dll chat --interactive"
    echo "  Performance demo: $COMPOSE_CMD -f $COMPOSE_FILE exec tars-unified dotnet TarsEngine.FSharp.Cli.dll performance --combined"
    echo "  Stop services:    $COMPOSE_CMD -f $COMPOSE_FILE down"
    echo ""
}

# Cleanup function
cleanup() {
    print_step "Cleaning up..."
    $COMPOSE_CMD -f $COMPOSE_FILE down
    print_success "Services stopped"
}

# Main deployment function
main() {
    print_header
    
    # Parse command line arguments
    case "${1:-deploy}" in
        "deploy")
            check_prerequisites
            create_network
            build_tars
            deploy_services
            wait_for_services
            run_health_checks
            show_status
            print_success "TARS Unified Architecture deployed successfully! 🎉"
            ;;
        "stop")
            print_step "Stopping TARS services..."
            cleanup
            ;;
        "status")
            show_status
            ;;
        "logs")
            $COMPOSE_CMD -f $COMPOSE_FILE logs -f tars-unified
            ;;
        "health")
            run_health_checks
            ;;
        "rebuild")
            print_step "Rebuilding TARS..."
            $COMPOSE_CMD -f $COMPOSE_FILE down
            build_tars
            deploy_services
            wait_for_services
            run_health_checks
            show_status
            ;;
        *)
            echo "Usage: $0 {deploy|stop|status|logs|health|rebuild}"
            echo ""
            echo "Commands:"
            echo "  deploy   - Deploy TARS unified architecture (default)"
            echo "  stop     - Stop all services"
            echo "  status   - Show deployment status"
            echo "  logs     - Show TARS logs"
            echo "  health   - Run health checks"
            echo "  rebuild  - Rebuild and redeploy"
            exit 1
            ;;
    esac
}

# Handle signals
trap cleanup SIGINT SIGTERM

# Run main function
main "$@"
