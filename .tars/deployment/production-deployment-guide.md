# TARS AI Engine - Production Deployment Guide

## 🚀 Quick Start

TARS AI Engine can be deployed in multiple ways for different use cases:

- 🐳 **Docker**: Single-node deployment with GPU support
- ☸️ **Kubernetes**: Enterprise-scale orchestration
- 🖥️ **Local**: Development and testing
- ☁️ **Cloud**: AWS, GCP, Azure deployment

## 🐳 Docker Deployment

### Prerequisites
- Docker 20.10+ with GPU support
- NVIDIA Container Toolkit (for GPU acceleration)
- 8GB+ RAM, 50GB+ storage
- NVIDIA GPU with 8GB+ VRAM (recommended)

### Quick Deploy
```bash
# Clone repository
git clone https://github.com/GuitarAlchemist/tars.git
cd tars

# Build TARS AI image
docker build -f Dockerfile.ai -t tars-ai:latest .

# Run with GPU support
docker run -d \
  --name tars-ai \
  --gpus all \
  -p 11434:11434 \
  -v ./models:/app/models \
  -e TARS_CUDA_ENABLED=true \
  -e TARS_LOG_LEVEL=Info \
  tars-ai:latest

# Verify deployment
curl http://localhost:11434/api/tags
```

### Docker Compose Deployment
```bash
# Deploy full stack with monitoring
docker-compose -f docker-compose.ai.yml up -d

# Services included:
# - TARS AI Engine (port 11434)
# - Load Balancer (port 80/443)
# - Redis Cache (port 6379)
# - Prometheus (port 9090)
# - Grafana (port 3000)
```

### Environment Variables
```bash
# Core Configuration
TARS_MODELS_PATH=/app/models          # Model storage path
TARS_CUDA_ENABLED=true               # Enable GPU acceleration
TARS_LOG_LEVEL=Info                  # Logging level
TARS_MAX_CONCURRENT_REQUESTS=50      # Concurrent request limit
TARS_CACHE_SIZE=5000                 # Response cache size

# Optimization Settings
TARS_OPTIMIZATION_ENABLED=true       # Enable real-time optimization
TARS_OPTIMIZATION_INTERVAL=30m       # Optimization frequency
TARS_GENETIC_POPULATION_SIZE=15      # GA population size
TARS_GENETIC_MAX_ITERATIONS=25       # GA max iterations

# Monitoring
TARS_METRICS_ENABLED=true            # Enable metrics collection
TARS_HEALTH_CHECK_INTERVAL=30s       # Health check frequency
```

## ☸️ Kubernetes Deployment

### Prerequisites
- Kubernetes 1.20+
- NVIDIA GPU Operator
- Persistent storage (100GB+)
- Load balancer (NGINX Ingress)

### Deploy to Kubernetes
```bash
# Apply TARS AI manifests
kubectl apply -f k8s/tars-ai-deployment.yaml

# Verify deployment
kubectl get pods -n tars-ai
kubectl get services -n tars-ai

# Check logs
kubectl logs -f deployment/tars-ai-engine -n tars-ai
```

### Scaling Operations
```bash
# Scale to 10 replicas
kubectl scale deployment tars-ai-engine --replicas=10 -n tars-ai

# Enable auto-scaling (2-20 replicas)
kubectl apply -f k8s/tars-ai-hpa.yaml

# Monitor scaling
kubectl get hpa -n tars-ai
```

### Resource Requirements

#### Minimum (Development)
```yaml
resources:
  requests:
    memory: "4Gi"
    cpu: "2"
    nvidia.com/gpu: 1
  limits:
    memory: "8Gi"
    cpu: "4"
    nvidia.com/gpu: 1
```

#### Production (Recommended)
```yaml
resources:
  requests:
    memory: "8Gi"
    cpu: "4"
    nvidia.com/gpu: 1
  limits:
    memory: "32Gi"
    cpu: "16"
    nvidia.com/gpu: 1
```

#### High-Performance (Large Models)
```yaml
resources:
  requests:
    memory: "32Gi"
    cpu: "8"
    nvidia.com/gpu: 2
  limits:
    memory: "128Gi"
    cpu: "32"
    nvidia.com/gpu: 4
```

## 🖥️ Local Development

### Prerequisites
- .NET 8.0 SDK
- F# compiler
- CUDA Toolkit 12.2+ (optional)
- 16GB+ RAM

### Build and Run
```bash
# Clone repository
git clone https://github.com/GuitarAlchemist/tars.git
cd tars

# Build TARS Engine
dotnet build src/TarsEngine/TarsEngine.fsproj

# Run AI engine
dotnet run --project src/TarsEngine -- --server --port 11434

# Test API
curl -X POST http://localhost:11434/api/generate \
     -H "Content-Type: application/json" \
     -d '{"model":"tars-medium-7b","prompt":"Hello TARS!"}'
```

### Development Configuration
```bash
# Environment setup
export TARS_MODELS_PATH=./models
export TARS_CUDA_ENABLED=true
export TARS_LOG_LEVEL=Debug
export TARS_OPTIMIZATION_ENABLED=true

# Run with hot reload
dotnet watch run --project src/TarsEngine
```

## ☁️ Cloud Deployment

### AWS Deployment

#### EKS (Elastic Kubernetes Service)
```bash
# Create EKS cluster with GPU nodes
eksctl create cluster \
  --name tars-ai-cluster \
  --region us-west-2 \
  --nodegroup-name gpu-nodes \
  --node-type p3.2xlarge \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 10

# Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

# Deploy TARS AI
kubectl apply -f k8s/tars-ai-deployment.yaml
```

#### ECS (Elastic Container Service)
```json
{
  "family": "tars-ai-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["EC2"],
  "cpu": "4096",
  "memory": "16384",
  "containerDefinitions": [
    {
      "name": "tars-ai",
      "image": "tars-ai:latest",
      "portMappings": [{"containerPort": 11434}],
      "resourceRequirements": [
        {"type": "GPU", "value": "1"}
      ]
    }
  ]
}
```

### GCP Deployment

#### GKE (Google Kubernetes Engine)
```bash
# Create GKE cluster with GPU nodes
gcloud container clusters create tars-ai-cluster \
  --zone us-central1-a \
  --machine-type n1-standard-4 \
  --accelerator type=nvidia-tesla-v100,count=1 \
  --num-nodes 3 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 10

# Install GPU drivers
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml

# Deploy TARS AI
kubectl apply -f k8s/tars-ai-deployment.yaml
```

### Azure Deployment

#### AKS (Azure Kubernetes Service)
```bash
# Create AKS cluster with GPU nodes
az aks create \
  --resource-group tars-ai-rg \
  --name tars-ai-cluster \
  --node-count 3 \
  --node-vm-size Standard_NC6s_v3 \
  --enable-cluster-autoscaler \
  --min-count 1 \
  --max-count 10

# Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

# Deploy TARS AI
kubectl apply -f k8s/tars-ai-deployment.yaml
```

## 📊 Monitoring and Observability

### Prometheus Metrics
```yaml
# Key metrics exposed
- tars_requests_total
- tars_request_duration_seconds
- tars_tokens_generated_total
- tars_optimization_cycles_total
- tars_gpu_utilization_percent
- tars_memory_usage_bytes
```

### Grafana Dashboards
- **TARS AI Overview**: High-level performance metrics
- **Model Performance**: Per-model latency and throughput
- **Resource Utilization**: CPU, GPU, memory usage
- **Optimization Tracking**: Real-time optimization progress

### Health Checks
```bash
# Liveness probe
curl http://localhost:11434/health

# Readiness probe  
curl http://localhost:11434/api/tags

# Metrics endpoint
curl http://localhost:11434/metrics
```

## 🔒 Security Configuration

### Container Security
```dockerfile
# Non-root user
USER tars

# Read-only filesystem
--read-only

# No privileged access
--security-opt no-new-privileges

# Resource limits
--memory 16g --cpus 8
```

### Network Security
```yaml
# Network policies
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: tars-ai-network-policy
spec:
  podSelector:
    matchLabels:
      app: tars-ai-engine
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: load-balancer
    ports:
    - protocol: TCP
      port: 11434
```

### TLS Configuration
```yaml
# TLS termination at ingress
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: tars-ai-ingress
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - tars-ai.yourdomain.com
    secretName: tars-ai-tls
```

## 🚀 Performance Optimization

### GPU Optimization
```bash
# Enable GPU persistence mode
nvidia-smi -pm 1

# Set GPU performance mode
nvidia-smi -ac 877,1530

# Monitor GPU usage
nvidia-smi -l 1
```

### Memory Optimization
```yaml
# Kubernetes memory settings
env:
- name: TARS_MEMORY_POOL_SIZE
  value: "8GB"
- name: TARS_CACHE_SIZE
  value: "2GB"
- name: TARS_BATCH_SIZE
  value: "16"
```

### Network Optimization
```yaml
# Service configuration
apiVersion: v1
kind: Service
metadata:
  name: tars-ai-service
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
    service.beta.kubernetes.io/aws-load-balancer-backend-protocol: "tcp"
spec:
  type: LoadBalancer
  sessionAffinity: ClientIP
```

## 🔧 Troubleshooting

### Common Issues

#### GPU Not Detected
```bash
# Check GPU availability
nvidia-smi

# Verify CUDA installation
nvcc --version

# Check container GPU access
docker run --gpus all nvidia/cuda:12.2-base nvidia-smi
```

#### High Memory Usage
```bash
# Monitor memory usage
kubectl top pods -n tars-ai

# Check for memory leaks
kubectl logs -f deployment/tars-ai-engine -n tars-ai | grep "memory"

# Restart deployment
kubectl rollout restart deployment/tars-ai-engine -n tars-ai
```

#### Performance Issues
```bash
# Check resource limits
kubectl describe pod <pod-name> -n tars-ai

# Monitor metrics
curl http://localhost:11434/metrics

# Enable debug logging
kubectl set env deployment/tars-ai-engine TARS_LOG_LEVEL=Debug -n tars-ai
```

### Support Resources
- 📖 **Documentation**: https://github.com/GuitarAlchemist/tars/docs
- 🐛 **Issues**: https://github.com/GuitarAlchemist/tars/issues
- 💬 **Discussions**: https://github.com/GuitarAlchemist/tars/discussions
- 📧 **Support**: support@tars-ai.org

### Support Resources
- 📖 **Documentation**: https://github.com/GuitarAlchemist/tars/docs
- 🐛 **Issues**: https://github.com/GuitarAlchemist/tars/issues
- 💬 **Discussions**: https://github.com/GuitarAlchemist/tars/discussions
- 📧 **Support**: support@tars-ai.org

---

**Deployment Status**: ✅ PRODUCTION READY
**Scalability**: ♾️ UNLIMITED
**Support**: 🌟 COMPREHENSIVE
**Documentation**: 📚 COMPLETE
