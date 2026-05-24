# TARS Unified Architecture - Production Deployment Guide

## 🚀 Quick Start

Deploy TARS unified architecture in production with a single command:

```bash
./deploy.sh
```

## 📋 Prerequisites

- **Docker** 20.10+ with Docker Compose
- **4GB+ RAM** (8GB+ recommended)
- **10GB+ disk space** available
- **NVIDIA GPU** (optional, for CUDA acceleration)

## 🎯 What Gets Deployed

### Core Services
- **TARS Unified** (Port 8080) - Complete unified architecture
- **Redis** (Port 6379) - Distributed caching
- **Legacy Services** (Ports 8081-8084) - Existing TARS components

### Unified Systems Included
- ✅ **Unified Core Foundation** - Error handling and correlation tracking
- ✅ **Unified Configuration Management** - Centralized configuration with hot-reloading
- ✅ **Unified Proof Generation** - Cryptographic evidence for all operations
- ✅ **Unified Caching System** - Multi-level caching (memory, disk, distributed)
- ✅ **Unified Monitoring System** - Real-time monitoring with intelligent alerting
- ✅ **Unified CUDA Engine** - GPU acceleration with CPU fallback
- ✅ **Unified Agent Coordination** - Intelligent multi-agent orchestration

## 🔧 Deployment Commands

### Deploy Everything
```bash
./deploy.sh deploy
```

### Check Status
```bash
./deploy.sh status
```

### View Logs
```bash
./deploy.sh logs
```

### Run Health Checks
```bash
./deploy.sh health
```

### Rebuild and Redeploy
```bash
./deploy.sh rebuild
```

### Stop All Services
```bash
./deploy.sh stop
```

## 🌐 Service URLs

After deployment, access TARS services at:

- **TARS Unified**: http://localhost:8080
- **TARS Core**: http://localhost:8081
- **TARS Knowledge**: http://localhost:8082
- **TARS Agents**: http://localhost:8083
- **Model Runner**: http://localhost:8084

## 🧪 Testing the Deployment

### Run Comprehensive Diagnostics
```bash
docker-compose exec tars-unified dotnet TarsEngine.FSharp.Cli.dll diagnose --full
```

### Run All Tests
```bash
docker-compose exec tars-unified dotnet TarsEngine.FSharp.Cli.dll test
```

### Interactive Chat
```bash
docker-compose exec tars-unified dotnet TarsEngine.FSharp.Cli.dll chat --interactive
```

### Performance Demo
```bash
docker-compose exec tars-unified dotnet TarsEngine.FSharp.Cli.dll performance --combined
```

### Show Unified System Overview
```bash
docker-compose exec tars-unified dotnet TarsEngine.FSharp.Cli.dll unified --demo
```

## ⚙️ Configuration

### Default Configuration
The system uses `/app/data/config/tars.config.json` with production-ready defaults:

- **Caching**: 50K memory entries, 500K disk entries, Redis distributed cache
- **Monitoring**: 30-day metric retention, real-time alerting
- **CUDA**: Enabled with CPU fallback
- **Proof Generation**: 90-day retention with encryption
- **Logging**: Information level with file rotation

### Custom Configuration
1. Edit `docker/tars.config.json` before deployment
2. Or mount your own config file:
   ```bash
   docker-compose exec tars-unified cp /path/to/your/config.json /app/data/config/tars.config.json
   ```

## 📊 Monitoring and Observability

### Health Monitoring
- **Automatic Health Checks**: Every 30 seconds
- **Component Health Scoring**: Real-time health metrics
- **Intelligent Alerting**: Threshold-based alerts with severity levels

### Performance Monitoring
- **Cache Performance**: Hit ratios, response times
- **System Metrics**: CPU, memory, GPU utilization
- **Proof Generation**: Cryptographic audit trails

### Logs
- **Structured Logging**: JSON format with correlation IDs
- **Log Rotation**: 10 files × 100MB each
- **Real-time Viewing**: `./deploy.sh logs`

## 🔒 Security Features

### Cryptographic Proofs
- **All Operations**: Generate cryptographic evidence
- **Tamper Detection**: Automatic proof verification
- **Audit Trails**: Complete operation history

### Container Security
- **Non-root User**: Runs as `tars` user
- **Resource Limits**: Memory and CPU constraints
- **Network Isolation**: Dedicated Docker network

## 🚀 Production Optimizations

### Performance
- **Multi-level Caching**: Memory → Disk → Distributed
- **CUDA Acceleration**: GPU-accelerated operations
- **Connection Pooling**: Optimized database connections
- **Compression**: Enabled for cache and network

### Reliability
- **Health Checks**: Automatic service monitoring
- **Graceful Shutdown**: Signal handling
- **Resource Management**: Automatic cleanup
- **Error Recovery**: Graceful degradation

### Scalability
- **Horizontal Scaling**: Ready for Kubernetes
- **Load Balancing**: Built-in agent coordination
- **Resource Monitoring**: Capacity planning metrics

## 🐛 Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check logs
./deploy.sh logs

# Check Docker resources
docker system df
docker system prune  # If low on space
```

#### Health Check Failures
```bash
# Run manual diagnostics
docker-compose exec tars-unified dotnet TarsEngine.FSharp.Cli.dll diagnose --full

# Check service dependencies
docker-compose ps
```

#### Performance Issues
```bash
# Check resource usage
docker stats

# Run performance tests
docker-compose exec tars-unified dotnet TarsEngine.FSharp.Cli.dll performance --combined
```

#### CUDA Issues
```bash
# Check GPU availability
docker-compose exec tars-unified nvidia-smi

# Check CUDA configuration
docker-compose exec tars-unified dotnet TarsEngine.FSharp.Cli.dll unified --health
```

### Log Analysis
```bash
# Filter by correlation ID
docker-compose logs tars-unified | grep "correlation-id-here"

# Filter by error level
docker-compose logs tars-unified | grep "ERROR"

# Real-time monitoring
docker-compose logs -f tars-unified
```

## 📈 Scaling for Production

### Kubernetes Deployment
The Docker images are ready for Kubernetes deployment:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tars-unified
spec:
  replicas: 3
  selector:
    matchLabels:
      app: tars-unified
  template:
    metadata:
      labels:
        app: tars-unified
    spec:
      containers:
      - name: tars-unified
        image: tars-unified:unified-v2.0
        ports:
        - containerPort: 8080
        env:
        - name: TARS_ENVIRONMENT
          value: "Kubernetes"
        resources:
          requests:
            memory: "1Gi"
            cpu: "1"
          limits:
            memory: "2Gi"
            cpu: "2"
```

### Load Balancing
Use nginx or cloud load balancers to distribute traffic across multiple TARS instances.

### Database Scaling
For high-load scenarios, consider:
- PostgreSQL read replicas
- Redis clustering
- Separate cache and session storage

## 🎉 Success!

If everything is working correctly, you should see:

```
✅ TARS Unified Architecture deployed successfully! 🎉

🌐 Service URLs:
  TARS Unified:     http://localhost:8080
  
🚀 Starting TARS with unified systems:
   ✅ Unified Core Foundation
   ✅ Unified Configuration Management
   ✅ Unified Proof Generation
   ✅ Unified Caching System
   ✅ Unified Monitoring System
   ✅ Unified CUDA Engine
   ✅ Unified Agent Coordination
```

**TARS Unified Architecture is now running in production mode!** 🚀
