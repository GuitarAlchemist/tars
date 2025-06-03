# TARS Deployment Guide

## 🚀 **Quick Start Options**

TARS can be deployed in multiple ways depending on your needs:

### **1. 🐳 Docker Compose (Recommended for Development)**
```bash
# Windows
.\scripts\deploy-docker.cmd

# Linux/Mac
docker-compose up -d
```

### **2. ☸️ Minikube (Local Kubernetes)**
```bash
# Linux/Mac
chmod +x scripts/deploy-minikube.sh
./scripts/deploy-minikube.sh

# Windows (WSL)
bash scripts/deploy-minikube.sh
```

### **3. ⎈ Kubernetes with Helm**
```bash
# Add TARS Helm repository
helm repo add tars ./helm

# Install TARS
helm install tars tars/tars -n tars --create-namespace

# Or with custom values
helm install tars tars/tars -n tars --create-namespace -f custom-values.yaml
```

### **4. ☁️ Cloud Deployment (Azure AKS / AWS EKS)**
```bash
# Azure AKS
az aks get-credentials --resource-group myResourceGroup --name myAKSCluster
helm install tars tars/tars -n tars --create-namespace -f values-azure.yaml

# AWS EKS
aws eks update-kubeconfig --region us-west-2 --name my-cluster
helm install tars tars/tars -n tars --create-namespace -f values-aws.yaml
```

---

## 📋 **Prerequisites**

### **For Docker Deployment:**
- Docker Desktop 4.0+ with Docker Compose
- 8GB RAM minimum, 16GB recommended
- 20GB free disk space

### **For Kubernetes Deployment:**
- kubectl 1.28+
- Helm 3.8+
- Kubernetes cluster 1.28+
- NGINX Ingress Controller (for ingress)

### **For Minikube:**
- minikube 1.32+
- VirtualBox, Docker, or Hyper-V driver
- 8GB RAM minimum for minikube VM

---

## 🏗️ **Architecture Overview**

TARS consists of 6 main microservices:

```
┌─────────────────────────────────────────────────────────────┐
│                    TARS Architecture                       │
├─────────────────┬─────────────────┬─────────────────────────┤
│ TARS UI Service │ TARS Core       │ TARS Knowledge Service │
│ (Port 3000)     │ (Port 8080)     │ (Port 8081)             │
│ • Internal      │ • Executive     │ • Historian Agent      │
│   Dialogue UI   │   Leadership    │ • Librarian Agent      │
│ • Template-Free │ • Operations    │ • Researcher Agent     │
│   Generation    │ • Infrastructure│ • Reporter Agent       │
├─────────────────┼─────────────────┼─────────────────────────┤
│ TARS Agent      │ Redis Cache     │ Monitoring Stack       │
│ Service         │ (Port 6379)     │ • Prometheus (9091)     │
│ (Port 8082)     │ • Session Store │ • Grafana (3001)        │
│ • Humor Gen     │ • Cache Layer   │ • Metrics Collection    │
│ • Personality   │ • Pub/Sub       │ • Dashboards            │
└─────────────────┴─────────────────┴─────────────────────────┘
```

---

## 🎯 **Access URLs After Deployment**

### **Docker Compose:**
- **TARS UI:** http://localhost:3000
- **TARS API:** http://localhost:8080
- **Knowledge API:** http://localhost:8081
- **Agent API:** http://localhost:8082
- **Prometheus:** http://localhost:9091
- **Grafana:** http://localhost:3001 (admin/tars-admin)

### **Minikube:**
- **TARS UI:** http://$(minikube ip):30080
- **Ingress:** http://tars.local (add to /etc/hosts)
- **Dashboard:** `minikube dashboard`

### **Kubernetes with Ingress:**
- **TARS UI:** http://tars.local
- **API:** http://tars.local/api
- **Knowledge:** http://tars.local/knowledge
- **Agents:** http://tars.local/agents

---

## 🔧 **Configuration Options**

### **Environment Variables:**
```bash
# Core Configuration
TARS_ENVIRONMENT=development|production
TARS_LOG_LEVEL=Debug|Information|Warning|Error
TARS_ENABLE_SWAGGER=true|false

# Feature Flags
ENABLE_INTERNAL_DIALOGUE_ACCESS=true
ENABLE_TEMPLATE_FREE_UI=true
ENABLE_LIVE_DOCUMENTATION=true
ENABLE_HUMOR_GENERATION=true
ENABLE_PERSONALITY_PARAMETERS=true
ENABLE_HYPERLIGHT_INTEGRATION=false

# Personality Defaults
DEFAULT_WIT_LEVEL=0.7
DEFAULT_SARCASM_FREQUENCY=0.3
DEFAULT_PUN_TENDENCY=0.5
DEFAULT_ENTHUSIASM=0.7
```

### **Helm Values Customization:**
```yaml
# custom-values.yaml
tarsCore:
  replicaCount: 3
  resources:
    requests:
      memory: "512Mi"
      cpu: "200m"
    limits:
      memory: "1Gi"
      cpu: "1000m"

ingress:
  hosts:
    - host: tars.yourdomain.com
      paths:
        - path: /
          pathType: Prefix
          service: tars-ui

monitoring:
  prometheus:
    enabled: true
  grafana:
    enabled: true
    adminPassword: "your-secure-password"
```

---

## 🛠️ **Management Commands**

### **Docker Compose:**
```bash
# View logs
docker-compose logs -f [service-name]

# Scale services
docker-compose up -d --scale tars-core=3

# Update services
docker-compose pull && docker-compose up -d

# Stop all services
docker-compose stop

# Remove all containers and volumes
docker-compose down -v
```

### **Kubernetes:**
```bash
# View pods
kubectl get pods -n tars

# View services
kubectl get services -n tars

# View logs
kubectl logs -f deployment/tars-core-service -n tars

# Port forward for local access
kubectl port-forward service/tars-ui-service 3000:80 -n tars

# Scale deployment
kubectl scale deployment tars-core-service --replicas=3 -n tars

# Update deployment
kubectl set image deployment/tars-core-service tars-core=tars/core-service:v1.1.0 -n tars
```

### **Helm:**
```bash
# List releases
helm list -n tars

# Upgrade release
helm upgrade tars tars/tars -n tars -f values.yaml

# Rollback release
helm rollback tars 1 -n tars

# Uninstall release
helm uninstall tars -n tars
```

---

## 🔍 **Troubleshooting**

### **Common Issues:**

#### **1. Services Not Starting**
```bash
# Check container logs
docker-compose logs tars-core

# Check Kubernetes events
kubectl get events -n tars --sort-by='.lastTimestamp'

# Check pod status
kubectl describe pod <pod-name> -n tars
```

#### **2. Cannot Access UI**
```bash
# Check if services are running
docker-compose ps
kubectl get pods -n tars

# Check port forwarding
kubectl port-forward service/tars-ui-service 3000:80 -n tars

# Check ingress configuration
kubectl get ingress -n tars
```

#### **3. Database Connection Issues**
```bash
# Check Redis connectivity
docker-compose exec redis redis-cli ping
kubectl exec -it deployment/redis -n tars -- redis-cli ping

# Check environment variables
docker-compose exec tars-core env | grep DATABASE
kubectl exec -it deployment/tars-core-service -n tars -- env | grep DATABASE
```

#### **4. Memory/CPU Issues**
```bash
# Check resource usage
docker stats
kubectl top pods -n tars

# Adjust resource limits in docker-compose.yml or values.yaml
```

### **Health Checks:**
```bash
# Docker Compose
curl http://localhost:8080/health/live
curl http://localhost:8080/health/ready

# Kubernetes
kubectl get pods -n tars
kubectl exec -it deployment/tars-core-service -n tars -- curl localhost:8080/health/live
```

---

## 📊 **Monitoring & Observability**

### **Prometheus Metrics:**
- **Application Metrics:** http://localhost:9090/metrics
- **Custom TARS Metrics:** Reasoning performance, agent interactions
- **Infrastructure Metrics:** CPU, memory, network, storage

### **Grafana Dashboards:**
- **TARS Overview:** System health and performance
- **Agent Performance:** Humor generation, personality adaptation
- **Knowledge Management:** Milestone capture, research metrics
- **Infrastructure:** Kubernetes cluster metrics

### **Log Aggregation:**
```bash
# Centralized logging with Docker Compose
docker-compose logs -f

# Kubernetes logging
kubectl logs -f -l app.kubernetes.io/name=tars -n tars

# Structured logging format: JSON with correlation IDs
```

---

## 🔒 **Security Considerations**

### **Production Deployment:**
1. **Change default passwords** in Grafana and other services
2. **Use proper TLS certificates** for HTTPS
3. **Configure network policies** to restrict pod-to-pod communication
4. **Enable authentication** for all external endpoints
5. **Use secrets management** for sensitive configuration
6. **Regular security updates** for base images

### **Network Security:**
```yaml
# Example NetworkPolicy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: tars-network-policy
  namespace: tars
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
```

---

## 🚀 **Next Steps**

After successful deployment:

1. **Access TARS UI** at the configured URL
2. **Explore Internal Dialogue** visualization
3. **Test Template-Free UI** generation
4. **Adjust Personality Parameters** for different agents
5. **Monitor System Performance** via Grafana dashboards
6. **Review Knowledge Management** capabilities
7. **Experiment with Humor Generation** features

---

## 📞 **Support**

For deployment issues or questions:
- **Documentation:** Check this guide and inline comments
- **Logs:** Always check service logs first
- **Health Checks:** Verify all health endpoints are responding
- **Resource Monitoring:** Ensure adequate CPU/memory allocation

**🎉 Happy deploying with TARS!** 🤖
