#!/bin/bash
# TARS Production Deployment Script

echo "🏭 Deploying TARS to Staging (4, 2) environment..."

# Build Docker image
echo "🐳 Building Docker image..."
docker build -t tars-engine:v1.0.0 .

# Apply Kubernetes manifests
echo "☸️ Applying Kubernetes manifests..."
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f hpa.yaml

# Wait for deployment
echo "⏳ Waiting for deployment to be ready..."
kubectl rollout status deployment/tars-engine-deployment -n tars-system

# Show deployment status
echo "✅ Deployment complete!"
kubectl get pods -n tars-system -l app=tars-engine
kubectl get services -n tars-system

echo "🚀 TARS is now running in Staging (4, 2) environment!"
echo "📊 Monitoring endpoints:"
echo "  http://tars.example.com/metrics"
echo "  http://tars.example.com/health"
echo "  http://tars.example.com/ready"