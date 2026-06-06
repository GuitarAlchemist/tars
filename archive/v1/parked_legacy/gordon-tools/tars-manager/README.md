# TARS Manager - Gordon Integration Tool

🚀 **Intelligent TARS infrastructure management powered by Gordon AI**

TARS Manager is a comprehensive tool that integrates Gordon's AI capabilities with TARS infrastructure management, providing intelligent analysis, automated consolidation, and real-time monitoring.

## 🌟 Features

### 🤖 AI-Powered Analysis
- **Gordon Integration**: Leverages Gordon's AI for intelligent infrastructure analysis
- **Health Monitoring**: Comprehensive health scoring and status tracking
- **Performance Analysis**: AI-driven performance optimization recommendations
- **Security Assessment**: Automated security vulnerability detection

### 🔄 Smart Consolidation
- **Staged Migration**: Database → Application → Web tier consolidation
- **Safety Checks**: Built-in rollback strategies and risk assessment
- **Dry Run Mode**: Preview changes before execution
- **Gordon Guidance**: AI-assisted consolidation planning

### 📊 Real-Time Monitoring
- **Container Tracking**: Monitor all TARS-related containers
- **Service Health**: Real-time service status and metrics
- **Performance Metrics**: CPU, memory, and network monitoring
- **Alert System**: Intelligent alerting based on AI analysis

### ⚡ Performance Optimization
- **AI Recommendations**: Gordon-powered optimization suggestions
- **Resource Efficiency**: Automated resource usage optimization
- **Scalability Planning**: AI-driven scaling recommendations
- **Cost Optimization**: Resource cost analysis and optimization

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Gordon AI assistant running
- TARS infrastructure deployed

### Installation

1. **Build the TARS Manager container:**
```bash
cd gordon-tools/tars-manager
docker build -t tars/gordon-manager:latest .
```

2. **Start TARS Manager with unified stack:**
```bash
docker-compose -f docker-compose.unified.yml up -d tars-manager
```

3. **Verify installation:**
```bash
curl http://localhost:8998/api/v1/health
```

## 📋 Usage

### Command Line Interface

The TARS CLI provides easy access to all TARS Manager features:

```bash
# Make CLI executable
chmod +x gordon-tools/tars-manager/tars-cli.py

# Basic health analysis
python tars-cli.py analyze --type=health

# Preview consolidation
python tars-cli.py consolidate --dry-run

# Monitor infrastructure
python tars-cli.py monitor --services=all

# Get optimization recommendations
python tars-cli.py optimize --target=performance

# Check status
python tars-cli.py status
```

### API Endpoints

TARS Manager exposes a REST API for programmatic access:

```bash
# Health check
GET /api/v1/health

# Infrastructure status
GET /api/v1/tars/status

# Trigger analysis
POST /api/v1/tars/analyze
{
  "analysis_type": "health",
  "deep_analysis": false
}

# Execute consolidation
POST /api/v1/tars/consolidate
{
  "stage": "all",
  "dry_run": true,
  "force": false
}

# Get optimization recommendations
POST /api/v1/tars/optimize
```

### Gordon Toolbox Integration

TARS Manager integrates seamlessly with Gordon's toolbox interface:

1. **Quick Actions**: One-click health checks, monitoring, and consolidation
2. **Smart Commands**: AI-assisted command completion and suggestions
3. **Context Awareness**: Gordon understands TARS infrastructure context
4. **Learning**: Gordon learns from TARS operations to improve recommendations

## 🔧 Configuration

### Environment Variables

```bash
# Gordon API connection
GORDON_API_URL=http://gordon:8997

# TARS API connection
TARS_API_URL=http://tars-main:8080

# Docker socket for container management
DOCKER_SOCKET=/var/run/docker.sock

# Logging level
LOG_LEVEL=INFO

# Analysis interval (seconds)
ANALYSIS_INTERVAL=30
```

### Configuration Files

- `config/tars-manager.yaml`: Main configuration
- `gordon-integration.yaml`: Gordon toolbox integration settings
- `tool-definition.json`: Tool metadata and capabilities

## 📊 Monitoring and Metrics

### Health Scoring
TARS Manager provides a comprehensive health score (0-100) based on:
- Container status and health
- Service availability
- Resource utilization
- Performance metrics
- Security posture

### Key Metrics
- **Infrastructure Health**: Overall system health percentage
- **Service Availability**: Uptime and response time metrics
- **Resource Usage**: CPU, memory, disk, and network utilization
- **Performance**: Response times and throughput metrics
- **Security**: Vulnerability and compliance scores

## 🔄 Consolidation Process

### Staged Approach
1. **Database Tier**: MongoDB, ChromaDB, Redis, Fuseki, Virtuoso
2. **Application Tier**: TARS main, autonomous agents, AI services
3. **Web Tier**: NGINX, management UIs, monitoring dashboards

### Safety Features
- **Dry Run Mode**: Preview all changes before execution
- **Rollback Strategy**: Automatic rollback on failure
- **Health Checks**: Continuous monitoring during consolidation
- **Gordon Analysis**: AI-powered risk assessment

## 🤖 Gordon AI Integration

### Capabilities
- **Infrastructure Analysis**: AI-powered infrastructure assessment
- **Predictive Maintenance**: Proactive issue detection
- **Optimization Recommendations**: Performance and cost optimization
- **Automated Responses**: Intelligent incident response

### Learning Features
- **Pattern Recognition**: Learns from infrastructure patterns
- **Anomaly Detection**: Identifies unusual behavior
- **Trend Analysis**: Predicts future resource needs
- **Best Practices**: Suggests industry best practices

## 🔒 Security

### Authentication
- API key-based authentication
- Role-based access control (RBAC)
- TLS 1.3 encryption

### Audit Logging
- All operations logged with timestamps
- User attribution and action tracking
- Security event monitoring
- Compliance reporting

## 🚨 Troubleshooting

### Common Issues

1. **Gordon Connection Failed**
   ```bash
   # Check Gordon service status
   docker ps | grep gordon
   
   # Verify Gordon API
   curl http://localhost:8997/api/health
   ```

2. **TARS API Unreachable**
   ```bash
   # Check TARS main service
   docker ps | grep tars-main
   
   # Verify TARS API
   curl http://localhost:8080/api/health
   ```

3. **Docker Socket Permission Denied**
   ```bash
   # Add user to docker group
   sudo usermod -aG docker $USER
   
   # Restart Docker service
   sudo systemctl restart docker
   ```

### Debug Mode
Enable debug logging for detailed troubleshooting:
```bash
export LOG_LEVEL=DEBUG
python tars-manager.py
```

## 📈 Performance Tuning

### Optimization Tips
1. **Resource Allocation**: Adjust container resource limits
2. **Network Optimization**: Use Docker networks for internal communication
3. **Storage Optimization**: Use volume mounts for persistent data
4. **Monitoring Interval**: Adjust based on infrastructure size

### Scaling
- **Horizontal Scaling**: Deploy multiple TARS Manager instances
- **Load Balancing**: Use NGINX for load distribution
- **High Availability**: Deploy across multiple nodes

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Documentation**: [TARS Documentation](../../../TARS_Comprehensive_Documentation/)
- **Issues**: [GitHub Issues](https://github.com/GuitarAlchemist/tars/issues)
- **Community**: [TARS Discord](https://discord.gg/tars)

---

**🚀 TARS Manager - Making infrastructure management intelligent with Gordon AI**
