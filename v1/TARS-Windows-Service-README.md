# 🤖 TARS Windows Service

**Autonomous Development Platform running as a Windows Service for unattended operation**

## 🎯 Overview

The TARS Windows Service enables the autonomous development platform to run continuously in the background as a Windows service, providing:

- **Unattended Operation**: Runs automatically on system startup
- **Background Processing**: Continuous task execution and monitoring
- **Service Management**: Easy installation, start, stop, and status monitoring
- **Autonomous Capabilities**: Multi-agent orchestration and semantic coordination
- **Health Monitoring**: Built-in health checks and auto-recovery

## 🚀 Quick Start

### 1. Build the Windows Service

```bash
# Build the Windows service
dotnet build TarsEngine.FSharp.WindowsService --configuration Release

# Build the service manager
dotnet build TarsServiceManager --configuration Release
```

### 2. Install as Windows Service

**⚠️ Administrator privileges required for service installation**

```bash
# Install TARS as a Windows service
tars service install

# Or reinstall if already exists
tars service install --force
```

### 3. Manage the Service

```bash
# Start the service
tars service start

# Check service status
tars service status

# Stop the service
tars service stop

# Uninstall the service
tars service uninstall
```

## 📋 Service Management Commands

### Installation Commands

| Command | Description |
|---------|-------------|
| `tars service install` | Install TARS as Windows service |
| `tars service install --force` | Reinstall service (removes existing first) |
| `tars service uninstall` | Remove TARS Windows service |

### Operation Commands

| Command | Description |
|---------|-------------|
| `tars service start` | Start the TARS service |
| `tars service stop` | Stop the TARS service |
| `tars service status` | Show detailed service status |
| `tars service help` | Show service management help |

## 🔧 Service Configuration

The service is configured through YAML files in the `Configuration/` directory:

- **`service.config.yaml`**: Main service settings
- **`agents.config.yaml`**: Agent system configuration  
- **`monitoring.config.yaml`**: Health monitoring and diagnostics

### Key Configuration Options

```yaml
service:
  name: "TarsService"
  displayName: "TARS Autonomous Development Platform"
  maxConcurrentTasks: 100
  healthCheckIntervalSeconds: 30
  autoRestart: true

agents:
  maxConcurrentAgents: 20
  agentTimeoutMinutes: 30
  
monitoring:
  enabled: true
  intervalSeconds: 60
  alerting: true
```

## 📊 Service Status Information

The `tars service status` command provides comprehensive information:

```
📊 TARS Windows Service Status
═════════════════════════════

🤖 Service Name: TarsService
📝 Display Name: TARS Autonomous Development Platform
🎯 Status: Running
🔧 Start Type: Automatic

✅ Service is running normally
```

## 🛠️ Troubleshooting

### Common Issues

#### Service Installation Fails
- **Cause**: Not running as Administrator
- **Solution**: Run Command Prompt or PowerShell as Administrator

#### Service Executable Not Found
- **Cause**: Windows service not built
- **Solution**: Build the service first:
  ```bash
  dotnet build TarsEngine.FSharp.WindowsService --configuration Release
  ```

#### Service Won't Start
- **Cause**: Configuration errors or missing dependencies
- **Solution**: Check Windows Event Log for detailed error messages

### Checking Service Logs

1. **Windows Event Log**: Check Application logs for TARS service events
2. **Service Logs**: Located in `logs/` directory (if file logging enabled)
3. **Console Output**: When running in development mode

### Manual Service Management

You can also manage the service using Windows built-in tools:

```bash
# Using sc.exe
sc query TarsService
sc start TarsService
sc stop TarsService

# Using PowerShell
Get-Service TarsService
Start-Service TarsService
Stop-Service TarsService
```

## 🏗️ Development Mode

For development and testing, you can run the service directly:

```bash
# Run service in console mode (not as Windows service)
dotnet run --project TarsEngine.FSharp.WindowsService
```

This runs the service in the foreground with console output for debugging.

## 🔐 Security Considerations

- **Administrator Rights**: Required for service installation/uninstallation
- **Service Account**: Runs under Local System account by default
- **Network Access**: Configure firewall rules if network features are enabled
- **File Permissions**: Ensure service has access to configuration and log directories

## 📁 File Structure

```
tars/
├── TarsEngine.FSharp.WindowsService/     # Main Windows service project
│   ├── Core/                             # Service core components
│   ├── Configuration/                    # YAML configuration files
│   └── bin/Release/net9.0/              # Built service executable
├── TarsServiceManager/                   # Service management CLI
│   └── bin/Release/net9.0/tars.exe      # Service manager executable
├── tars.cmd                              # Windows batch launcher
├── tars.ps1                              # PowerShell launcher
└── TARS-Windows-Service-README.md        # This documentation
```

## 🎯 Service Features

### Autonomous Capabilities
- **Multi-Agent Orchestration**: Coordinate multiple specialized agents
- **Semantic Task Routing**: Intelligent task assignment based on capabilities
- **Continuous Improvement**: Self-monitoring and optimization
- **Health Management**: Automatic recovery and error handling

### Monitoring & Diagnostics
- **Real-time Health Checks**: Monitor service and agent health
- **Performance Metrics**: Track task throughput and resource usage
- **Event Logging**: Comprehensive logging to Windows Event Log
- **Alerting**: Configurable alerts for critical issues

### Configuration Management
- **Hot Reload**: Update configuration without service restart
- **Environment-specific**: Support for different environments
- **Validation**: Configuration validation on startup
- **Defaults**: Sensible defaults for all settings

## 🚀 Next Steps

1. **Install the Service**: Follow the Quick Start guide
2. **Configure Settings**: Customize configuration files as needed
3. **Monitor Operation**: Use `tars service status` to monitor
4. **Explore Features**: Check logs and metrics for autonomous operation

For more advanced configuration and features, see the main TARS documentation.

---

**🤖 TARS - Autonomous Development Platform**  
*Continuously improving software development through intelligent automation*
