# 🚀 TARS Swarm Demo

A beautiful command-line demonstration of the TARS (The Autonomous Reasoning System) swarm capabilities using Spectre.Console for rich terminal UI.

## ✨ Features

- **Beautiful CLI Interface**: Rich terminal UI with colors, tables, and interactive menus
- **Container Status Display**: Visual representation of TARS swarm containers
- **Health Testing**: Comprehensive swarm health checks and diagnostics
- **Performance Monitoring**: Real-time performance metrics with visual indicators
- **Command Execution**: Simulated command execution across the swarm
- **Interactive Mode**: Full interactive menu system for exploration

## 🎯 Demo Modes

### 1. Simple Demo (Default)
```bash
dotnet run
```
Runs a quick demonstration showing swarm status and test results.

### 2. Interactive Demo
```bash
dotnet run -- interactive
# or
dotnet run -- -i
```
Launches an interactive menu where you can explore different features:
- 📊 View Swarm Status
- 🧪 Run Swarm Tests  
- 📈 Performance Monitor
- 📋 Execute Commands
- 🔄 Restart Containers (Simulated)

### 3. Status Only
```bash
dotnet run -- status
```
Shows only the current swarm container status.

### 4. Test Only
```bash
dotnet run -- test
```
Runs only the swarm health tests.

### 5. Performance Monitor
```bash
dotnet run -- monitor
```
Displays real-time performance monitoring with visual indicators.

## 🛠️ Quick Start Scripts

### Windows Batch Files
- `run-tars-swarm-demo.cmd` - Simple demo launcher
- `run-tars-swarm-interactive.cmd` - Interactive demo launcher
- `run-tars-swarm-status.cmd` - Status check only
- `run-tars-swarm-monitor.cmd` - Performance monitor only

### PowerShell (Cross-platform)
```powershell
# Simple demo
.\run-tars-swarm-demo.ps1

# Interactive mode
.\run-tars-swarm-demo.ps1 interactive

# Status check
.\run-tars-swarm-demo.ps1 status

# Performance monitor
.\run-tars-swarm-demo.ps1 monitor
```

## 📊 What You'll See

### Container Status Table
```
┌───────────────┬────────────┬────────┬───────────┬─────────────────┐
│ Container     │ Status     │ Uptime │ Ports     │ Role            │
├───────────────┼────────────┼────────┼───────────┼─────────────────┤
│ tars-alpha    │ 🟢 Running │ 2h 15m │ 8080-8081 │ 🎯 Primary      │
│ tars-beta     │ 🟢 Running │ 2h 14m │ 8082-8083 │ 🔄 Secondary    │
│ tars-gamma    │ 🟢 Running │ 2h 13m │ 8084-8085 │ 🧪 Experimental │
│ tars-delta    │ 🟢 Running │ 2h 12m │ 8086-8087 │ 🔍 QA           │
│ tars-postgres │ 🟢 Running │ 2h 16m │ 5432      │ 🗄️ Database     │
│ tars-redis    │ 🟢 Running │ 2h 16m │ 6379      │ ⚡ Cache        │
└───────────────┴────────────┴────────┴───────────┴─────────────────┘
```

### Health Test Results
```
┌───────────────────────────────┬─────────┐
│ Test                          │ Result  │
├───────────────────────────────┼─────────┤
│ Container Health Check        │ ✅ PASS │
│ Network Connectivity          │ ✅ PASS │
│ TARS CLI Availability         │ ✅ PASS │
│ Metascript Execution          │ ✅ PASS │
│ Inter-Container Communication │ ❌ FAIL │
│ Load Balancing                │ ✅ PASS │
└───────────────────────────────┴─────────┘
```

### Performance Monitoring
```
Iteration 1/5
CPU: 45% | Memory: 67% | Network: 23% | Disk: 15%
CPU Load: ████
```

## 🏗️ Building

```bash
# Build the project
dotnet build TarsSwarmDemo.fsproj

# Run directly
dotnet run --project TarsSwarmDemo.fsproj

# Build and run with arguments
dotnet run --project TarsSwarmDemo.fsproj -- interactive
```

## 🔧 Requirements

- .NET 9.0 or later
- Windows, macOS, or Linux
- Terminal with Unicode support for best experience

## 📦 Dependencies

- **Spectre.Console** - Rich terminal UI framework
- **F# 9.0** - Functional programming language

## 🎨 Features Demonstrated

1. **Rich Terminal UI**: Beautiful tables, colors, and formatting
2. **Interactive Menus**: Arrow key navigation and selection
3. **Real-time Updates**: Dynamic content with progress indicators
4. **Cross-platform**: Works on Windows, macOS, and Linux
5. **Multiple Modes**: Different ways to run and explore the demo
6. **Simulated Operations**: Realistic container management simulation

## 🚀 Integration with TARS

This demo showcases the visual capabilities that could be integrated into the actual TARS CLI for:
- Real container monitoring
- Actual health checks
- Live performance metrics
- Interactive swarm management
- Command execution across containers

The demo serves as a proof-of-concept for beautiful CLI interfaces in the TARS ecosystem.
