# TARS Scripts Collection

This directory contains comprehensive scripts for TARS development, testing, automation, and deployment.

## 📁 **Scripts Directory Structure**

```
.tars/scripts/
├── README.md                           # This file
│
├── demo/                               # Demo and showcase scripts
│   ├── tars-demo.ps1                   # Main TARS functionality demo
│   └── metascript-showcase.ps1         # Metascript capabilities showcase
│
├── test/                               # Testing scripts
│   └── run-all-tests.ps1               # Comprehensive test runner
│
├── automation/                         # Automation and setup scripts
│   └── setup-development.ps1           # Development environment setup
│
├── build/                              # Build and compilation scripts
│   └── build-all.ps1                   # Comprehensive build script
│
├── deployment/                         # Deployment scripts (future)
│   └── (deployment scripts will go here)
│
├── utilities/                          # Utility and maintenance scripts
│   └── cleanup.ps1                     # Cleanup build artifacts and temp files
│
└── docker/                            # Docker-related scripts (future)
    └── (Docker scripts will go here)
```

## 🎯 **Script Categories**

### **🎮 Demo Scripts**
Scripts for demonstrating TARS capabilities and showcasing features.

#### **tars-demo.ps1**
Main TARS functionality demonstration script.

**Usage:**
```powershell
# Basic demo (version, help, intelligence, ML)
.\.tars\scripts\demo\tars-demo.ps1 -DemoType basic

# Full demo (all features including metascripts)
.\.tars\scripts\demo\tars-demo.ps1 -DemoType full

# Metascripts-only demo
.\.tars\scripts\demo\tars-demo.ps1 -DemoType metascripts

# Intelligence features demo
.\.tars\scripts\demo\tars-demo.ps1 -DemoType intelligence

# Verbose output
.\.tars\scripts\demo\tars-demo.ps1 -DemoType full -Verbose
```

#### **metascript-showcase.ps1**
Metascript capabilities showcase by category.

**Usage:**
```powershell
# Show all metascript categories
.\.tars\scripts\demo\metascript-showcase.ps1

# Show specific category
.\.tars\scripts\demo\metascript-showcase.ps1 -Category autonomous

# Interactive mode
.\.tars\scripts\demo\metascript-showcase.ps1 -Interactive
```

### **🧪 Test Scripts**
Scripts for testing TARS functionality and validating system health.

#### **run-all-tests.ps1**
Comprehensive test runner for all TARS components.

**Usage:**
```powershell
# Run all tests
.\.tars\scripts\test\run-all-tests.ps1

# Run specific test category
.\.tars\scripts\test\run-all-tests.ps1 -TestCategory cli
.\.tars\scripts\test\run-all-tests.ps1 -TestCategory metascripts
.\.tars\scripts\test\run-all-tests.ps1 -TestCategory services

# Skip build and run tests only
.\.tars\scripts\test\run-all-tests.ps1 -SkipBuild

# Verbose output
.\.tars\scripts\test\run-all-tests.ps1 -Verbose
```

**Test Categories:**
- `all` - All tests (default)
- `cli` - CLI command tests
- `metascripts` - Metascript discovery and execution tests
- `services` - Intelligence and ML service tests
- `structure` - Project structure validation tests

### **🤖 Automation Scripts**
Scripts for automating setup, configuration, and maintenance tasks.

#### **setup-development.ps1**
Development environment setup and configuration.

**Usage:**
```powershell
# Full development setup
.\.tars\scripts\automation\setup-development.ps1

# Skip .NET installation check
.\.tars\scripts\automation\setup-development.ps1 -SkipDotNet

# Skip project build
.\.tars\scripts\automation\setup-development.ps1 -SkipBuild

# Verbose output
.\.tars\scripts\automation\setup-development.ps1 -Verbose
```

**Features:**
- .NET SDK installation verification
- Project structure validation
- Dependency restoration
- Project compilation
- TARS installation testing
- Environment validation

### **🔨 Build Scripts**
Scripts for building, compiling, and packaging TARS projects.

#### **build-all.ps1**
Comprehensive build script with multiple options.

**Usage:**
```powershell
# Debug build (default)
.\.tars\scripts\build\build-all.ps1

# Release build
.\.tars\scripts\build\build-all.ps1 -Configuration Release

# Clean build
.\.tars\scripts\build\build-all.ps1 -Clean

# Build with tests
.\.tars\scripts\build\build-all.ps1 -Test

# Build with packaging
.\.tars\scripts\build\build-all.ps1 -Package

# Full build pipeline
.\.tars\scripts\build\build-all.ps1 -Configuration Release -Clean -Test -Package

# Skip restore
.\.tars\scripts\build\build-all.ps1 -Restore:$false
```

**Features:**
- Multi-project build coordination
- Configuration management (Debug/Release)
- Clean build support
- Dependency restoration
- Test execution integration
- Packaging and publishing
- Comprehensive build reporting

### **🛠️ Utility Scripts**
Scripts for maintenance, cleanup, and system utilities.

#### **cleanup.ps1**
Cleanup build artifacts, temporary files, and logs.

**Usage:**
```powershell
# Clean build artifacts
.\.tars\scripts\utilities\cleanup.ps1 -BuildArtifacts

# Clean temporary files
.\.tars\scripts\utilities\cleanup.ps1 -TempFiles

# Clean log files
.\.tars\scripts\utilities\cleanup.ps1 -Logs

# Clean everything
.\.tars\scripts\utilities\cleanup.ps1 -All

# Dry run (show what would be deleted)
.\.tars\scripts\utilities\cleanup.ps1 -All -DryRun

# Verbose output
.\.tars\scripts\utilities\cleanup.ps1 -All -Verbose
```

**Cleanup Targets:**
- Build artifacts (bin, obj, build directories)
- Temporary files (temp, cache, user settings)
- Log files (logs, build logs, MSBuild logs)
- NuGet cache
- Visual Studio cache files

## 🚀 **Quick Start Guide**

### **1. First-Time Setup**
```powershell
# Set up development environment
.\.tars\scripts\automation\setup-development.ps1

# Run basic demo to verify installation
.\.tars\scripts\demo\tars-demo.ps1 -DemoType basic
```

### **2. Development Workflow**
```powershell
# Clean build
.\.tars\scripts\build\build-all.ps1 -Clean

# Run tests
.\.tars\scripts\test\run-all-tests.ps1

# Demo new features
.\.tars\scripts\demo\tars-demo.ps1 -DemoType full
```

### **3. Maintenance**
```powershell
# Clean up build artifacts
.\.tars\scripts\utilities\cleanup.ps1 -BuildArtifacts

# Full cleanup
.\.tars\scripts\utilities\cleanup.ps1 -All
```

## 📊 **Script Features**

### **✅ Comprehensive Coverage**
- **Demo Scripts**: Showcase all TARS capabilities
- **Test Scripts**: Validate system functionality
- **Build Scripts**: Complete build pipeline
- **Automation Scripts**: Environment setup and configuration
- **Utility Scripts**: Maintenance and cleanup

### **✅ Professional Quality**
- **Error Handling**: Robust error detection and reporting
- **Logging**: Detailed output with color coding
- **Parameters**: Flexible command-line options
- **Validation**: Input validation and safety checks
- **Documentation**: Comprehensive usage examples

### **✅ Cross-Platform Support**
- **PowerShell Core**: Compatible with Windows, Linux, macOS
- **Path Handling**: Cross-platform path resolution
- **Command Detection**: Automatic tool detection
- **Environment Adaptation**: Adapts to different environments

### **✅ Integration Ready**
- **CI/CD Integration**: Suitable for automated pipelines
- **Exit Codes**: Proper exit code handling
- **Logging Standards**: Structured logging output
- **Configuration**: Configurable through parameters

## 🎯 **Common Use Cases**

### **Development**
```powershell
# Daily development workflow
.\.tars\scripts\build\build-all.ps1 -Test
.\.tars\scripts\demo\tars-demo.ps1 -DemoType basic
```

### **Testing**
```powershell
# Comprehensive testing
.\.tars\scripts\test\run-all-tests.ps1 -Verbose
.\.tars\scripts\demo\metascript-showcase.ps1
```

### **Deployment Preparation**
```powershell
# Release build with packaging
.\.tars\scripts\build\build-all.ps1 -Configuration Release -Clean -Test -Package
```

### **Maintenance**
```powershell
# System cleanup
.\.tars\scripts\utilities\cleanup.ps1 -All
.\.tars\scripts\automation\setup-development.ps1
```

## 🔧 **Script Development Guidelines**

### **Standards**
- **PowerShell**: Use PowerShell Core for cross-platform compatibility
- **Parameters**: Support common parameters (Verbose, WhatIf, etc.)
- **Error Handling**: Implement comprehensive error handling
- **Logging**: Use consistent color-coded output
- **Documentation**: Include usage examples and parameter descriptions

### **Best Practices**
- **Validation**: Validate inputs and prerequisites
- **Safety**: Implement dry-run modes for destructive operations
- **Modularity**: Create reusable functions
- **Performance**: Optimize for speed and efficiency
- **Compatibility**: Ensure cross-platform compatibility

## 📈 **Future Enhancements**

### **Planned Scripts**
- **Deployment Scripts**: Automated deployment to various environments
- **Docker Scripts**: Container build and orchestration
- **Performance Scripts**: Performance testing and benchmarking
- **Security Scripts**: Security scanning and validation
- **Monitoring Scripts**: System monitoring and health checks

### **Integration Opportunities**
- **GitHub Actions**: CI/CD pipeline integration
- **Azure DevOps**: Build and release pipeline integration
- **Docker**: Container orchestration scripts
- **Kubernetes**: Deployment and scaling scripts

---

**TARS Scripts Collection - Comprehensive automation for TARS development and deployment.**  
*Professional-grade scripts for efficient TARS project management.*
