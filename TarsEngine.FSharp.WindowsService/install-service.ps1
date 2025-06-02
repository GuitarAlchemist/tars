# TARS Windows Service Installation Script
# Installs and configures TARS as a Windows service for unattended operation

param(
    [string]$ServiceName = "TarsEngine",
    [string]$DisplayName = "TARS Autonomous Development Engine",
    [string]$Description = "TARS autonomous development platform with multi-agent orchestration and intelligent task execution",
    [string]$StartupType = "Automatic",
    [string]$ServiceAccount = "LocalSystem",
    [switch]$Force,
    [switch]$Uninstall
)

# Ensure running as Administrator
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "❌ This script requires Administrator privileges. Please run as Administrator." -ForegroundColor Red
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    exit 1
}

Write-Host "🤖 TARS Windows Service Installation" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════" -ForegroundColor Cyan

# Get script directory and service executable path
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ServiceExePath = Join-Path $ScriptDir "bin\Debug\net9.0\TarsEngine.FSharp.WindowsService.exe"
$ConfigPath = Join-Path $ScriptDir "Configuration\service.config.json"

# Check if uninstall was requested
if ($Uninstall) {
    Write-Host "🗑️ Uninstalling TARS Windows Service..." -ForegroundColor Yellow
    
    # Stop service if running
    $service = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
    if ($service) {
        if ($service.Status -eq "Running") {
            Write-Host "⏹️ Stopping TARS service..." -ForegroundColor Yellow
            Stop-Service -Name $ServiceName -Force
            Start-Sleep -Seconds 3
        }
        
        # Remove service
        Write-Host "🗑️ Removing service registration..." -ForegroundColor Yellow
        sc.exe delete $ServiceName
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ TARS Windows Service uninstalled successfully!" -ForegroundColor Green
        } else {
            Write-Host "❌ Failed to uninstall service. Error code: $LASTEXITCODE" -ForegroundColor Red
        }
    } else {
        Write-Host "ℹ️ TARS service is not installed." -ForegroundColor Blue
    }
    exit 0
}

# Validate service executable exists
if (-not (Test-Path $ServiceExePath)) {
    Write-Host "❌ Service executable not found: $ServiceExePath" -ForegroundColor Red
    Write-Host "Please build the project first:" -ForegroundColor Yellow
    Write-Host "  dotnet build TarsEngine.FSharp.WindowsService.fsproj" -ForegroundColor White
    exit 1
}

# Validate configuration file exists
if (-not (Test-Path $ConfigPath)) {
    Write-Host "⚠️ Configuration file not found: $ConfigPath" -ForegroundColor Yellow
    Write-Host "Creating default configuration..." -ForegroundColor Yellow
    
    # Create default configuration
    $defaultConfig = @{
        ServiceName = $ServiceName
        DisplayName = $DisplayName
        LogLevel = "Information"
        MaxConcurrentAgents = 20
        TaskExecutionTimeout = "00:30:00"
        HealthCheckInterval = "00:01:00"
        PerformanceCollectionInterval = "00:00:30"
        EnableSemanticSystem = $true
        EnableClosureFactory = $true
        ClosureDirectory = ".tars/closures"
        DataDirectory = ".tars/data"
        LogDirectory = ".tars/logs"
    } | ConvertTo-Json -Depth 10
    
    # Ensure configuration directory exists
    $configDir = Split-Path -Parent $ConfigPath
    if (-not (Test-Path $configDir)) {
        New-Item -ItemType Directory -Path $configDir -Force | Out-Null
    }
    
    $defaultConfig | Out-File -FilePath $ConfigPath -Encoding UTF8
    Write-Host "✅ Created default configuration: $ConfigPath" -ForegroundColor Green
}

# Check if service already exists
$existingService = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
if ($existingService -and -not $Force) {
    Write-Host "⚠️ Service '$ServiceName' already exists!" -ForegroundColor Yellow
    Write-Host "Use -Force to reinstall or -Uninstall to remove" -ForegroundColor Yellow
    exit 1
}

# Stop and remove existing service if Force is specified
if ($existingService -and $Force) {
    Write-Host "🔄 Reinstalling existing service..." -ForegroundColor Yellow
    
    if ($existingService.Status -eq "Running") {
        Write-Host "⏹️ Stopping existing service..." -ForegroundColor Yellow
        Stop-Service -Name $ServiceName -Force
        Start-Sleep -Seconds 3
    }
    
    Write-Host "🗑️ Removing existing service..." -ForegroundColor Yellow
    sc.exe delete $ServiceName | Out-Null
    Start-Sleep -Seconds 2
}

# Create TARS directories
Write-Host "📁 Creating TARS directories..." -ForegroundColor Blue
$tarsDirectories = @(
    ".tars",
    ".tars/closures",
    ".tars/closures/definitions",
    ".tars/closures/templates", 
    ".tars/closures/scripts",
    ".tars/closures/configs",
    ".tars/closures/examples",
    ".tars/closures/marketplace",
    ".tars/data",
    ".tars/logs",
    ".tars/sandbox"
)

foreach ($dir in $tarsDirectories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  ✅ Created: $dir" -ForegroundColor Green
    } else {
        Write-Host "  ℹ️ Exists: $dir" -ForegroundColor Blue
    }
}

# Install the service
Write-Host "🔧 Installing TARS Windows Service..." -ForegroundColor Blue
Write-Host "  Service Name: $ServiceName" -ForegroundColor White
Write-Host "  Display Name: $DisplayName" -ForegroundColor White
Write-Host "  Executable: $ServiceExePath" -ForegroundColor White
Write-Host "  Startup Type: $StartupType" -ForegroundColor White

# Create the service
$createResult = sc.exe create $ServiceName binPath= "`"$ServiceExePath`"" DisplayName= "$DisplayName" start= auto
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to create service. Error code: $LASTEXITCODE" -ForegroundColor Red
    Write-Host "Output: $createResult" -ForegroundColor Red
    exit 1
}

# Set service description
sc.exe description $ServiceName "$Description" | Out-Null

# Configure service recovery options
Write-Host "🔧 Configuring service recovery options..." -ForegroundColor Blue
sc.exe failure $ServiceName reset= 86400 actions= restart/5000/restart/10000/restart/30000 | Out-Null

# Set service to start automatically
sc.exe config $ServiceName start= auto | Out-Null

Write-Host "✅ TARS Windows Service installed successfully!" -ForegroundColor Green

# Ask if user wants to start the service now
$startNow = Read-Host "🚀 Start TARS service now? (y/N)"
if ($startNow -eq "y" -or $startNow -eq "Y" -or $startNow -eq "yes") {
    Write-Host "🚀 Starting TARS service..." -ForegroundColor Blue
    
    try {
        Start-Service -Name $ServiceName
        Start-Sleep -Seconds 3
        
        $service = Get-Service -Name $ServiceName
        if ($service.Status -eq "Running") {
            Write-Host "✅ TARS service started successfully!" -ForegroundColor Green
            Write-Host "🎯 Service Status: $($service.Status)" -ForegroundColor Green
        } else {
            Write-Host "⚠️ Service status: $($service.Status)" -ForegroundColor Yellow
        }
    }
    catch {
        Write-Host "❌ Failed to start service: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "Check the Windows Event Log for details" -ForegroundColor Yellow
    }
} else {
    Write-Host "ℹ️ Service installed but not started. Start manually when ready:" -ForegroundColor Blue
    Write-Host "  Start-Service -Name $ServiceName" -ForegroundColor White
}

Write-Host ""
Write-Host "🎉 TARS Installation Complete!" -ForegroundColor Green
Write-Host "═══════════════════════════════════" -ForegroundColor Green
Write-Host "Service Management Commands:" -ForegroundColor Cyan
Write-Host "  Start:   Start-Service -Name $ServiceName" -ForegroundColor White
Write-Host "  Stop:    Stop-Service -Name $ServiceName" -ForegroundColor White
Write-Host "  Status:  Get-Service -Name $ServiceName" -ForegroundColor White
Write-Host "  Restart: Restart-Service -Name $ServiceName" -ForegroundColor White
Write-Host ""
Write-Host "Configuration File: $ConfigPath" -ForegroundColor Cyan
Write-Host "Log Directory: .tars/logs" -ForegroundColor Cyan
Write-Host "Closure Directory: .tars/closures" -ForegroundColor Cyan
Write-Host ""
Write-Host "To uninstall: .\install-service.ps1 -Uninstall" -ForegroundColor Yellow
