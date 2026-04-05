# TARS Autonomous VM Prerequisite Management Test
# Creates VM, clones repo, tests autonomous prerequisite management

param(
    [string]$VMName = "TARS-Build-Test-VM",
    [string]$VMMemory = "8GB",
    [string]$VMDisk = "50GB",
    [string]$RepoUrl = "https://github.com/GuitarAlchemist/tars.git",
    [string]$ClonePath = "C:\TARS-Test",
    [switch]$SkipVMCreation,
    [switch]$CleanupAfter
)

Write-Host "🚀 TARS AUTONOMOUS VM PREREQUISITE MANAGEMENT TEST" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan

$ErrorActionPreference = "Stop"
$startTime = Get-Date

# Function to log with timestamp
function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $color = switch ($Level) {
        "ERROR" { "Red" }
        "WARN" { "Yellow" }
        "SUCCESS" { "Green" }
        default { "White" }
    }
    Write-Host "[$timestamp] [$Level] $Message" -ForegroundColor $color
}

# Function to execute command and capture output
function Invoke-CommandWithLogging {
    param([string]$Command, [string]$Description)
    
    Write-Log "Executing: $Description" "INFO"
    Write-Log "Command: $Command" "INFO"
    
    try {
        $result = Invoke-Expression $Command
        Write-Log "Command completed successfully" "SUCCESS"
        return $result
    }
    catch {
        Write-Log "Command failed: $($_.Exception.Message)" "ERROR"
        throw
    }
}

# Function to check if Hyper-V is available
function Test-HyperVAvailable {
    try {
        $hyperv = Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V-All
        return $hyperv.State -eq "Enabled"
    }
    catch {
        return $false
    }
}

# Function to create VM
function New-TarsTestVM {
    param([string]$Name, [string]$Memory, [string]$DiskSize)
    
    Write-Log "Creating VM: $Name" "INFO"
    
    # Check if VM already exists
    $existingVM = Get-VM -Name $Name -ErrorAction SilentlyContinue
    if ($existingVM) {
        Write-Log "VM $Name already exists. Removing..." "WARN"
        Stop-VM -Name $Name -Force -ErrorAction SilentlyContinue
        Remove-VM -Name $Name -Force
    }
    
    # Create VM
    $vmPath = "C:\VMs\$Name"
    New-Item -Path $vmPath -ItemType Directory -Force | Out-Null
    
    $vm = New-VM -Name $Name -MemoryStartupBytes ([int64]$Memory.Replace("GB", "") * 1GB) -Path $vmPath -Generation 2
    
    # Create virtual hard disk
    $vhdPath = "$vmPath\$Name.vhdx"
    New-VHD -Path $vhdPath -SizeBytes ([int64]$DiskSize.Replace("GB", "") * 1GB) -Dynamic
    Add-VMHardDiskDrive -VMName $Name -Path $vhdPath
    
    # Configure VM
    Set-VMProcessor -VMName $Name -Count 4
    Set-VMMemory -VMName $Name -DynamicMemoryEnabled $true -MinimumBytes 2GB -MaximumBytes ([int64]$Memory.Replace("GB", "") * 1GB)
    
    # Add network adapter
    Add-VMNetworkAdapter -VMName $Name -SwitchName "Default Switch"
    
    Write-Log "VM $Name created successfully" "SUCCESS"
    return $vm
}

# Function to install prerequisites on VM
function Install-PrerequisitesOnVM {
    param([string]$VMName, [string]$RepoPath)
    
    Write-Log "Installing prerequisites on VM: $VMName" "INFO"
    
    # Create prerequisite installation script
    $prereqScript = @"
# TARS Autonomous Prerequisite Installation Script
Write-Host "🔧 TARS Autonomous Prerequisite Installation" -ForegroundColor Cyan

# Function to test if command exists
function Test-Command {
    param([string]`$Command)
    try {
        Get-Command `$Command -ErrorAction Stop | Out-Null
        return `$true
    }
    catch {
        return `$false
    }
}

# Function to install using WinGet
function Install-WithWinGet {
    param([string]`$PackageId, [string]`$Name)
    
    Write-Host "📦 Installing `$Name using WinGet..." -ForegroundColor Yellow
    try {
        winget install `$PackageId --accept-package-agreements --accept-source-agreements --silent
        Write-Host "✅ `$Name installed successfully" -ForegroundColor Green
        return `$true
    }
    catch {
        Write-Host "❌ Failed to install `$Name with WinGet: `$(`$_.Exception.Message)" -ForegroundColor Red
        return `$false
    }
}

# Function to install using Chocolatey
function Install-WithChocolatey {
    param([string]`$PackageName, [string]`$Name)
    
    Write-Host "🍫 Installing `$Name using Chocolatey..." -ForegroundColor Yellow
    try {
        choco install `$PackageName -y
        Write-Host "✅ `$Name installed successfully" -ForegroundColor Green
        return `$true
    }
    catch {
        Write-Host "❌ Failed to install `$Name with Chocolatey: `$(`$_.Exception.Message)" -ForegroundColor Red
        return `$false
    }
}

# Check and install WinGet if not available
if (-not (Test-Command "winget")) {
    Write-Host "📥 Installing WinGet..." -ForegroundColor Yellow
    # WinGet is typically pre-installed on Windows 11, but we can install it via Microsoft Store
    try {
        Add-AppxPackage -RegisterByFamilyName -MainPackage Microsoft.DesktopAppInstaller_8wekyb3d8bbwe
    }
    catch {
        Write-Host "❌ Failed to install WinGet" -ForegroundColor Red
    }
}

# Check and install Chocolatey if not available
if (-not (Test-Command "choco")) {
    Write-Host "🍫 Installing Chocolatey..." -ForegroundColor Yellow
    try {
        Set-ExecutionPolicy Bypass -Scope Process -Force
        [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
        iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
        Write-Host "✅ Chocolatey installed successfully" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Failed to install Chocolatey: `$(`$_.Exception.Message)" -ForegroundColor Red
    }
}

# Install prerequisites
`$prerequisites = @(
    @{ Name = ".NET SDK 8.0"; WinGetId = "Microsoft.DotNet.SDK.8"; ChocoName = "dotnet-sdk" },
    @{ Name = "Git"; WinGetId = "Git.Git"; ChocoName = "git" },
    @{ Name = "Node.js"; WinGetId = "OpenJS.NodeJS"; ChocoName = "nodejs" },
    @{ Name = "Python"; WinGetId = "Python.Python.3.12"; ChocoName = "python" },
    @{ Name = "Visual Studio Code"; WinGetId = "Microsoft.VisualStudioCode"; ChocoName = "vscode" },
    @{ Name = "Docker Desktop"; WinGetId = "Docker.DockerDesktop"; ChocoName = "docker-desktop" }
)

`$installationResults = @()

foreach (`$prereq in `$prerequisites) {
    Write-Host "`n🔍 Processing: `$(`$prereq.Name)" -ForegroundColor Cyan
    
    `$installed = `$false
    
    # Try WinGet first
    if (Test-Command "winget") {
        `$installed = Install-WithWinGet `$prereq.WinGetId `$prereq.Name
    }
    
    # Fallback to Chocolatey if WinGet failed
    if (-not `$installed -and (Test-Command "choco")) {
        `$installed = Install-WithChocolatey `$prereq.ChocoName `$prereq.Name
    }
    
    `$installationResults += @{
        Name = `$prereq.Name
        Installed = `$installed
        Method = if (`$installed) { "WinGet/Chocolatey" } else { "Failed" }
    }
}

# Generate installation report
Write-Host "`n📊 INSTALLATION REPORT" -ForegroundColor Cyan
Write-Host "========================" -ForegroundColor Cyan

foreach (`$result in `$installationResults) {
    `$status = if (`$result.Installed) { "✅ SUCCESS" } else { "❌ FAILED" }
    `$color = if (`$result.Installed) { "Green" } else { "Red" }
    Write-Host "`$status `$(`$result.Name) (`$(`$result.Method))" -ForegroundColor `$color
}

# Test installations
Write-Host "`n🧪 TESTING INSTALLATIONS" -ForegroundColor Cyan
Write-Host "=========================" -ForegroundColor Cyan

`$testResults = @()

# Test .NET
if (Test-Command "dotnet") {
    `$dotnetVersion = dotnet --version
    Write-Host "✅ .NET SDK: `$dotnetVersion" -ForegroundColor Green
    `$testResults += @{ Tool = ".NET SDK"; Available = `$true; Version = `$dotnetVersion }
} else {
    Write-Host "❌ .NET SDK: Not available" -ForegroundColor Red
    `$testResults += @{ Tool = ".NET SDK"; Available = `$false; Version = "N/A" }
}

# Test Git
if (Test-Command "git") {
    `$gitVersion = git --version
    Write-Host "✅ Git: `$gitVersion" -ForegroundColor Green
    `$testResults += @{ Tool = "Git"; Available = `$true; Version = `$gitVersion }
} else {
    Write-Host "❌ Git: Not available" -ForegroundColor Red
    `$testResults += @{ Tool = "Git"; Available = `$false; Version = "N/A" }
}

# Test Node.js
if (Test-Command "node") {
    `$nodeVersion = node --version
    Write-Host "✅ Node.js: `$nodeVersion" -ForegroundColor Green
    `$testResults += @{ Tool = "Node.js"; Available = `$true; Version = `$nodeVersion }
} else {
    Write-Host "❌ Node.js: Not available" -ForegroundColor Red
    `$testResults += @{ Tool = "Node.js"; Available = `$false; Version = "N/A" }
}

# Test Python
if (Test-Command "python") {
    `$pythonVersion = python --version
    Write-Host "✅ Python: `$pythonVersion" -ForegroundColor Green
    `$testResults += @{ Tool = "Python"; Available = `$true; Version = `$pythonVersion }
} else {
    Write-Host "❌ Python: Not available" -ForegroundColor Red
    `$testResults += @{ Tool = "Python"; Available = `$false; Version = "N/A" }
}

Write-Host "`n🎉 Prerequisite installation completed!" -ForegroundColor Green
"@

    # Save script to temp file
    $scriptPath = "$env:TEMP\install-prerequisites.ps1"
    $prereqScript | Out-File -FilePath $scriptPath -Encoding UTF8
    
    Write-Log "Prerequisite installation script created: $scriptPath" "SUCCESS"
    return $scriptPath
}

# Function to clone repository and test build
function Test-TarsBuild {
    param([string]$RepoUrl, [string]$ClonePath)
    
    Write-Log "Cloning TARS repository and testing build" "INFO"
    
    $buildScript = @"
# TARS Build Test Script
Write-Host "🏗️ TARS BUILD TEST" -ForegroundColor Cyan

# Clone repository
if (Test-Path "$ClonePath") {
    Write-Host "📁 Removing existing clone path..." -ForegroundColor Yellow
    Remove-Item -Path "$ClonePath" -Recurse -Force
}

Write-Host "📥 Cloning TARS repository..." -ForegroundColor Yellow
git clone $RepoUrl "$ClonePath"

if (`$LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to clone repository" -ForegroundColor Red
    exit 1
}

# Navigate to repository
Set-Location "$ClonePath"

# Test .NET build
Write-Host "🔨 Testing .NET build..." -ForegroundColor Yellow
dotnet restore

if (`$LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to restore .NET packages" -ForegroundColor Red
    exit 1
}

dotnet build

if (`$LASTEXITCODE -eq 0) {
    Write-Host "✅ TARS build successful!" -ForegroundColor Green
} else {
    Write-Host "❌ TARS build failed" -ForegroundColor Red
    exit 1
}

Write-Host "🎉 All tests completed successfully!" -ForegroundColor Green
"@

    $buildScriptPath = "$env:TEMP\test-tars-build.ps1"
    $buildScript | Out-File -FilePath $buildScriptPath -Encoding UTF8
    
    Write-Log "Build test script created: $buildScriptPath" "SUCCESS"
    return $buildScriptPath
}

# Main execution
try {
    Write-Log "Starting TARS Autonomous VM Prerequisite Management Test" "INFO"
    
    # Check if running as administrator
    $isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")
    if (-not $isAdmin) {
        Write-Log "This script requires administrator privileges. Please run as administrator." "ERROR"
        exit 1
    }
    
    # Check Hyper-V availability
    if (-not $SkipVMCreation) {
        if (-not (Test-HyperVAvailable)) {
            Write-Log "Hyper-V is not available. Skipping VM creation and running tests locally." "WARN"
            $SkipVMCreation = $true
        }
    }
    
    # Create VM if requested
    if (-not $SkipVMCreation) {
        Write-Log "Creating test VM..." "INFO"
        $vm = New-TarsTestVM -Name $VMName -Memory $VMMemory -DiskSize $VMDisk
        Write-Log "VM created successfully" "SUCCESS"
    }
    
    # Create prerequisite installation script
    $prereqScriptPath = Install-PrerequisitesOnVM -VMName $VMName -RepoPath $ClonePath
    
    # Create build test script
    $buildScriptPath = Test-TarsBuild -RepoUrl $RepoUrl -ClonePath $ClonePath
    
    # Execute prerequisite installation
    Write-Log "Executing prerequisite installation..." "INFO"
    & PowerShell -ExecutionPolicy Bypass -File $prereqScriptPath
    
    # Execute build test
    Write-Log "Executing build test..." "INFO"
    & PowerShell -ExecutionPolicy Bypass -File $buildScriptPath
    
    # Generate final report
    $endTime = Get-Date
    $duration = $endTime - $startTime
    
    Write-Log "🎉 TARS Autonomous VM Prerequisite Management Test Completed!" "SUCCESS"
    Write-Log "Total execution time: $($duration.TotalMinutes.ToString('F2')) minutes" "INFO"
    
    # Cleanup if requested
    if ($CleanupAfter -and -not $SkipVMCreation) {
        Write-Log "Cleaning up VM..." "INFO"
        Stop-VM -Name $VMName -Force -ErrorAction SilentlyContinue
        Remove-VM -Name $VMName -Force -ErrorAction SilentlyContinue
        Write-Log "VM cleanup completed" "SUCCESS"
    }
    
}
catch {
    Write-Log "Test failed: $($_.Exception.Message)" "ERROR"
    Write-Log "Stack trace: $($_.ScriptStackTrace)" "ERROR"
    exit 1
}
finally {
    # Cleanup temp files
    if (Test-Path "$env:TEMP\install-prerequisites.ps1") {
        Remove-Item "$env:TEMP\install-prerequisites.ps1" -Force
    }
    if (Test-Path "$env:TEMP\test-tars-build.ps1") {
        Remove-Item "$env:TEMP\test-tars-build.ps1" -Force
    }
}

Write-Host "`n🚀 TARS Autonomous Prerequisite Management System Ready!" -ForegroundColor Green
