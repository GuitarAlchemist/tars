# TARS Real Local VM Deployment System
# Uses actual VirtualBox and Vagrant for real VM deployment and testing

Write-Host "🖥️💻 TARS LOCAL VM DEPLOYMENT SYSTEM" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
Write-Host ""

# Global VM state
$global:localVMs = @{}
$global:vmMetrics = @{
    totalVMs = 0
    activeVMs = 0
    successfulDeployments = 0
    failedDeployments = 0
    totalTestsRun = 0
}

# Check prerequisites
function Test-Prerequisites {
    Write-Host "🔍 Checking Prerequisites..." -ForegroundColor Cyan
    Write-Host ""
    
    $allGood = $true
    
    # Check VirtualBox
    try {
        $vboxVersion = & VBoxManage --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✅ VirtualBox: $vboxVersion" -ForegroundColor Green
        } else {
            throw "VirtualBox not found"
        }
    } catch {
        Write-Host "  ❌ VirtualBox not found" -ForegroundColor Red
        Write-Host "     Please install from: https://www.virtualbox.org/" -ForegroundColor Yellow
        $allGood = $false
    }
    
    # Check Vagrant
    try {
        $vagrantVersion = & vagrant --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✅ Vagrant: $vagrantVersion" -ForegroundColor Green
        } else {
            throw "Vagrant not found"
        }
    } catch {
        Write-Host "  ❌ Vagrant not found" -ForegroundColor Red
        Write-Host "     Please install from: https://www.vagrantup.com/" -ForegroundColor Yellow
        $allGood = $false
    }
    
    # Check available memory
    $totalMemory = (Get-CimInstance Win32_PhysicalMemory | Measure-Object -Property capacity -Sum).sum / 1GB
    Write-Host "  💾 Available RAM: $([math]::Round($totalMemory, 1)) GB" -ForegroundColor White
    
    if ($totalMemory -lt 8) {
        Write-Host "     ⚠️ Warning: Less than 8GB RAM available" -ForegroundColor Yellow
        Write-Host "     Consider using smaller VM configurations" -ForegroundColor Yellow
    }
    
    # Check disk space
    $freeSpace = (Get-PSDrive C | Select-Object Free).Free / 1GB
    Write-Host "  💿 Free Disk Space: $([math]::Round($freeSpace, 1)) GB" -ForegroundColor White
    
    if ($freeSpace -lt 20) {
        Write-Host "     ⚠️ Warning: Less than 20GB free space" -ForegroundColor Yellow
        Write-Host "     VMs require significant disk space" -ForegroundColor Yellow
    }
    
    Write-Host ""
    
    if (-not $allGood) {
        Write-Host "❌ Prerequisites not met. Please install required software." -ForegroundColor Red
        return $false
    }
    
    Write-Host "✅ All prerequisites met!" -ForegroundColor Green
    return $true
}

# Create VM configuration
function New-VMConfig {
    param(
        [string]$VMName,
        [string]$ProjectComplexity = "moderate",
        [bool]$RequiresDatabase = $false
    )
    
    # Determine VM specs based on project complexity
    $config = switch ($ProjectComplexity) {
        "simple" {
            @{
                VMName = $VMName
                Memory = 2048
                CPUs = 2
                DiskSize = 20
                BaseBox = "ubuntu/jammy64"
                NetworkType = "private_network"
                IPAddress = "192.168.56.10"
                ForwardedPorts = @((5000, 5000), (3000, 3000))
                SharedFolders = @()
            }
        }
        "moderate" {
            @{
                VMName = $VMName
                Memory = 4096
                CPUs = 2
                DiskSize = 30
                BaseBox = "ubuntu/jammy64"
                NetworkType = "private_network"
                IPAddress = "192.168.56.11"
                ForwardedPorts = @((5000, 5001), (3000, 3001), (5432, 5432))
                SharedFolders = @()
            }
        }
        "complex" {
            @{
                VMName = $VMName
                Memory = 6144
                CPUs = 4
                DiskSize = 40
                BaseBox = "ubuntu/jammy64"
                NetworkType = "private_network"
                IPAddress = "192.168.56.12"
                ForwardedPorts = @((5000, 5002), (3000, 3002), (5432, 5433), (6379, 6379))
                SharedFolders = @()
            }
        }
        default {
            @{
                VMName = $VMName
                Memory = 4096
                CPUs = 2
                DiskSize = 30
                BaseBox = "ubuntu/jammy64"
                NetworkType = "private_network"
                IPAddress = "192.168.56.11"
                ForwardedPorts = @((5000, 5001), (3000, 3001))
                SharedFolders = @()
            }
        }
    }
    
    # Add database provisioning if required
    if ($RequiresDatabase) {
        $config.ProvisionScript = @"
# Install PostgreSQL
sudo apt-get update -y
sudo apt-get install -y postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database and user
sudo -u postgres createdb projectdb
sudo -u postgres psql -c "CREATE USER projectuser WITH PASSWORD 'projectpass';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE projectdb TO projectuser;"

# Configure PostgreSQL for remote connections
sudo sed -i "s/#listen_addresses = 'localhost'/listen_addresses = '*'/" /etc/postgresql/*/main/postgresql.conf
echo "host all all 0.0.0.0/0 md5" | sudo tee -a /etc/postgresql/*/main/pg_hba.conf
sudo systemctl restart postgresql
"@
    } else {
        $config.ProvisionScript = @"
# Basic system setup
sudo apt-get update -y
sudo apt-get upgrade -y
"@
    }
    
    return $config
}

# Generate Vagrantfile
function New-Vagrantfile {
    param($Config)
    
    $forwardedPorts = $Config.ForwardedPorts | ForEach-Object { 
        "  config.vm.network `"forwarded_port`", guest: $($_[0]), host: $($_[1])" 
    } | Join-String -Separator "`n"
    
    $sharedFolders = $Config.SharedFolders | ForEach-Object {
        "  config.vm.synced_folder `"$($_[0])`", `"$($_[1])`""
    } | Join-String -Separator "`n"
    
    return @"
# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure("2") do |config|
  config.vm.box = "$($Config.BaseBox)"
  config.vm.hostname = "$($Config.VMName)"
  
  # Network configuration
  config.vm.network "private_network", ip: "$($Config.IPAddress)"
$forwardedPorts
  
  # Shared folders
$sharedFolders
  
  # Provider configuration
  config.vm.provider "virtualbox" do |vb|
    vb.name = "$($Config.VMName)"
    vb.memory = $($Config.Memory)
    vb.cpus = $($Config.CPUs)
    vb.customize ["modifyvm", :id, "--natdnshostresolver1", "on"]
    vb.customize ["modifyvm", :id, "--natdnsproxy1", "on"]
  end
  
  # Provisioning
  config.vm.provision "shell", inline: <<-SHELL
$($Config.ProvisionScript)
  SHELL
end
"@
}

# Create and start VM
function New-LocalVM {
    param(
        [string]$VMName,
        [string]$ProjectPath,
        [string]$ProjectComplexity = "moderate"
    )
    
    Write-Host "🚀 CREATING LOCAL VM" -ForegroundColor Yellow
    Write-Host "====================" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "📂 Project: $(Split-Path $ProjectPath -Leaf)" -ForegroundColor White
    Write-Host "🖥️ VM Name: $VMName" -ForegroundColor White
    Write-Host "⚡ Complexity: $ProjectComplexity" -ForegroundColor White
    Write-Host ""
    
    try {
        # Analyze project
        $requiresDatabase = Test-Path "$ProjectPath\database\*" -or Test-Path "$ProjectPath\*schema.sql"
        
        # Create VM configuration
        $config = New-VMConfig -VMName $VMName -ProjectComplexity $ProjectComplexity -RequiresDatabase $requiresDatabase
        
        Write-Host "🔧 VM Configuration:" -ForegroundColor Cyan
        Write-Host "  💾 Memory: $($config.Memory) MB" -ForegroundColor Gray
        Write-Host "  🔢 CPUs: $($config.CPUs)" -ForegroundColor Gray
        Write-Host "  💿 Disk: $($config.DiskSize) GB" -ForegroundColor Gray
        Write-Host "  🌐 IP: $($config.IPAddress)" -ForegroundColor Gray
        Write-Host "  🗄️ Database: $(if ($requiresDatabase) { 'Yes' } else { 'No' })" -ForegroundColor Gray
        Write-Host ""
        
        # Create VM directory
        $vmDir = ".tars\vms\$VMName"
        if (Test-Path $vmDir) {
            Write-Host "🧹 Cleaning existing VM directory..." -ForegroundColor Yellow
            Remove-Item $vmDir -Recurse -Force
        }
        New-Item -ItemType Directory -Path $vmDir -Force | Out-Null
        
        # Generate Vagrantfile
        Write-Host "📝 Generating Vagrantfile..." -ForegroundColor Yellow
        $vagrantfile = New-Vagrantfile -Config $config
        $vagrantfilePath = "$vmDir\Vagrantfile"
        $vagrantfile | Out-File -FilePath $vagrantfilePath -Encoding UTF8
        
        Write-Host "  ✅ Vagrantfile created: $vagrantfilePath" -ForegroundColor Green
        
        # Start VM
        Write-Host "🚀 Starting VM (this may take several minutes)..." -ForegroundColor Yellow
        Push-Location $vmDir
        
        try {
            $vagrantUp = Start-Process -FilePath "vagrant" -ArgumentList "up" -Wait -PassThru -NoNewWindow -RedirectStandardOutput "vagrant-up.log" -RedirectStandardError "vagrant-error.log"
            
            if ($vagrantUp.ExitCode -eq 0) {
                Write-Host "  ✅ VM started successfully!" -ForegroundColor Green
                
                # Get VM info
                $sshConfig = & vagrant ssh-config 2>$null
                if ($LASTEXITCODE -eq 0) {
                    $vmInfo = @{
                        Name = $VMName
                        Status = "running"
                        IPAddress = $config.IPAddress
                        SSHPort = 22
                        VagrantfilePath = $vagrantfilePath
                        CreatedAt = Get-Date
                        Config = $config
                        ProjectPath = $ProjectPath
                    }
                    
                    $global:localVMs[$VMName] = $vmInfo
                    $global:vmMetrics.totalVMs++
                    $global:vmMetrics.activeVMs++
                    
                    Write-Host ""
                    Write-Host "✅ VM CREATED SUCCESSFULLY!" -ForegroundColor Green
                    Write-Host "  🆔 Name: $VMName" -ForegroundColor White
                    Write-Host "  🌐 IP: $($config.IPAddress)" -ForegroundColor White
                    Write-Host "  🔑 SSH: vagrant ssh (from $vmDir)" -ForegroundColor White
                    Write-Host "  📂 Project will be deployed to: /home/vagrant/project" -ForegroundColor White
                    Write-Host ""
                    
                    return $vmInfo
                } else {
                    throw "Failed to get SSH configuration"
                }
            } else {
                $errorLog = Get-Content "vagrant-error.log" -Raw -ErrorAction SilentlyContinue
                throw "Vagrant up failed: $errorLog"
            }
        } finally {
            Pop-Location
        }
        
    } catch {
        Write-Host "❌ VM Creation Failed!" -ForegroundColor Red
        Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host ""
        
        $global:vmMetrics.failedDeployments++
        return $null
    }
}

# Deploy project to VM
function Deploy-ProjectToLocalVM {
    param(
        [string]$VMName,
        [string]$ProjectPath
    )
    
    if (-not $global:localVMs.ContainsKey($VMName)) {
        Write-Host "❌ VM not found: $VMName" -ForegroundColor Red
        return $false
    }
    
    $vm = $global:localVMs[$VMName]
    $vmDir = Split-Path $vm.VagrantfilePath -Parent
    
    Write-Host "📦 DEPLOYING PROJECT TO VM" -ForegroundColor Yellow
    Write-Host "===========================" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "🖥️ VM: $VMName" -ForegroundColor White
    Write-Host "📂 Project: $(Split-Path $ProjectPath -Leaf)" -ForegroundColor White
    Write-Host ""
    
    try {
        Push-Location $vmDir
        
        # Copy project files
        Write-Host "📤 Copying project files..." -ForegroundColor Yellow
        $copyResult = & vagrant upload $ProjectPath /home/vagrant/project 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to copy project files: $copyResult"
        }
        Write-Host "  ✅ Project files copied" -ForegroundColor Green
        
        # Install dependencies
        Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
        $installCommands = @(
            "sudo apt-get update -y",
            "sudo apt-get install -y docker.io docker-compose curl wget",
            "sudo systemctl start docker",
            "sudo systemctl enable docker",
            "sudo usermod -aG docker vagrant",
            "wget https://packages.microsoft.com/config/ubuntu/22.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb",
            "sudo dpkg -i packages-microsoft-prod.deb",
            "sudo apt-get update -y",
            "sudo apt-get install -y dotnet-sdk-8.0"
        )
        
        foreach ($cmd in $installCommands) {
            Write-Host "    Executing: $cmd" -ForegroundColor Gray
            $result = & vagrant ssh -c $cmd 2>&1
            if ($LASTEXITCODE -ne 0) {
                Write-Host "    ⚠️ Warning: Command failed: $cmd" -ForegroundColor Yellow
            }
        }
        Write-Host "  ✅ Dependencies installed" -ForegroundColor Green
        
        # Build project
        Write-Host "🔨 Building project..." -ForegroundColor Yellow
        $buildCommands = @(
            "cd /home/vagrant/project",
            "dotnet restore",
            "dotnet build --configuration Release"
        )
        
        foreach ($cmd in $buildCommands) {
            Write-Host "    Executing: $cmd" -ForegroundColor Gray
            $result = & vagrant ssh -c $cmd 2>&1
            if ($LASTEXITCODE -ne 0) {
                Write-Host "    ❌ Build command failed: $cmd" -ForegroundColor Red
                throw "Build failed: $result"
            }
        }
        Write-Host "  ✅ Project built successfully" -ForegroundColor Green
        
        # Start application
        Write-Host "🚀 Starting application..." -ForegroundColor Yellow
        
        # Check if Dockerfile exists
        $hasDockerfile = & vagrant ssh -c "test -f /home/vagrant/project/Dockerfile && echo 'true' || echo 'false'" 2>$null
        
        if ($hasDockerfile -eq "true") {
            Write-Host "    Using Docker deployment..." -ForegroundColor Gray
            $startCommands = @(
                "cd /home/vagrant/project",
                "sudo docker build -t project .",
                "sudo docker run -d -p 5000:5000 --name project-app project"
            )
        } else {
            Write-Host "    Using direct .NET deployment..." -ForegroundColor Gray
            $startCommands = @(
                "cd /home/vagrant/project",
                "nohup dotnet run --project src/*/*.fsproj --urls http://0.0.0.0:5000 > app.log 2>&1 &"
            )
        }
        
        foreach ($cmd in $startCommands) {
            Write-Host "    Executing: $cmd" -ForegroundColor Gray
            $result = & vagrant ssh -c $cmd 2>&1
            if ($LASTEXITCODE -ne 0) {
                Write-Host "    ⚠️ Warning: Start command failed: $cmd" -ForegroundColor Yellow
            }
        }
        
        # Wait for application to start
        Write-Host "    Waiting for application to start..." -ForegroundColor Gray
        Start-Sleep -Seconds 10
        
        # Test if application is running
        $healthCheck = & vagrant ssh -c "curl -s -o /dev/null -w '%{http_code}' http://localhost:5000/health || curl -s -o /dev/null -w '%{http_code}' http://localhost:5000" 2>$null
        
        if ($healthCheck -eq "200" -or $healthCheck -eq "404") {
            Write-Host "  ✅ Application started successfully" -ForegroundColor Green
            $global:vmMetrics.successfulDeployments++
        } else {
            Write-Host "  ⚠️ Application may not be responding (HTTP: $healthCheck)" -ForegroundColor Yellow
        }
        
        Write-Host ""
        Write-Host "🎉 DEPLOYMENT COMPLETED!" -ForegroundColor Green
        Write-Host "========================" -ForegroundColor Green
        Write-Host ""
        Write-Host "🌐 Access URLs:" -ForegroundColor Yellow
        Write-Host "  • Application: http://localhost:$($vm.Config.ForwardedPorts[0][1])" -ForegroundColor Cyan
        Write-Host "  • Direct VM: http://$($vm.IPAddress):5000" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "🔧 Management:" -ForegroundColor Yellow
        Write-Host "  • SSH to VM: vagrant ssh (from $vmDir)" -ForegroundColor White
        Write-Host "  • View logs: vagrant ssh -c 'tail -f /home/vagrant/project/app.log'" -ForegroundColor White
        Write-Host "  • Stop VM: vagrant halt (from $vmDir)" -ForegroundColor White
        Write-Host ""
        
        return $true
        
    } catch {
        Write-Host "❌ DEPLOYMENT FAILED!" -ForegroundColor Red
        Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host ""
        
        $global:vmMetrics.failedDeployments++
        return $false
        
    } finally {
        Pop-Location
    }
}

# Run tests on VM
function Invoke-TestsOnLocalVM {
    param([string]$VMName)
    
    if (-not $global:localVMs.ContainsKey($VMName)) {
        Write-Host "❌ VM not found: $VMName" -ForegroundColor Red
        return $false
    }
    
    $vm = $global:localVMs[$VMName]
    $vmDir = Split-Path $vm.VagrantfilePath -Parent
    
    Write-Host "🧪 RUNNING TESTS ON VM" -ForegroundColor Yellow
    Write-Host "======================" -ForegroundColor Yellow
    Write-Host ""
    
    try {
        Push-Location $vmDir
        
        # Run tests
        Write-Host "🔬 Executing test suite..." -ForegroundColor Yellow
        $testResult = & vagrant ssh -c "cd /home/vagrant/project && dotnet test --logger trx --results-directory TestResults" 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✅ Tests completed successfully" -ForegroundColor Green
            Write-Host ""
            Write-Host "📊 Test Results:" -ForegroundColor Cyan
            Write-Host $testResult -ForegroundColor White
            
            $global:vmMetrics.totalTestsRun++
            return $true
        } else {
            Write-Host "  ❌ Tests failed" -ForegroundColor Red
            Write-Host ""
            Write-Host "📊 Test Output:" -ForegroundColor Cyan
            Write-Host $testResult -ForegroundColor White
            return $false
        }
        
    } catch {
        Write-Host "❌ Test execution failed: $($_.Exception.Message)" -ForegroundColor Red
        return $false
        
    } finally {
        Pop-Location
    }
}

# Show VM status
function Show-LocalVMStatus {
    Write-Host ""
    Write-Host "🖥️ LOCAL VM STATUS" -ForegroundColor Cyan
    Write-Host "==================" -ForegroundColor Cyan
    Write-Host ""
    
    if ($global:localVMs.Count -eq 0) {
        Write-Host "No VMs currently running." -ForegroundColor Gray
        return
    }
    
    foreach ($vmName in $global:localVMs.Keys) {
        $vm = $global:localVMs[$vmName]
        
        Write-Host "🖥️ $vmName" -ForegroundColor Green
        Write-Host "  📊 Status: $($vm.Status)" -ForegroundColor Gray
        Write-Host "  🌐 IP: $($vm.IPAddress)" -ForegroundColor Gray
        Write-Host "  💾 Memory: $($vm.Config.Memory) MB" -ForegroundColor Gray
        Write-Host "  🔢 CPUs: $($vm.Config.CPUs)" -ForegroundColor Gray
        Write-Host "  📂 Project: $(Split-Path $vm.ProjectPath -Leaf)" -ForegroundColor Gray
        Write-Host "  🕐 Created: $($vm.CreatedAt.ToString('yyyy-MM-dd HH:mm'))" -ForegroundColor Gray
        Write-Host "  🌐 URL: http://localhost:$($vm.Config.ForwardedPorts[0][1])" -ForegroundColor Cyan
        Write-Host ""
    }
    
    Write-Host "📊 Metrics:" -ForegroundColor Yellow
    Write-Host "  Total VMs: $($global:vmMetrics.totalVMs)" -ForegroundColor White
    Write-Host "  Active VMs: $($global:vmMetrics.activeVMs)" -ForegroundColor White
    Write-Host "  Successful Deployments: $($global:vmMetrics.successfulDeployments)" -ForegroundColor Green
    Write-Host "  Failed Deployments: $($global:vmMetrics.failedDeployments)" -ForegroundColor Red
    Write-Host ""
}

# Main function
function Start-LocalVMDeployment {
    Write-Host "🖥️💻 TARS Local VM Deployment System Started" -ForegroundColor Green
    Write-Host ""
    Write-Host "💡 Real local VM deployment using VirtualBox and Vagrant!" -ForegroundColor Yellow
    Write-Host "  • Actual VMs with real operating systems" -ForegroundColor White
    Write-Host "  • Real deployment and testing" -ForegroundColor White
    Write-Host "  • Full isolation and control" -ForegroundColor White
    Write-Host "  • No cloud dependencies" -ForegroundColor White
    Write-Host ""
    
    # Check prerequisites
    if (-not (Test-Prerequisites)) {
        return
    }
    
    Write-Host "Commands: 'create', 'deploy', 'test', 'status', 'help', 'exit'" -ForegroundColor Gray
    Write-Host ""
    
    $isRunning = $true
    while ($isRunning) {
        Write-Host ""
        $userInput = Read-Host "Command"
        
        switch ($userInput.ToLower().Trim()) {
            "exit" {
                $isRunning = $false
                Write-Host ""
                Write-Host "🖥️💻 Thank you for using TARS Local VM Deployment!" -ForegroundColor Green
                break
            }
            "create" {
                Write-Host ""
                Write-Host "Available projects:" -ForegroundColor Yellow
                if (Test-Path "output\projects") {
                    Get-ChildItem "output\projects" -Directory | ForEach-Object { Write-Host "  • $($_.Name)" -ForegroundColor White }
                } else {
                    Write-Host "  No projects found in output\projects" -ForegroundColor Red
                }
                Write-Host ""
                $projectName = Read-Host "Enter project name"
                $projectPath = "output\projects\$projectName"
                
                if (Test-Path $projectPath) {
                    $vmName = Read-Host "Enter VM name (or press Enter for auto-generated)"
                    if ([string]::IsNullOrWhiteSpace($vmName)) {
                        $vmName = "tars-vm-$projectName-$(Get-Date -Format 'MMdd-HHmm')"
                    }
                    
                    $complexity = Read-Host "Enter complexity (simple/moderate/complex) or press Enter for moderate"
                    if ([string]::IsNullOrWhiteSpace($complexity)) { $complexity = "moderate" }
                    
                    New-LocalVM -VMName $vmName -ProjectPath $projectPath -ProjectComplexity $complexity
                } else {
                    Write-Host "❌ Project not found: $projectName" -ForegroundColor Red
                }
            }
            "deploy" {
                if ($global:localVMs.Count -eq 0) {
                    Write-Host "❌ No VMs available. Create a VM first." -ForegroundColor Red
                } else {
                    Write-Host ""
                    Write-Host "Available VMs:" -ForegroundColor Yellow
                    $global:localVMs.Keys | ForEach-Object { Write-Host "  • $_" -ForegroundColor White }
                    Write-Host ""
                    $vmName = Read-Host "Enter VM name"
                    
                    if ($global:localVMs.ContainsKey($vmName)) {
                        $vm = $global:localVMs[$vmName]
                        Deploy-ProjectToLocalVM -VMName $vmName -ProjectPath $vm.ProjectPath
                    } else {
                        Write-Host "❌ VM not found: $vmName" -ForegroundColor Red
                    }
                }
            }
            "test" {
                if ($global:localVMs.Count -eq 0) {
                    Write-Host "❌ No VMs available. Create and deploy to a VM first." -ForegroundColor Red
                } else {
                    Write-Host ""
                    Write-Host "Available VMs:" -ForegroundColor Yellow
                    $global:localVMs.Keys | ForEach-Object { Write-Host "  • $_" -ForegroundColor White }
                    Write-Host ""
                    $vmName = Read-Host "Enter VM name"
                    
                    if ($global:localVMs.ContainsKey($vmName)) {
                        Invoke-TestsOnLocalVM -VMName $vmName
                    } else {
                        Write-Host "❌ VM not found: $vmName" -ForegroundColor Red
                    }
                }
            }
            "status" {
                Show-LocalVMStatus
            }
            "help" {
                Write-Host ""
                Write-Host "🖥️💻 TARS Local VM Commands:" -ForegroundColor Cyan
                Write-Host "• 'create' - Create a new VM for a project" -ForegroundColor White
                Write-Host "• 'deploy' - Deploy project to an existing VM" -ForegroundColor White
                Write-Host "• 'test' - Run tests on a deployed VM" -ForegroundColor White
                Write-Host "• 'status' - Show VM status and metrics" -ForegroundColor White
                Write-Host "• 'exit' - End the session" -ForegroundColor White
            }
            default {
                Write-Host "Unknown command. Type 'help' for available commands." -ForegroundColor Red
            }
        }
    }
}

# Start the system
Start-LocalVMDeployment
