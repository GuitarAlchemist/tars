namespace TarsEngine.FSharp.LocalVM

open System
open System.IO
open System.Diagnostics
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// <summary>
/// Real Local VM Manager using VirtualBox and Vagrant
/// Provides actual VM deployment and testing capabilities
/// </summary>
module VagrantVMManager =
    
    /// VM Configuration for local deployment
    type LocalVMConfig = {
        VMName: string
        Memory: int // MB
        CPUs: int
        DiskSize: int // GB
        BaseBox: string // "ubuntu/jammy64", "generic/ubuntu2204", etc.
        NetworkType: string // "private_network", "public_network"
        IPAddress: string option
        ForwardedPorts: (int * int) list // (guest_port, host_port)
        ProvisionScript: string option
        SharedFolders: (string * string) list // (host_path, guest_path)
    }
    
    /// VM Instance information
    type VMInstance = {
        Name: string
        Status: string
        IPAddress: string
        SSHPort: int
        VagrantfilePath: string
        CreatedAt: DateTime
        LastAccessed: DateTime
    }
    
    /// Command execution result
    type CommandResult = {
        ExitCode: int
        StandardOutput: string
        StandardError: string
        Duration: TimeSpan
    }
    
    /// <summary>
    /// Vagrant VM Manager for real local virtualization
    /// </summary>
    type VagrantVMManager(logger: ILogger<VagrantVMManager>) =
        
        let mutable activeVMs = Map.empty<string, VMInstance>
        
        /// <summary>
        /// Check if required tools are installed
        /// </summary>
        member this.CheckPrerequisites() : Task<bool> =
            task {
                logger.LogInformation("Checking prerequisites for local VM deployment")
                
                try
                    // Check VirtualBox
                    let! vboxResult = this.ExecuteCommand("VBoxManage", "--version", ".")
                    if vboxResult.ExitCode <> 0 then
                        logger.LogError("VirtualBox not found. Please install VirtualBox from https://www.virtualbox.org/")
                        return false
                    
                    logger.LogInformation("VirtualBox found: {Version}", vboxResult.StandardOutput.Trim())
                    
                    // Check Vagrant
                    let! vagrantResult = this.ExecuteCommand("vagrant", "--version", ".")
                    if vagrantResult.ExitCode <> 0 then
                        logger.LogError("Vagrant not found. Please install Vagrant from https://www.vagrantup.com/")
                        return false
                    
                    logger.LogInformation("Vagrant found: {Version}", vagrantResult.StandardOutput.Trim())
                    
                    return true
                    
                with
                | ex ->
                    logger.LogError(ex, "Error checking prerequisites")
                    return false
            }
        
        /// <summary>
        /// Create and start a new VM
        /// </summary>
        member this.CreateVM(config: LocalVMConfig) : Task<VMInstance option> =
            task {
                logger.LogInformation("Creating VM: {VMName}", config.VMName)
                
                try
                    // Create VM directory
                    let vmDir = Path.Combine(".tars", "vms", config.VMName)
                    Directory.CreateDirectory(vmDir) |> ignore
                    
                    // Generate Vagrantfile
                    let vagrantfile = this.GenerateVagrantfile(config)
                    let vagrantfilePath = Path.Combine(vmDir, "Vagrantfile")
                    File.WriteAllText(vagrantfilePath, vagrantfile)
                    
                    logger.LogInformation("Generated Vagrantfile at: {Path}", vagrantfilePath)
                    
                    // Start VM
                    logger.LogInformation("Starting VM: {VMName}", config.VMName)
                    let! upResult = this.ExecuteCommand("vagrant", "up", vmDir)
                    
                    if upResult.ExitCode <> 0 then
                        logger.LogError("Failed to start VM: {Error}", upResult.StandardError)
                        return None
                    
                    // Get VM info
                    let! vmInfo = this.GetVMInfo(config.VMName, vmDir)
                    
                    match vmInfo with
                    | Some vm ->
                        activeVMs <- activeVMs.Add(config.VMName, vm)
                        logger.LogInformation("VM created successfully: {VMName} at {IP}", vm.Name, vm.IPAddress)
                        return Some vm
                    | None ->
                        logger.LogError("Failed to get VM information after creation")
                        return None
                        
                with
                | ex ->
                    logger.LogError(ex, "Error creating VM: {VMName}", config.VMName)
                    return None
            }
        
        /// <summary>
        /// Generate Vagrantfile content
        /// </summary>
        member private this.GenerateVagrantfile(config: LocalVMConfig) : string =
            let forwardedPortsConfig = 
                config.ForwardedPorts
                |> List.map (fun (guest, host) -> $"  config.vm.network \"forwarded_port\", guest: {guest}, host: {host}")
                |> String.concat "\n"
            
            let sharedFoldersConfig =
                config.SharedFolders
                |> List.map (fun (host, guest) -> $"  config.vm.synced_folder \"{host}\", \"{guest}\"")
                |> String.concat "\n"
            
            let networkConfig = 
                match config.IPAddress with
                | Some ip -> $"  config.vm.network \"private_network\", ip: \"{ip}\""
                | None -> "  config.vm.network \"private_network\", type: \"dhcp\""
            
            let provisionConfig =
                match config.ProvisionScript with
                | Some script -> $"  config.vm.provision \"shell\", inline: <<-SHELL\n{script}\n  SHELL"
                | None -> ""
            
            $"""# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure("2") do |config|
  config.vm.box = "{config.BaseBox}"
  config.vm.hostname = "{config.VMName}"
  
  # Network configuration
{networkConfig}
{forwardedPortsConfig}
  
  # Shared folders
{sharedFoldersConfig}
  
  # Provider configuration
  config.vm.provider "virtualbox" do |vb|
    vb.name = "{config.VMName}"
    vb.memory = {config.Memory}
    vb.cpus = {config.CPUs}
    vb.customize ["modifyvm", :id, "--natdnshostresolver1", "on"]
    vb.customize ["modifyvm", :id, "--natdnsproxy1", "on"]
  end
  
  # Provisioning
{provisionConfig}
end
"""
        
        /// <summary>
        /// Get VM information
        /// </summary>
        member private this.GetVMInfo(vmName: string, vmDir: string) : Task<VMInstance option> =
            task {
                try
                    // Get VM status
                    let! statusResult = this.ExecuteCommand("vagrant", "status", vmDir)
                    
                    if statusResult.ExitCode <> 0 then
                        return None
                    
                    // Get SSH config
                    let! sshConfigResult = this.ExecuteCommand("vagrant", "ssh-config", vmDir)
                    
                    if sshConfigResult.ExitCode <> 0 then
                        return None
                    
                    // Parse SSH config to get IP and port
                    let sshConfig = sshConfigResult.StandardOutput
                    let ipAddress = this.ExtractFromSSHConfig(sshConfig, "HostName")
                    let sshPort = this.ExtractFromSSHConfig(sshConfig, "Port") |> int
                    
                    let vmInstance = {
                        Name = vmName
                        Status = "running"
                        IPAddress = ipAddress
                        SSHPort = sshPort
                        VagrantfilePath = Path.Combine(vmDir, "Vagrantfile")
                        CreatedAt = DateTime.UtcNow
                        LastAccessed = DateTime.UtcNow
                    }
                    
                    return Some vmInstance
                    
                with
                | ex ->
                    logger.LogError(ex, "Error getting VM info for: {VMName}", vmName)
                    return None
            }
        
        /// <summary>
        /// Execute command on VM via SSH
        /// </summary>
        member this.ExecuteOnVM(vmName: string, command: string) : Task<CommandResult> =
            task {
                match activeVMs.TryFind(vmName) with
                | Some vm ->
                    let vmDir = Path.GetDirectoryName(vm.VagrantfilePath)
                    let vagrantCommand = $"ssh -c \"{command}\""
                    return! this.ExecuteCommand("vagrant", vagrantCommand, vmDir)
                | None ->
                    logger.LogError("VM not found: {VMName}", vmName)
                    return {
                        ExitCode = 1
                        StandardOutput = ""
                        StandardError = $"VM not found: {vmName}"
                        Duration = TimeSpan.Zero
                    }
            }
        
        /// <summary>
        /// Copy files to VM
        /// </summary>
        member this.CopyToVM(vmName: string, localPath: string, remotePath: string) : Task<bool> =
            task {
                try
                    match activeVMs.TryFind(vmName) with
                    | Some vm ->
                        // Use SCP to copy files
                        let scpCommand = $"scp -P {vm.SSHPort} -o StrictHostKeyChecking=no -r \"{localPath}\" vagrant@{vm.IPAddress}:\"{remotePath}\""
                        let! result = this.ExecuteCommand("cmd", $"/c {scpCommand}", ".")
                        
                        if result.ExitCode = 0 then
                            logger.LogInformation("Successfully copied {LocalPath} to VM {VMName}:{RemotePath}", localPath, vmName, remotePath)
                            return true
                        else
                            logger.LogError("Failed to copy files to VM: {Error}", result.StandardError)
                            return false
                    | None ->
                        logger.LogError("VM not found: {VMName}", vmName)
                        return false
                        
                with
                | ex ->
                    logger.LogError(ex, "Error copying files to VM: {VMName}", vmName)
                    return false
            }
        
        /// <summary>
        /// Deploy project to VM
        /// </summary>
        member this.DeployProject(vmName: string, projectPath: string) : Task<bool> =
            task {
                logger.LogInformation("Deploying project to VM: {VMName}", vmName)
                
                try
                    // Copy project files
                    let! copyResult = this.CopyToVM(vmName, projectPath, "/home/vagrant/project")
                    
                    if not copyResult then
                        return false
                    
                    // Install dependencies and build
                    let setupCommands = [
                        "sudo apt-get update -y"
                        "sudo apt-get install -y docker.io docker-compose"
                        "sudo systemctl start docker"
                        "sudo systemctl enable docker"
                        "sudo usermod -aG docker vagrant"
                        
                        // Install .NET 8
                        "wget https://packages.microsoft.com/config/ubuntu/22.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb"
                        "sudo dpkg -i packages-microsoft-prod.deb"
                        "sudo apt-get update -y"
                        "sudo apt-get install -y dotnet-sdk-8.0"
                        
                        // Build and run project
                        "cd /home/vagrant/project"
                        "dotnet restore"
                        "dotnet build --configuration Release"
                        
                        // Start with Docker if Dockerfile exists
                        "if [ -f Dockerfile ]; then docker build -t project .; docker run -d -p 5000:5000 project; fi"
                        
                        // Or start directly
                        "if [ ! -f Dockerfile ]; then nohup dotnet run --project src/*/*.fsproj --urls http://0.0.0.0:5000 > app.log 2>&1 & fi"
                    ]
                    
                    for command in setupCommands do
                        logger.LogDebug("Executing: {Command}", command)
                        let! result = this.ExecuteOnVM(vmName, command)
                        
                        if result.ExitCode <> 0 then
                            logger.LogWarning("Command failed: {Command}, Error: {Error}", command, result.StandardError)
                            // Continue with other commands
                    
                    logger.LogInformation("Project deployment completed for VM: {VMName}", vmName)
                    return true
                    
                with
                | ex ->
                    logger.LogError(ex, "Error deploying project to VM: {VMName}", vmName)
                    return false
            }
        
        /// <summary>
        /// Run tests on VM
        /// </summary>
        member this.RunTests(vmName: string) : Task<CommandResult> =
            task {
                logger.LogInformation("Running tests on VM: {VMName}", vmName)
                
                let testCommand = "cd /home/vagrant/project && dotnet test --logger trx --results-directory TestResults"
                return! this.ExecuteOnVM(vmName, testCommand)
            }
        
        /// <summary>
        /// Stop and destroy VM
        /// </summary>
        member this.DestroyVM(vmName: string) : Task<bool> =
            task {
                logger.LogInformation("Destroying VM: {VMName}", vmName)
                
                try
                    match activeVMs.TryFind(vmName) with
                    | Some vm ->
                        let vmDir = Path.GetDirectoryName(vm.VagrantfilePath)
                        
                        // Stop and destroy VM
                        let! destroyResult = this.ExecuteCommand("vagrant", "destroy -f", vmDir)
                        
                        if destroyResult.ExitCode = 0 then
                            activeVMs <- activeVMs.Remove(vmName)
                            logger.LogInformation("VM destroyed successfully: {VMName}", vmName)
                            return true
                        else
                            logger.LogError("Failed to destroy VM: {Error}", destroyResult.StandardError)
                            return false
                    | None ->
                        logger.LogWarning("VM not found for destruction: {VMName}", vmName)
                        return true
                        
                with
                | ex ->
                    logger.LogError(ex, "Error destroying VM: {VMName}", vmName)
                    return false
            }
        
        /// <summary>
        /// List all active VMs
        /// </summary>
        member this.ListVMs() : VMInstance list =
            activeVMs |> Map.toList |> List.map snd
        
        /// <summary>
        /// Execute system command
        /// </summary>
        member private this.ExecuteCommand(command: string, arguments: string, workingDirectory: string) : Task<CommandResult> =
            task {
                let startTime = DateTime.UtcNow
                
                let processInfo = ProcessStartInfo(
                    FileName = command,
                    Arguments = arguments,
                    WorkingDirectory = workingDirectory,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                )
                
                use proc = new Process(StartInfo = processInfo)
                proc.Start() |> ignore
                
                let! output = proc.StandardOutput.ReadToEndAsync()
                let! error = proc.StandardError.ReadToEndAsync()
                
                proc.WaitForExit()
                
                let endTime = DateTime.UtcNow
                let duration = endTime - startTime
                
                return {
                    ExitCode = proc.ExitCode
                    StandardOutput = output
                    StandardError = error
                    Duration = duration
                }
            }
        
        /// <summary>
        /// Extract value from SSH config
        /// </summary>
        member private this.ExtractFromSSHConfig(sshConfig: string, key: string) : string =
            sshConfig.Split('\n')
            |> Array.tryFind (fun line -> line.Trim().StartsWith(key))
            |> Option.map (fun line -> line.Split(' ') |> Array.last |> fun s -> s.Trim())
            |> Option.defaultValue "127.0.0.1"
