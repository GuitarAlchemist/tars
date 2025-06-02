namespace TarsEngine.FSharp.Cli.Commands

open System
open System.IO
open System.Diagnostics
open System.Threading.Tasks
open Spectre.Console

/// Real VM deployment command using actual tools
module VMDeploymentCommand =
    
    type VMProvider = 
        | VirtualBox
        | Docker
        | WSL
    
    type DeploymentResult = {
        Success: bool
        VMName: string option
        IPAddress: string option
        Port: int option
        ErrorMessage: string option
        LogPath: string option
    }
    
    /// Execute shell command and return result
    let executeCommand (command: string) (args: string) (workingDir: string) =
        try
            let processInfo = ProcessStartInfo(
                FileName = command,
                Arguments = args,
                WorkingDirectory = workingDir,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            )
            
            use proc = new Process(StartInfo = processInfo)
            proc.Start() |> ignore
            
            let output = proc.StandardOutput.ReadToEnd()
            let error = proc.StandardError.ReadToEnd()
            proc.WaitForExit()
            
            (proc.ExitCode, output, error)
        with
        | ex -> (-1, "", ex.Message)
    
    /// Check if required tools are installed
    let checkPrerequisites () =
        AnsiConsole.MarkupLine("[cyan]Checking prerequisites...[/]")
        
        let mutable allGood = true
        
        // Check Docker
        let (dockerCode, dockerOut, _) = executeCommand "docker" "--version" "."
        if dockerCode = 0 then
            AnsiConsole.MarkupLine($"[green]✓[/] Docker: {dockerOut.Trim()}")
        else
            AnsiConsole.MarkupLine("[red]✗[/] Docker not found")
            allGood <- false
        
        // Check VirtualBox (optional)
        let (vboxCode, vboxOut, _) = executeCommand "VBoxManage" "--version" "."
        if vboxCode = 0 then
            AnsiConsole.MarkupLine($"[green]✓[/] VirtualBox: {vboxOut.Trim()}")
        else
            AnsiConsole.MarkupLine("[yellow]![/] VirtualBox not found (optional)")
        
        // Check WSL (Windows only)
        if Environment.OSVersion.Platform = PlatformID.Win32NT then
            let (wslCode, wslOut, _) = executeCommand "wsl" "--version" "."
            if wslCode = 0 then
                AnsiConsole.MarkupLine("[green]✓[/] WSL available")
            else
                AnsiConsole.MarkupLine("[yellow]![/] WSL not available")
        
        allGood
    
    /// Deploy using Docker (most reliable)
    let deployWithDocker (projectPath: string) (projectName: string) =
        AnsiConsole.MarkupLine("[cyan]Deploying with Docker...[/]")
        
        let dockerfilePath = Path.Combine(projectPath, "Dockerfile")
        
        if not (File.Exists(dockerfilePath)) then
            // Create a basic Dockerfile
            let dockerfile = """FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src
COPY . .
RUN dotnet restore
RUN dotnet build -c Release

FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS runtime
WORKDIR /app
COPY --from=build /src/src/*/bin/Release/net8.0/ .
EXPOSE 5000
ENTRYPOINT ["dotnet", "*.dll"]
"""
            File.WriteAllText(dockerfilePath, dockerfile)
            AnsiConsole.MarkupLine("[yellow]Created basic Dockerfile[/]")
        
        // Build Docker image
        let imageName = $"tars-{projectName.ToLowerInvariant()}"
        let (buildCode, buildOut, buildErr) = executeCommand "docker" $"build -t {imageName} ." projectPath
        
        if buildCode <> 0 then
            AnsiConsole.MarkupLine($"[red]Docker build failed:[/] {buildErr}")
            { Success = false; VMName = None; IPAddress = None; Port = None; ErrorMessage = Some buildErr; LogPath = None }
        else
            AnsiConsole.MarkupLine("[green]Docker image built successfully[/]")
            
            // Run container
            let containerName = $"tars-{projectName.ToLowerInvariant()}-container"
            let port = 5000 + Random().Next(1000) // Random port to avoid conflicts
            
            // Stop existing container if running
            executeCommand "docker" $"stop {containerName}" "." |> ignore
            executeCommand "docker" $"rm {containerName}" "." |> ignore
            
            let (runCode, runOut, runErr) = executeCommand "docker" $"run -d --name {containerName} -p {port}:5000 {imageName}" "."
            
            if runCode <> 0 then
                AnsiConsole.MarkupLine($"[red]Docker run failed:[/] {runErr}")
                { Success = false; VMName = None; IPAddress = None; Port = None; ErrorMessage = Some runErr; LogPath = None }
            else
                AnsiConsole.MarkupLine($"[green]Container started on port {port}[/]")
                
                // Wait a moment for container to start
                System.Threading.Thread.Sleep(2000)
                
                // Test if container is running
                let (psCode, psOut, _) = executeCommand "docker" $"ps --filter name={containerName} --format \"table {{{{.Names}}}}\\t{{{{.Status}}}}\"" "."
                
                { 
                    Success = true
                    VMName = Some containerName
                    IPAddress = Some "localhost"
                    Port = Some port
                    ErrorMessage = None
                    LogPath = None
                }
    
    /// Deploy using WSL (Windows Subsystem for Linux)
    let deployWithWSL (projectPath: string) (projectName: string) =
        AnsiConsole.MarkupLine("[cyan]Deploying with WSL...[/]")
        
        if Environment.OSVersion.Platform <> PlatformID.Win32NT then
            { Success = false; VMName = None; IPAddress = None; Port = None; ErrorMessage = Some "WSL only available on Windows"; LogPath = None }
        else
            // Copy project to WSL
            let wslProjectPath = $"/tmp/tars-{projectName}"
            let (copyCode, _, copyErr) = executeCommand "wsl" $"cp -r \"{projectPath}\" {wslProjectPath}" "."
            
            if copyCode <> 0 then
                { Success = false; VMName = None; IPAddress = None; Port = None; ErrorMessage = Some copyErr; LogPath = None }
            else
                // Install .NET in WSL if needed
                let installCommands = [
                    "sudo apt-get update -y"
                    "sudo apt-get install -y wget"
                    "wget https://packages.microsoft.com/config/ubuntu/22.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb"
                    "sudo dpkg -i packages-microsoft-prod.deb"
                    "sudo apt-get update -y"
                    "sudo apt-get install -y dotnet-sdk-8.0"
                ]
                
                for cmd in installCommands do
                    let (code, _, _) = executeCommand "wsl" cmd "."
                    if code <> 0 then
                        AnsiConsole.MarkupLine($"[yellow]Warning: Command failed: {cmd}[/]")
                
                // Build and run project
                let port = 5000 + Random().Next(1000)
                let runCommand = $"cd {wslProjectPath} && dotnet restore && dotnet build && nohup dotnet run --urls http://0.0.0.0:{port} > app.log 2>&1 &"
                let (runCode, runOut, runErr) = executeCommand "wsl" runCommand "."
                
                if runCode <> 0 then
                    { Success = false; VMName = None; IPAddress = None; Port = None; ErrorMessage = Some runErr; LogPath = None }
                else
                    { 
                        Success = true
                        VMName = Some "WSL"
                        IPAddress = Some "localhost"
                        Port = Some port
                        ErrorMessage = None
                        LogPath = Some $"{wslProjectPath}/app.log"
                    }
    
    /// Main deployment function
    let deployProject (projectPath: string) =
        if not (Directory.Exists(projectPath)) then
            AnsiConsole.MarkupLine($"[red]Project path not found: {projectPath}[/]")
            { Success = false; VMName = None; IPAddress = None; Port = None; ErrorMessage = Some "Project not found"; LogPath = None }
        else
            let projectName = Path.GetFileName(projectPath)
            
            AnsiConsole.MarkupLine($"[green]Deploying project: {projectName}[/]")
            AnsiConsole.MarkupLine($"[gray]Path: {projectPath}[/]")
            
            if not (checkPrerequisites()) then
                { Success = false; VMName = None; IPAddress = None; Port = None; ErrorMessage = Some "Prerequisites not met"; LogPath = None }
            else
                // Try Docker first (most reliable)
                let dockerResult = deployWithDocker projectPath projectName
                
                if dockerResult.Success then
                    dockerResult
                else
                    // Fallback to WSL on Windows
                    if Environment.OSVersion.Platform = PlatformID.Win32NT then
                        deployWithWSL projectPath projectName
                    else
                        dockerResult
    
    /// List running deployments
    let listDeployments () =
        AnsiConsole.MarkupLine("[cyan]Checking running deployments...[/]")
        
        // Check Docker containers
        let (dockerCode, dockerOut, _) = executeCommand "docker" "ps --filter name=tars- --format \"table {{.Names}}\\t{{.Status}}\\t{{.Ports}}\"" "."
        
        if dockerCode = 0 && not (String.IsNullOrWhiteSpace(dockerOut)) then
            AnsiConsole.MarkupLine("[green]Docker containers:[/]")
            AnsiConsole.WriteLine(dockerOut)
        else
            AnsiConsole.MarkupLine("[yellow]No Docker containers running[/]")
        
        // Check WSL processes (if on Windows)
        if Environment.OSVersion.Platform = PlatformID.Win32NT then
            let (wslCode, wslOut, _) = executeCommand "wsl" "ps aux | grep dotnet | grep -v grep" "."
            if wslCode = 0 && not (String.IsNullOrWhiteSpace(wslOut)) then
                AnsiConsole.MarkupLine("[green]WSL processes:[/]")
                AnsiConsole.WriteLine(wslOut)
    
    /// Stop deployment
    let stopDeployment (vmName: string) =
        AnsiConsole.MarkupLine($"[cyan]Stopping deployment: {vmName}[/]")
        
        if vmName.StartsWith("tars-") && vmName.EndsWith("-container") then
            // Docker container
            let (stopCode, _, stopErr) = executeCommand "docker" $"stop {vmName}" "."
            let (rmCode, _, rmErr) = executeCommand "docker" $"rm {vmName}" "."
            
            if stopCode = 0 then
                AnsiConsole.MarkupLine("[green]Container stopped successfully[/]")
                true
            else
                AnsiConsole.MarkupLine($"[red]Failed to stop container: {stopErr}[/]")
                false
        else
            AnsiConsole.MarkupLine("[yellow]Manual cleanup required for WSL deployments[/]")
            false
    
    /// Run VM deployment command
    let runVMDeploymentCommand (args: string[]) =
        match args with
        | [| "deploy"; projectPath |] ->
            let result = deployProject projectPath
            
            if result.Success then
                AnsiConsole.MarkupLine("[green]✓ Deployment successful![/]")
                
                match result.VMName, result.IPAddress, result.Port with
                | Some vm, Some ip, Some port ->
                    AnsiConsole.MarkupLine($"[cyan]VM/Container:[/] {vm}")
                    AnsiConsole.MarkupLine($"[cyan]Access URL:[/] http://{ip}:{port}")
                    
                    // Test connectivity
                    AnsiConsole.MarkupLine("[cyan]Testing connectivity...[/]")
                    try
                        use client = new System.Net.Http.HttpClient()
                        client.Timeout <- TimeSpan.FromSeconds(5.0)
                        let response = client.GetAsync($"http://{ip}:{port}").Result
                        AnsiConsole.MarkupLine($"[green]✓ Application responding (HTTP {int response.StatusCode})[/]")
                    with
                    | ex -> AnsiConsole.MarkupLine($"[yellow]! Application may still be starting: {ex.Message}[/]")
                | _ -> ()
            else
                AnsiConsole.MarkupLine($"[red]✗ Deployment failed: {result.ErrorMessage |> Option.defaultValue "Unknown error"}[/]")
        
        | [| "list" |] ->
            listDeployments()
        
        | [| "stop"; vmName |] ->
            if stopDeployment vmName then
                AnsiConsole.MarkupLine("[green]✓ Deployment stopped[/]")
            else
                AnsiConsole.MarkupLine("[red]✗ Failed to stop deployment[/]")
        
        | _ ->
            AnsiConsole.MarkupLine("[yellow]Usage:[/]")
            AnsiConsole.MarkupLine("  tars vm deploy <project-path>")
            AnsiConsole.MarkupLine("  tars vm list")
            AnsiConsole.MarkupLine("  tars vm stop <vm-name>")
            AnsiConsole.MarkupLine("")
            AnsiConsole.MarkupLine("[cyan]Examples:[/]")
            AnsiConsole.MarkupLine("  tars vm deploy output/projects/taskmanager")
            AnsiConsole.MarkupLine("  tars vm list")
            AnsiConsole.MarkupLine("  tars vm stop tars-taskmanager-container")
