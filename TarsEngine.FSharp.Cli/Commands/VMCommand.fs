namespace TarsEngine.FSharp.Cli.Commands

open System
open System.IO
open System.Diagnostics
open Spectre.Console
// Types are in the same namespace

/// Real VM deployment command using Docker
module VMCommand =
    
    type VMProvider = 
        | Docker
        | WSL
    
    type DeploymentResult = {
        Success: bool
        VMName: string option
        IPAddress: string option
        Port: int option
        ErrorMessage: string option
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
    
    /// Check if Docker is available
    let checkDocker () =
        let (code, output, _) = executeCommand "docker" "--version" "."
        if code = 0 then
            AnsiConsole.MarkupLine($"[green]✓ Docker: {output.Trim()}[/]")
            true
        else
            AnsiConsole.MarkupLine("[red]✗ Docker not found[/]")
            false
    
    /// Deploy using Docker
    let deployWithDocker (projectPath: string) (projectName: string) =
        AnsiConsole.MarkupLine("[cyan]Deploying with Docker...[/]")
        
        let dockerfilePath = Path.Combine(projectPath, "Dockerfile")
        
        if not (File.Exists(dockerfilePath)) then
            // Create a basic Dockerfile for F# projects
            let dockerfile = """FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src
COPY . .
RUN dotnet restore
RUN dotnet build -c Release

FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS runtime
WORKDIR /app
COPY --from=build /src/src/*/bin/Release/net8.0/ .
EXPOSE 5000
ENV ASPNETCORE_URLS=http://+:5000
ENTRYPOINT ["dotnet", "*.dll"]
"""
            File.WriteAllText(dockerfilePath, dockerfile)
            AnsiConsole.MarkupLine("[yellow]Created Dockerfile[/]")
        
        // Build Docker image
        let imageName = $"tars-{projectName.ToLowerInvariant()}"
        AnsiConsole.MarkupLine($"[cyan]Building image: {imageName}[/]")
        
        let (buildCode, buildOut, buildErr) = executeCommand "docker" $"build -t {imageName} ." projectPath
        
        if buildCode <> 0 then
            AnsiConsole.MarkupLine($"[red]Build failed:[/] {buildErr}")
            { Success = false; VMName = None; IPAddress = None; Port = None; ErrorMessage = Some buildErr }
        else
            AnsiConsole.MarkupLine("[green]Image built successfully[/]")
            
            // Run container
            let containerName = $"tars-{projectName.ToLowerInvariant()}-container"
            let port = 5000 + Random().Next(1000)
            
            // Stop existing container
            executeCommand "docker" $"stop {containerName}" "." |> ignore
            executeCommand "docker" $"rm {containerName}" "." |> ignore
            
            AnsiConsole.MarkupLine($"[cyan]Starting container on port {port}[/]")
            let (runCode, runOut, runErr) = executeCommand "docker" $"run -d --name {containerName} -p {port}:5000 {imageName}" "."
            
            if runCode <> 0 then
                AnsiConsole.MarkupLine($"[red]Failed to start container:[/] {runErr}")
                { Success = false; VMName = None; IPAddress = None; Port = None; ErrorMessage = Some runErr }
            else
                // Wait for container to start
                System.Threading.Thread.Sleep(3000)
                
                { 
                    Success = true
                    VMName = Some containerName
                    IPAddress = Some "localhost"
                    Port = Some port
                    ErrorMessage = None
                }
    
    /// List running deployments
    let listDeployments () =
        AnsiConsole.MarkupLine("[cyan]Active deployments:[/]")
        
        let (code, output, _) = executeCommand "docker" "ps --filter name=tars- --format \"table {{.Names}}\\t{{.Status}}\\t{{.Ports}}\"" "."
        
        if code = 0 && not (String.IsNullOrWhiteSpace(output)) then
            AnsiConsole.WriteLine(output)
        else
            AnsiConsole.MarkupLine("[yellow]No active deployments[/]")
    
    /// Stop deployment
    let stopDeployment (vmName: string) =
        AnsiConsole.MarkupLine($"[cyan]Stopping: {vmName}[/]")
        
        let (stopCode, _, stopErr) = executeCommand "docker" $"stop {vmName}" "."
        let (rmCode, _, _) = executeCommand "docker" $"rm {vmName}" "."
        
        if stopCode = 0 then
            AnsiConsole.MarkupLine("[green]Stopped successfully[/]")
            true
        else
            AnsiConsole.MarkupLine($"[red]Failed: {stopErr}[/]")
            false
    
    /// Test deployment connectivity
    let testDeployment (ip: string) (port: int) =
        AnsiConsole.MarkupLine("[cyan]Testing connectivity...[/]")
        try
            use client = new System.Net.Http.HttpClient()
            client.Timeout <- TimeSpan.FromSeconds(10.0)
            let response = client.GetAsync($"http://{ip}:{port}").Result
            AnsiConsole.MarkupLine($"[green]✓ Responding (HTTP {int response.StatusCode})[/]")
            true
        with
        | ex -> 
            AnsiConsole.MarkupLine($"[yellow]! Not responding: {ex.Message}[/]")
            false
    
    /// VM command implementation
    type VMCommand() =
        interface ICommand with
            member _.Name = "vm"
            member _.Description = "Deploy and manage projects in VMs/containers"
            member self.Usage = "vm <deploy|test|list|stop> [options]"
            member self.Examples = [
                "vm deploy output/projects/taskmanager"
                "vm test output/projects/taskmanager"
                "vm list"
                "vm stop tars-taskmanager-container"
            ]

            member self.ValidateOptions(options: CommandOptions) =
                options.Arguments.Length > 0

            member self.ExecuteAsync(options: CommandOptions) =
                task {
                    match options.Arguments with
                    | "deploy" :: projectPath :: _ ->
                        if not (Directory.Exists(projectPath)) then
                            AnsiConsole.MarkupLine($"[red]Project not found: {projectPath}[/]")
                            return CommandResult.failure "Project not found"
                        else
                            let projectName = Path.GetFileName(projectPath)
                            AnsiConsole.MarkupLine($"[green]Deploying: {projectName}[/]")

                            if checkDocker() then
                                let result = deployWithDocker projectPath projectName

                                if result.Success then
                                    AnsiConsole.MarkupLine("[green]✓ Deployment successful![/]")

                                    match result.IPAddress, result.Port with
                                    | Some ip, Some port ->
                                        AnsiConsole.MarkupLine($"[cyan]URL: http://{ip}:{port}[/]")
                                        testDeployment ip port |> ignore
                                    | _ -> ()

                                    return CommandResult.success "Deployment successful"
                                else
                                    AnsiConsole.MarkupLine($"[red]✗ Deployment failed[/]")
                                    return CommandResult.failure "Deployment failed"
                            else
                                AnsiConsole.MarkupLine("[red]Docker required for VM deployment[/]")
                                return CommandResult.failure "Docker required"
            
                    | "list" :: _ ->
                        listDeployments()
                        return CommandResult.success "Deployments listed"

                    | "stop" :: vmName :: _ ->
                        if stopDeployment vmName then
                            return CommandResult.success "Deployment stopped"
                        else
                            return CommandResult.failure "Failed to stop deployment"
            
                    | "test" :: projectPath :: _ ->
                        if not (Directory.Exists(projectPath)) then
                            AnsiConsole.MarkupLine($"[red]Project not found: {projectPath}[/]")
                            return CommandResult.failure "Project not found"
                        else
                            // Deploy and run tests
                            let projectName = Path.GetFileName(projectPath)
                            AnsiConsole.MarkupLine($"[green]Testing: {projectName}[/]")

                            if checkDocker() then
                                let result = deployWithDocker projectPath projectName

                                if result.Success then
                                    match result.IPAddress, result.Port with
                                    | Some ip, Some port ->
                                        if testDeployment ip port then
                                            AnsiConsole.MarkupLine("[green]✓ QA test passed![/]")
                                            return CommandResult.success "QA test passed"
                                        else
                                            AnsiConsole.MarkupLine("[red]✗ QA test failed![/]")
                                            return CommandResult.failure "QA test failed"
                                    | _ -> return CommandResult.failure "No IP/Port available"
                                else
                                    return CommandResult.failure "Deployment failed"
                            else
                                return CommandResult.failure "Docker required"

                    | _ ->
                        AnsiConsole.MarkupLine("[yellow]VM Commands:[/]")
                        AnsiConsole.MarkupLine("  vm deploy <project-path>  - Deploy project to container")
                        AnsiConsole.MarkupLine("  vm test <project-path>    - Deploy and test project")
                        AnsiConsole.MarkupLine("  vm list                   - List active deployments")
                        AnsiConsole.MarkupLine("  vm stop <container-name>  - Stop deployment")
                        AnsiConsole.MarkupLine("")
                        AnsiConsole.MarkupLine("[cyan]Examples:[/]")
                        AnsiConsole.MarkupLine("  tars vm deploy output/projects/taskmanager")
                        AnsiConsole.MarkupLine("  tars vm test output/projects/taskmanager")
                        AnsiConsole.MarkupLine("  tars vm list")
                        return CommandResult.success "Help displayed"
                }
