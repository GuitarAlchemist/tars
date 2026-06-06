module Tars.Interface.Cli.Commands.LlamaServer

open System
open System.Diagnostics
open System.Net.Http
open System.Threading.Tasks
open Serilog
open Spectre.Console

/// llama-server configuration
type LlamaServerConfig = {
    ExecutablePath: string
    ModelPath: string
    Port: int
    GpuLayers: int
    ParallelSlots: int
    ContextSize: int
    Host: string
}

/// Default configuration
let defaultConfig = {
    ExecutablePath = @"C:\Users\spare\AppData\Local\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-server.exe"
    ModelPath = @"C:\Users\spare\source\repos\tars\v2\models\magistral\Magistral-Small-2507-Q4_K_M.gguf"
    Port = 8080
    GpuLayers = -1  // All layers to GPU
    ParallelSlots = 4
    ContextSize = 32768
    Host = "0.0.0.0"
}

/// Check if llama-server is running on the specified port
let isServerRunning (port: int) : Task<bool> =
    task {
        try
            use client = new HttpClient(Timeout = TimeSpan.FromSeconds(2.0))
            let! response = client.GetAsync($"http://localhost:{port}/health")
            return response.IsSuccessStatusCode
        with
        | _ -> return false
    }

/// Start llama-server process
let startServer (config: LlamaServerConfig) (logger: ILogger) : Process option =
    try
        let startInfo = ProcessStartInfo()
        startInfo.FileName <- config.ExecutablePath
        startInfo.Arguments <- 
            $"-m \"{config.ModelPath}\" " +
            $"--port {config.Port} " +
            $"-ngl {config.GpuLayers} " +
            $"-np {config.ParallelSlots} " +
            $"-c {config.ContextSize} " +
            $"--host {config.Host} " +
            $"--n-predict -1 " +
            $"--verbose"
        
        startInfo.UseShellExecute <- false
        startInfo.RedirectStandardOutput <- true
        startInfo.RedirectStandardError <- true
        startInfo.CreateNoWindow <- true
        
        let proc = Process.Start(startInfo)
        
        // Log output asynchronously
        proc.OutputDataReceived.Add(fun args ->
            if not (String.IsNullOrWhiteSpace(args.Data)) then
                logger.Debug("llama-server: {Output}", args.Data))
        
        proc.ErrorDataReceived.Add(fun args ->
            if not (String.IsNullOrWhiteSpace(args.Data)) then
                logger.Warning("llama-server: {Error}", args.Data))
        
        proc.BeginOutputReadLine()
        proc.BeginErrorReadLine()
        
        Some proc
    with ex ->
        logger.Error(ex, "Failed to start llama-server")
        None

/// Display server status
let displayStatus (port: int) (logger: ILogger) : Task<int> =
    task {
        AnsiConsole.Write(new Rule("[bold blue]llama-server Status[/]"))
        
        let! isRunning = isServerRunning port
        
        if isRunning then
            AnsiConsole.MarkupLine($"[green]✓[/] Server is [green]RUNNING[/] on port {port}")
            
            try
                use client = new HttpClient(Timeout = TimeSpan.FromSeconds(5.0))
                let! response = client.GetAsync($"http://localhost:{port}/health")
                let! content = response.Content.ReadAsStringAsync()
                
                AnsiConsole.MarkupLine($"[grey]Health endpoint: {content}[/]")
            with ex ->
                AnsiConsole.MarkupLine($"[yellow]⚠[/] Server running but health check failed: {ex.Message}")
        else
            AnsiConsole.MarkupLine($"[red]✗[/] Server is [red]NOT RUNNING[/] on port {port}")
            AnsiConsole.MarkupLine("[grey]Run 'tars llama start' to start the server[/]")
        
        return if isRunning then 0 else 1
    }

/// Start server with status monitoring
let runStart (config: LlamaServerConfig) (logger: ILogger) : Task<int> =
    task {
        AnsiConsole.Write(new Rule("[bold green]Starting llama-server[/]"))
        
        // Check if already running
        let! isRunning = isServerRunning config.Port
        
        if isRunning then
            AnsiConsole.MarkupLine($"[yellow]⚠[/] Server is already running on port {config.Port}")
            return 0
        else
            AnsiConsole.MarkupLine($"[cyan]📍[/] Executable: [grey]{config.ExecutablePath}[/]")
            AnsiConsole.MarkupLine($"[cyan]📦[/] Model: [grey]{config.ModelPath}[/]")
            AnsiConsole.MarkupLine($"[cyan]🔌[/] Port: [grey]{config.Port}[/]")
            AnsiConsole.MarkupLine($"[cyan]🎮[/] GPU Layers: [grey]{config.GpuLayers} (all)[/]")
            AnsiConsole.MarkupLine($"[cyan]⚡[/] Parallel Slots: [grey]{config.ParallelSlots}[/]")
            AnsiConsole.MarkupLine($"[cyan]📏[/] Context Size: [grey]{config.ContextSize}[/]")
            AnsiConsole.WriteLine()
            
            match startServer config logger with
            | Some proc ->
                AnsiConsole.MarkupLine($"[green]✓[/] Server process started (PID: {proc.Id})")
                AnsiConsole.MarkupLine("[grey]Waiting for server to be ready...[/]")
                
                // Wait for server to be ready (with timeout)
                let mutable attempts = 0
                let mutable ready = false
                
                while not ready && attempts < 30 do
                    do! Task.Delay(1000)
                    let! isRunning = isServerRunning config.Port
                    ready <- isRunning
                    attempts <- attempts + 1
                    
                    if attempts % 5 = 0 then
                        AnsiConsole.MarkupLine($"[grey]  Still waiting... ({attempts}s)[/]")
                
                if ready then
                    AnsiConsole.MarkupLine("[green]✓[/] Server is ready!")
                    return 0
                else
                    AnsiConsole.MarkupLine("[red]✗[/] Server failed to start within timeout")
                    return 1
            | None ->
                AnsiConsole.MarkupLine("[red]✗[/] Failed to start server process")
                return 1
    }

/// Stop server
let runStop (port: int) (logger: ILogger) : Task<int> =
    task {
        AnsiConsole.Write(new Rule("[bold red]Stopping llama-server[/]"))
        
        let! isRunning = isServerRunning port
        
        if not isRunning then
            AnsiConsole.MarkupLine($"[grey]Server is not running on port {port}[/]")
            return 0
        else
            // Find and kill the process
            let procs = 
                Process.GetProcesses()
                |> Array.filter (fun p -> 
                    try
                        p.ProcessName.Contains("llama-server") ||
                        p.ProcessName.Contains("llama_server")
                    with _ -> false)
            
            if procs.Length > 0 then
                for proc in procs do
                    try
                        AnsiConsole.MarkupLine($"[yellow]Killing process {proc.Id}...[/]")
                        proc.Kill()
                        proc.WaitForExit(5000) |> ignore
                    with ex ->
                        logger.Warning(ex, "Failed to kill process {Pid}", proc.Id)
                
                AnsiConsole.MarkupLine("[green]✓[/] Server stopped")
                return 0
            else
                AnsiConsole.MarkupLine("[yellow]⚠[/] No llama-server process found")
                return 1
    }
