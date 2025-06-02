namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Text.Json
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open Spectre.Console
open TarsEngine.FSharp.Cli.WebSocket

/// <summary>
/// TARS CLI WebSocket Service Command
/// Provides real-time full-duplex communication with Windows service
/// </summary>
type WebSocketServiceCommand(logger: ILogger<WebSocketServiceCommand>) =
    
    let mutable webSocketClient: TarsWebSocketClient option = None
    let mutable isMonitoring = false
    
    /// Initialize WebSocket client
    member private this.InitializeWebSocket(serviceUrl: string) =
        match webSocketClient with
        | Some client -> client
        | None ->
            let client = new TarsWebSocketClient(logger)
            
            // Set up event handlers
            client.OnConnected(fun connectionId ->
                AnsiConsole.MarkupLine($"[green]✅ Connected to TARS service (ID: {connectionId})[/]")
            )
            
            client.OnResponse(fun command data ->
                match command with
                | "documentation.start" | "documentation.pause" | "documentation.resume" | "documentation.stop" ->
                    try
                        let success = data.GetProperty("success").GetBoolean()
                        let message = data.GetProperty("message").GetString()
                        let color = if success then "green" else "red"
                        AnsiConsole.MarkupLine($"[{color}]{if success then "✅" else "❌"} {message}[/]")
                    with _ -> ()
                
                | "documentation.status" ->
                    this.DisplayDocumentationStatus(data)
                
                | "service.status" ->
                    this.DisplayServiceStatus(data)
                
                | "pong" ->
                    try
                        let timestamp = data.GetProperty("timestamp").GetDateTime()
                        let latency = DateTime.UtcNow - timestamp
                        AnsiConsole.MarkupLine($"[green]🏓 Pong! Latency: {latency.TotalMilliseconds:F1}ms[/]")
                    with _ -> ()
                
                | _ -> ()
            )
            
            client.OnProgressUpdate(fun data ->
                if isMonitoring then
                    this.DisplayProgressUpdate(data)
            )
            
            client.OnError(fun error ->
                AnsiConsole.MarkupLine($"[red]❌ Error: {error}[/]")
            )
            
            webSocketClient <- Some client
            client
    
    /// Display documentation status
    member private this.DisplayDocumentationStatus(data: JsonElement) =
        try
            let status = data.GetProperty("status")
            let progress = status.GetProperty("Progress")
            let state = status.GetProperty("State").GetString()
            let completedTasks = progress.GetProperty("CompletedTasks").GetInt32()
            let totalTasks = progress.GetProperty("TotalTasks").GetInt32()
            let currentTask = progress.GetProperty("CurrentTask").GetString()
            let percentage = (float completedTasks / float totalTasks) * 100.0
            
            let stateColor = match state with
                            | "Running" -> "green"
                            | "Paused" -> "yellow"
                            | "Completed" -> "blue"
                            | _ -> "gray"
            
            let table = Table()
            table.AddColumn("Property") |> ignore
            table.AddColumn("Value") |> ignore
            table.Border <- TableBorder.Rounded
            table.Title <- TableTitle("📊 Documentation Task Status")
            
            table.AddRow("State", $"[{stateColor}]{state}[/]") |> ignore
            table.AddRow("Progress", $"[white]{completedTasks}/{totalTasks} ({percentage:F1}%)[/]") |> ignore
            table.AddRow("Current Task", $"[gray]{currentTask}[/]") |> ignore
            
            AnsiConsole.Write(table)
            
            // Display departments if available
            if progress.TryGetProperty("Departments", &_) then
                let departments = progress.GetProperty("Departments")
                
                let deptTable = Table()
                deptTable.AddColumn("Department") |> ignore
                deptTable.AddColumn("Progress") |> ignore
                deptTable.Border <- TableBorder.Simple
                deptTable.Title <- TableTitle("🏛️ University Departments")
                
                for dept in departments.EnumerateObject() do
                    let deptProgress = dept.Value.GetInt32()
                    let color = if deptProgress = 100 then "green" elif deptProgress > 0 then "yellow" else "gray"
                    deptTable.AddRow(dept.Name, $"[{color}]{deptProgress}%[/]") |> ignore
                
                AnsiConsole.Write(deptTable)
        
        with ex ->
            logger.LogError(ex, "Failed to display documentation status")
            AnsiConsole.MarkupLine("[red]❌ Failed to parse status data[/]")
    
    /// Display service status
    member private this.DisplayServiceStatus(data: JsonElement) =
        try
            let service = data.GetProperty("service").GetString()
            let status = data.GetProperty("status").GetString()
            let connections = data.GetProperty("connections").GetInt32()
            let version = data.GetProperty("version").GetString()
            
            let table = Table()
            table.AddColumn("Property") |> ignore
            table.AddColumn("Value") |> ignore
            table.Border <- TableBorder.Rounded
            table.Title <- TableTitle("🚀 TARS Service Status")
            
            table.AddRow("Service", $"[white]{service}[/]") |> ignore
            table.AddRow("Status", $"[green]{status}[/]") |> ignore
            table.AddRow("Version", $"[gray]{version}[/]") |> ignore
            table.AddRow("Connections", $"[blue]{connections}[/]") |> ignore
            
            AnsiConsole.Write(table)
        
        with ex ->
            logger.LogError(ex, "Failed to display service status")
            AnsiConsole.MarkupLine("[red]❌ Failed to parse service data[/]")
    
    /// Display progress update
    member private this.DisplayProgressUpdate(data: JsonElement) =
        try
            let status = data.GetProperty("status")
            let progress = status.GetProperty("Progress")
            let completedTasks = progress.GetProperty("CompletedTasks").GetInt32()
            let totalTasks = progress.GetProperty("TotalTasks").GetInt32()
            let currentTask = progress.GetProperty("CurrentTask").GetString()
            let percentage = (float completedTasks / float totalTasks) * 100.0
            
            AnsiConsole.Clear()
            
            // Create live display
            let rule = Rule("[cyan]📊 TARS Documentation Generation - Live Monitor[/]")
            rule.Style <- Style.Parse("cyan")
            AnsiConsole.Write(rule)
            
            // Progress bar
            let progressBar = ProgressBar()
            progressBar.Value <- percentage
            progressBar.Width <- 60
            
            AnsiConsole.MarkupLine($"[white]Progress: {completedTasks}/{totalTasks} ({percentage:F1}%)[/]")
            AnsiConsole.Write(progressBar)
            AnsiConsole.MarkupLine("")
            AnsiConsole.MarkupLine($"[yellow]Current Task: {currentTask}[/]")
            
            // Department progress if available
            if progress.TryGetProperty("Departments", &_) then
                let departments = progress.GetProperty("Departments")
                AnsiConsole.MarkupLine("")
                AnsiConsole.MarkupLine("[cyan]🏛️ Department Progress:[/]")
                
                for dept in departments.EnumerateObject() do
                    let deptProgress = dept.Value.GetInt32()
                    let color = if deptProgress = 100 then "green" elif deptProgress > 0 then "yellow" else "gray"
                    let bar = "█" |> String.replicate (deptProgress / 5) // Simple progress bar
                    let empty = "░" |> String.replicate (20 - (deptProgress / 5))
                    AnsiConsole.MarkupLine($"  [{color}]{dept.Name,-20} {bar}{empty} {deptProgress}%[/]")
            
            AnsiConsole.MarkupLine("")
            AnsiConsole.MarkupLine("[gray]Press Ctrl+C to stop monitoring...[/]")
        
        with ex ->
            logger.LogError(ex, "Failed to display progress update")
    
    /// Connect to service
    member this.ConnectAsync(serviceUrl: string) = task {
        let client = this.InitializeWebSocket(serviceUrl)
        
        AnsiConsole.Status()
            .Start("🔌 Connecting to TARS service...", fun ctx ->
                ctx.Spinner <- Spinner.Known.Star
                ctx.SpinnerStyle <- Style.Parse("green")
                
                task {
                    let! connected = client.ConnectAsync(serviceUrl)
                    
                    if not connected then
                        ctx.Status <- "[red]❌ Failed to connect[/]"
                        return false
                    else
                        ctx.Status <- "[green]✅ Connected successfully[/]"
                        return true
                }
            )
    }
    
    /// Disconnect from service
    member this.DisconnectAsync() = task {
        match webSocketClient with
        | Some client ->
            do! client.DisconnectAsync()
            AnsiConsole.MarkupLine("[yellow]🔌 Disconnected from TARS service[/]")
        | None -> ()
    }
    
    /// Execute interactive session
    member this.ExecuteInteractiveAsync(serviceUrl: string) = task {
        let! connected = this.ConnectAsync(serviceUrl)
        if not connected then 
            AnsiConsole.MarkupLine("[red]❌ Failed to connect to TARS service[/]")
            return 1
        
        let client = webSocketClient.Value
        
        AnsiConsole.Clear()
        let rule = Rule("[cyan]🤖 TARS Interactive WebSocket Session[/]")
        rule.Style <- Style.Parse("cyan")
        AnsiConsole.Write(rule)
        
        AnsiConsole.MarkupLine("[green]✅ Connected to TARS Windows Service[/]")
        AnsiConsole.MarkupLine("[blue]💡 Type 'help' for available commands, 'exit' to quit[/]")
        AnsiConsole.MarkupLine("")
        
        let mutable running = true
        
        while running do
            let input = AnsiConsole.Ask<string>("[cyan]TARS>[/] ")
            
            match input.ToLower().Trim() with
            | "exit" | "quit" | "q" ->
                running <- false
            
            | "help" | "h" ->
                let helpTable = Table()
                helpTable.AddColumn("Command") |> ignore
                helpTable.AddColumn("Description") |> ignore
                helpTable.Border <- TableBorder.Simple
                helpTable.Title <- TableTitle("📋 Available Commands")
                
                helpTable.AddRow("status", "Get service status") |> ignore
                helpTable.AddRow("doc-status", "Get documentation task status") |> ignore
                helpTable.AddRow("doc-start", "Start documentation generation") |> ignore
                helpTable.AddRow("doc-pause", "Pause documentation generation") |> ignore
                helpTable.AddRow("doc-resume", "Resume documentation generation") |> ignore
                helpTable.AddRow("doc-stop", "Stop documentation generation") |> ignore
                helpTable.AddRow("monitor", "Start live progress monitoring") |> ignore
                helpTable.AddRow("ping", "Ping the service") |> ignore
                helpTable.AddRow("clear", "Clear the screen") |> ignore
                helpTable.AddRow("exit", "Exit interactive session") |> ignore
                
                AnsiConsole.Write(helpTable)
            
            | "clear" | "cls" ->
                AnsiConsole.Clear()
            
            | "status" ->
                do! client.GetServiceStatusAsync()
                do! Task.Delay(500)
            
            | "doc-status" ->
                do! client.GetDocumentationStatusAsync()
                do! Task.Delay(500)
            
            | "doc-start" ->
                do! client.StartDocumentationAsync()
                do! Task.Delay(500)
            
            | "doc-pause" ->
                do! client.PauseDocumentationAsync()
                do! Task.Delay(500)
            
            | "doc-resume" ->
                do! client.ResumeDocumentationAsync()
                do! Task.Delay(500)
            
            | "doc-stop" ->
                do! client.StopDocumentationAsync()
                do! Task.Delay(500)
            
            | "monitor" ->
                AnsiConsole.MarkupLine("[blue]📡 Starting live monitoring (Press any key to stop)...[/]")
                isMonitoring <- true
                do! client.GetDocumentationStatusAsync()
                
                // Monitor until key press
                let monitorTask = task {
                    while isMonitoring && client.IsConnected do
                        do! Task.Delay(3000)
                }
                
                let keyTask = task {
                    Console.ReadKey(true) |> ignore
                    isMonitoring <- false
                }
                
                let! _ = Task.WhenAny([| monitorTask; keyTask |])
                isMonitoring <- false
                
                AnsiConsole.Clear()
                AnsiConsole.MarkupLine("[yellow]📡 Monitoring stopped[/]")
            
            | "ping" ->
                do! client.PingAsync()
                do! Task.Delay(500)
            
            | "" -> () // Empty input, do nothing
            
            | _ ->
                AnsiConsole.MarkupLine($"[red]❌ Unknown command: {input}[/]")
                AnsiConsole.MarkupLine("[yellow]💡 Type 'help' for available commands[/]")
        
        do! this.DisconnectAsync()
        return 0
    }
    
    /// Execute single command
    member this.ExecuteCommandAsync(action: string, serviceUrl: string) = task {
        let! connected = this.ConnectAsync(serviceUrl)
        if not connected then return 1
        
        let client = webSocketClient.Value
        
        match action.ToLower() with
        | "status" ->
            do! client.GetServiceStatusAsync()
            do! Task.Delay(1000)
        
        | "doc-status" ->
            do! client.GetDocumentationStatusAsync()
            do! Task.Delay(1000)
        
        | "doc-start" ->
            do! client.StartDocumentationAsync()
            do! Task.Delay(1000)
        
        | "doc-pause" ->
            do! client.PauseDocumentationAsync()
            do! Task.Delay(1000)
        
        | "doc-resume" ->
            do! client.ResumeDocumentationAsync()
            do! Task.Delay(1000)
        
        | "doc-stop" ->
            do! client.StopDocumentationAsync()
            do! Task.Delay(1000)
        
        | "ping" ->
            do! client.PingAsync()
            do! Task.Delay(1000)
        
        | "interactive" | "i" ->
            do! this.DisconnectAsync()
            return! this.ExecuteInteractiveAsync(serviceUrl)
        
        | _ ->
            AnsiConsole.MarkupLine($"[red]❌ Unknown action: {action}[/]")
            return 1
        
        do! this.DisconnectAsync()
        return 0
    }
    
    /// Dispose resources
    interface IDisposable with
        member this.Dispose() =
            match webSocketClient with
            | Some client -> (client :> IDisposable).Dispose()
            | None -> ()
