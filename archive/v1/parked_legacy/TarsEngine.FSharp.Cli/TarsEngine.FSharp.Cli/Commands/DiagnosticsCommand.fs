namespace TarsEngine.FSharp.Cli.Commands

open System
open System.IO
open System.Net.Http
open System.Text.Json
open System.Diagnostics
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open Spectre.Console

type DiagnosticsResult = {
    Component: string
    Status: string
    Details: string
    ResponseTime: float
    Success: bool
}

type DiagnosticsCommand(logger: ILogger<DiagnosticsCommand>) =

    interface ICommand with
        member _.Name = "diagnostics"
        member _.Description = "Run comprehensive TARS system diagnostics"
        member _.Usage = "tars diagnostics"
        member _.Examples = ["tars diagnostics"]
        member _.ValidateOptions(_) = true

        member this.ExecuteAsync(options: CommandOptions) = task {
            if options.Help then
                return CommandResult.success "TARS Diagnostics - Run comprehensive system health checks"
            else
                do! this.RunDiagnosticsAsync()
                return CommandResult.success "Diagnostics completed successfully"
        }

    member this.RunDiagnosticsAsync() = task {
        AnsiConsole.Write(
            FigletText("TARS DIAGNOSTICS")
                .Centered()
                .Color(Color.Cyan1)
        )
        
        AnsiConsole.WriteLine()
        AnsiConsole.MarkupLine("[bold cyan]🔍 TARS System Diagnostics[/]")
        AnsiConsole.MarkupLine("[dim]Comprehensive system health check[/]")
        AnsiConsole.WriteLine()
        
        let results = ResizeArray<DiagnosticsResult>()
        
        // Create progress display
        let progress = AnsiConsole.Progress()
        progress.AutoRefresh <- true
        progress.HideCompleted <- false
        
        do! progress.StartAsync(fun ctx ->
            task {
                let task1 = ctx.AddTask("[green]System Information[/]")
                let task2 = ctx.AddTask("[blue]File System[/]")
                let task3 = ctx.AddTask("[yellow]Network Connectivity[/]")
                let task4 = ctx.AddTask("[red]AI Models[/]")
                let task5 = ctx.AddTask("[purple]Performance[/]")
                
                // Test 1: System Information
                let! sysResult = this.TestSystemInformation()
                results.Add(sysResult)
                task1.Increment(100.0)
                
                // Test 2: File System
                let! fsResult = this.TestFileSystem()
                results.Add(fsResult)
                task2.Increment(100.0)
                
                // Test 3: Network Connectivity
                let! netResult = this.TestNetworkConnectivity()
                results.Add(netResult)
                task3.Increment(100.0)
                
                // Test 4: AI Models
                let! aiResult = this.TestAiModels()
                results.Add(aiResult)
                task4.Increment(100.0)
                
                // Test 5: Performance
                let! perfResult = this.TestPerformance()
                results.Add(perfResult)
                task5.Increment(100.0)
            }
        )
        
        AnsiConsole.WriteLine()
        
        // Display results in a table
        let table = Table()
        table.AddColumn("Component") |> ignore
        table.AddColumn("Status") |> ignore
        table.AddColumn("Response Time") |> ignore
        table.AddColumn("Details") |> ignore
        
        for result in results do
            let statusMarkup = if result.Success then "[green]✅ PASS[/]" else "[red]❌ FAIL[/]"
            let timeMarkup = sprintf "[dim]%.1fms[/]" result.ResponseTime
            table.AddRow(result.Component, statusMarkup, timeMarkup, result.Details) |> ignore
        
        AnsiConsole.Write(table)
        AnsiConsole.WriteLine()
        
        // Summary
        let passedCount = results |> Seq.filter (fun r -> r.Success) |> Seq.length
        let totalCount = results.Count
        let successRate = float passedCount / float totalCount * 100.0
        
        if successRate >= 80.0 then
            AnsiConsole.MarkupLine($"[bold green]🎉 TARS System Health: EXCELLENT ({passedCount}/{totalCount} tests passed - {successRate:F1}%%)[/]")
        elif successRate >= 60.0 then
            AnsiConsole.MarkupLine($"[bold yellow]⚠️ TARS System Health: GOOD ({passedCount}/{totalCount} tests passed - {successRate:F1}%%)[/]")
        else
            AnsiConsole.MarkupLine($"[bold red]🚨 TARS System Health: NEEDS ATTENTION ({passedCount}/{totalCount} tests passed - {successRate:F1}%%)[/]")
        
        AnsiConsole.WriteLine()
        logger.LogInformation("Diagnostics completed: {PassedCount}/{TotalCount} tests passed", passedCount, totalCount)
    }
    
    member private _.TestSystemInformation() = task {
        let startTime = DateTime.UtcNow
        
        try
            let currentProcess = Process.GetCurrentProcess()
            let osVersion = Environment.OSVersion.ToString()
            let dotnetVersion = Environment.Version.ToString()
            let workingSet = currentProcess.WorkingSet64 / (1024L * 1024L)
            let processorCount = Environment.ProcessorCount
            
            let details = $"OS: {osVersion}, .NET: {dotnetVersion}, Memory: {workingSet}MB, CPUs: {processorCount}"
            let responseTime = (DateTime.UtcNow - startTime).TotalMilliseconds
            
            return {
                Component = "System Information"
                Status = "PASS"
                Details = details
                ResponseTime = responseTime
                Success = true
            }
        with
        | ex ->
            let responseTime = (DateTime.UtcNow - startTime).TotalMilliseconds
            return {
                Component = "System Information"
                Status = "FAIL"
                Details = ex.Message
                ResponseTime = responseTime
                Success = false
            }
    }
    
    member private _.TestFileSystem() = task {
        let startTime = DateTime.UtcNow
        
        try
            let currentDir = Directory.GetCurrentDirectory()
            let testFile = Path.Combine(currentDir, "tars_diagnostic_test.tmp")
            
            // Test write
            File.WriteAllText(testFile, "TARS Diagnostic Test")
            
            // Test read
            let content = File.ReadAllText(testFile)
            
            // Test delete
            File.Delete(testFile)
            
            let responseTime = (DateTime.UtcNow - startTime).TotalMilliseconds
            let details = $"Read/Write: OK, Directory: {currentDir}"
            
            return {
                Component = "File System"
                Status = "PASS"
                Details = details
                ResponseTime = responseTime
                Success = content = "TARS Diagnostic Test"
            }
        with
        | ex ->
            let responseTime = (DateTime.UtcNow - startTime).TotalMilliseconds
            return {
                Component = "File System"
                Status = "FAIL"
                Details = ex.Message
                ResponseTime = responseTime
                Success = false
            }
    }
    
    member private _.TestNetworkConnectivity() = task {
        let startTime = DateTime.UtcNow
        
        try
            use httpClient = new HttpClient()
            httpClient.Timeout <- TimeSpan.FromSeconds(10.0)
            
            // Test Ollama connectivity
            let! response = httpClient.GetAsync("http://localhost:11434/api/tags")
            let responseTime = (DateTime.UtcNow - startTime).TotalMilliseconds
            
            if response.IsSuccessStatusCode then
                let! content = response.Content.ReadAsStringAsync()
                let details = $"Ollama: Connected, Response: {content.Length} bytes"
                
                return {
                    Component = "Network Connectivity"
                    Status = "PASS"
                    Details = details
                    ResponseTime = responseTime
                    Success = true
                }
            else
                return {
                    Component = "Network Connectivity"
                    Status = "FAIL"
                    Details = $"Ollama: {response.StatusCode}"
                    ResponseTime = responseTime
                    Success = false
                }
        with
        | ex ->
            let responseTime = (DateTime.UtcNow - startTime).TotalMilliseconds
            return {
                Component = "Network Connectivity"
                Status = "FAIL"
                Details = ex.Message
                ResponseTime = responseTime
                Success = false
            }
    }
    
    member private _.TestAiModels() = task {
        let startTime = DateTime.UtcNow
        
        try
            use httpClient = new HttpClient()
            httpClient.Timeout <- TimeSpan.FromSeconds(15.0)
            
            let requestBody = JsonSerializer.Serialize({|
                model = "llama3:latest"
                prompt = "Test"
                stream = false
                options = {| temperature = 0.1; max_tokens = 5 |}
            |})
            
            let content = new System.Net.Http.StringContent(requestBody, System.Text.Encoding.UTF8, "application/json")
            let! response = httpClient.PostAsync("http://localhost:11434/api/generate", content)
            let responseTime = (DateTime.UtcNow - startTime).TotalMilliseconds
            
            if response.IsSuccessStatusCode then
                let! responseBody = response.Content.ReadAsStringAsync()
                let responseJson = JsonDocument.Parse(responseBody)
                let mutable responseElement = Unchecked.defaultof<JsonElement>
                let aiResponse = 
                    if responseJson.RootElement.TryGetProperty("response", &responseElement) then
                        responseElement.GetString()
                    else
                        "No response"
                
                let details = $"LLaMA3: Working, Response: '{aiResponse.Substring(0, min 20 aiResponse.Length)}...'"
                
                return {
                    Component = "AI Models"
                    Status = "PASS"
                    Details = details
                    ResponseTime = responseTime
                    Success = true
                }
            else
                return {
                    Component = "AI Models"
                    Status = "FAIL"
                    Details = $"AI Model Error: {response.StatusCode}"
                    ResponseTime = responseTime
                    Success = false
                }
        with
        | ex ->
            let responseTime = (DateTime.UtcNow - startTime).TotalMilliseconds
            return {
                Component = "AI Models"
                Status = "FAIL"
                Details = ex.Message
                ResponseTime = responseTime
                Success = false
            }
    }
    
    member private _.TestPerformance() = task {
        let startTime = DateTime.UtcNow
        
        try
            // CPU performance test
            let mutable result = 0.0
            for i in 1 .. 100000 do
                result <- result + Math.Sin(float i)
            
            let cpuTime = (DateTime.UtcNow - startTime).TotalMilliseconds
            
            // Memory test
            let beforeMemory = GC.GetTotalMemory(false)
            let testArray = Array.zeroCreate<byte> (1024 * 1024) // 1MB
            let afterMemory = GC.GetTotalMemory(false)
            let memoryDelta = afterMemory - beforeMemory
            
            let responseTime = (DateTime.UtcNow - startTime).TotalMilliseconds
            let details = $"CPU: {cpuTime:F1}ms, Memory: {memoryDelta / 1024L / 1024L}MB allocated"
            
            return {
                Component = "Performance"
                Status = "PASS"
                Details = details
                ResponseTime = responseTime
                Success = cpuTime < 1000.0 // Should complete in under 1 second
            }
        with
        | ex ->
            let responseTime = (DateTime.UtcNow - startTime).TotalMilliseconds
            return {
                Component = "Performance"
                Status = "FAIL"
                Details = ex.Message
                ResponseTime = responseTime
                Success = false
            }
    }
