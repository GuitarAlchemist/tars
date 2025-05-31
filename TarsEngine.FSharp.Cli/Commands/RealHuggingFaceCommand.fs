namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open Spectre.Console
open TarsEngine.FSharp.Cli.Services

/// Real HuggingFace integration command with actual model downloads
type RealHuggingFaceCommand(logger: ILogger<RealHuggingFaceCommand>, huggingFaceService: HuggingFaceService) =

    member private this.ShowRealHuggingFaceHeader() =
        AnsiConsole.Clear()
        
        let figlet = FigletText("REAL HF")
        figlet.Color <- Color.Green
        AnsiConsole.Write(figlet)
        
        let rule = Rule("[bold yellow]🚀 REAL HuggingFace Transformers Integration[/]")
        rule.Style <- Style.Parse("green")
        AnsiConsole.Write(rule)
        AnsiConsole.WriteLine()

    member private this.DownloadRealModel() =
        task {
            this.ShowRealHuggingFaceHeader()
            
            AnsiConsole.MarkupLine("[bold cyan]🤗 Real HuggingFace Model Download[/]")
            AnsiConsole.WriteLine()
            
            // Suggest some small, real models that are likely to work
            let suggestedModels = [
                "microsoft/DialoGPT-small"
                "distilbert-base-uncased"
                "microsoft/CodeBERT-base"
                "sentence-transformers/all-MiniLM-L6-v2"
                "google/flan-t5-small"
            ]
            
            let selectedModel = AnsiConsole.Prompt(
                (SelectionPrompt<string>())
                    .Title("[cyan]🎯 Select a real model to download:[/]")
                    .AddChoices(suggestedModels)
            )
            
            AnsiConsole.WriteLine()
            AnsiConsole.MarkupLine($"[bold yellow]📥 Downloading real model: {selectedModel}[/]")
            AnsiConsole.MarkupLine("[dim]This will attempt to download actual files from HuggingFace Hub...[/]")
            AnsiConsole.WriteLine()
            
            // Create progress display
            let progressPanel = Panel("")
            progressPanel.Header <- PanelHeader("[bold blue]📊 Download Progress[/]")
            progressPanel.Border <- BoxBorder.Rounded
            
            let mutable lastStatus = ""
            
            let onProgress (progress: DownloadProgress) =
                let statusText = $"""
[yellow]Model:[/] {progress.ModelId}
[cyan]Status:[/] {progress.Status}
[green]Downloaded:[/] {progress.BytesDownloaded / 1024L} KB
[blue]Total:[/] {progress.TotalBytes |> Option.map (fun t -> $"{t / 1024L} KB") |> Option.defaultValue "Unknown"}
[magenta]Progress:[/] {progress.Percentage |> Option.map (fun p -> $"{p:F1}%%") |> Option.defaultValue "N/A"}
"""
                
                if statusText <> lastStatus then
                    lastStatus <- statusText
                    progressPanel.UpdateContent(statusText)
                    AnsiConsole.Clear()
                    this.ShowRealHuggingFaceHeader()
                    AnsiConsole.MarkupLine($"[bold yellow]📥 Downloading real model: {selectedModel}[/]")
                    AnsiConsole.Write(progressPanel)
            
            let! result = huggingFaceService.DownloadModelAsync(selectedModel, onProgress)
            
            AnsiConsole.Clear()
            this.ShowRealHuggingFaceHeader()
            
            match result with
            | Ok modelInfo ->
                let sizeText = modelInfo.Size |> Option.map (fun s -> $"{s / 1024L} KB") |> Option.defaultValue "N/A"
                let pathText = modelInfo.LocalPath |> Option.defaultValue "N/A"
                let dateText = modelInfo.CreatedAt |> Option.map (fun d -> d.ToString("yyyy-MM-dd HH:mm")) |> Option.defaultValue "N/A"

                let successText = $"""[bold green]✅ Real Download Successful![/]

[cyan]Model:[/] {modelInfo.Name}
[cyan]ID:[/] {modelInfo.ModelId}
[cyan]Local Path:[/] {pathText}
[cyan]Size:[/] {sizeText}
[cyan]Downloaded:[/] {dateText}

[bold yellow]🎯 Real model files are now available locally![/]
[dim]You can find the files in your ~/.tars/models/huggingface/ directory[/]"""

                let successPanel = Panel(successText)
                successPanel.Header <- PanelHeader("[bold green]🎉 Real Download Complete[/]")
                successPanel.Border <- BoxBorder.Double
                AnsiConsole.Write(successPanel)

                AnsiConsole.WriteLine()
                AnsiConsole.MarkupLine("[bold blue]🔍 Would you like to test inference with this model? (y/n)[/]")
                let testInference = Console.ReadKey(true).KeyChar.ToString().ToLower() = "y"

                if testInference then
                    do! this.TestRealInference(modelInfo)
                
            | Error error ->
                AnsiConsole.MarkupLine($"[red]❌ Real download failed: {error}[/]")
                AnsiConsole.MarkupLine("[yellow]💡 This might be due to network issues or model availability[/]")
        }

    member private this.TestRealInference(modelInfo: ModelInfo) =
        task {
            AnsiConsole.WriteLine()
            AnsiConsole.MarkupLine("[bold cyan]🧠 Testing Real Model Inference[/]")
            
            let prompt = AnsiConsole.Ask<string>("[cyan]💭 Enter your prompt for the real model:[/]")
            
            AnsiConsole.WriteLine()
            AnsiConsole.MarkupLine($"[dim]🧠 Running real inference with {modelInfo.Name}...[/]")
            
            let! result = huggingFaceService.GenerateTextAsync(modelInfo, prompt)
            match result with
            | Ok response ->
                let responsePanel = Panel(response)
                responsePanel.Header <- PanelHeader($"[bold green]🤖 Real {modelInfo.Name} Response[/]")
                responsePanel.Border <- BoxBorder.Rounded
                AnsiConsole.Write(responsePanel)
                
                AnsiConsole.WriteLine()
                AnsiConsole.MarkupLine("[bold green]✅ Real inference completed![/]")
            | Error error ->
                AnsiConsole.MarkupLine($"[red]❌ Real inference failed: {error}[/]")
                AnsiConsole.MarkupLine("[yellow]💡 The model might need ONNX conversion or additional setup[/]")
        }

    member private this.ShowRealLocalModels() =
        task {
            this.ShowRealHuggingFaceHeader()
            
            AnsiConsole.MarkupLine("[bold cyan]💾 Real Local HuggingFace Models[/]")
            AnsiConsole.WriteLine()
            
            let! result = huggingFaceService.GetLocalModelsAsync()
            match result with
            | Ok models ->
                if models.IsEmpty then
                    AnsiConsole.MarkupLine("[yellow]No real models found locally.[/]")
                    AnsiConsole.MarkupLine("[dim]Use 'download' command to get real models from HuggingFace Hub.[/]")
                else
                    let table = Table()
                    table.Border <- TableBorder.Rounded
                    table.BorderStyle <- Style.Parse("blue")
                    
                    table.AddColumn(TableColumn("[bold cyan]Model[/]")) |> ignore
                    table.AddColumn(TableColumn("[bold yellow]ID[/]")) |> ignore
                    table.AddColumn(TableColumn("[bold green]Size[/]").RightAligned()) |> ignore
                    table.AddColumn(TableColumn("[bold magenta]Downloaded[/]")) |> ignore
                    table.AddColumn(TableColumn("[bold blue]Files[/]").Centered()) |> ignore
                    
                    for model in models do
                        let size = 
                            model.Size 
                            |> Option.map (fun s -> $"{s / 1024L} KB")
                            |> Option.defaultValue "N/A"
                        
                        let downloaded =
                            model.CreatedAt
                            |> Option.map (fun d -> d.ToString("yyyy-MM-dd"))
                            |> Option.defaultValue "N/A"
                        
                        // Check what files exist
                        let filesExist = 
                            match model.LocalPath with
                            | Some path ->
                                let files = System.IO.Directory.GetFiles(path)
                                $"{files.Length} files"
                            | None -> "N/A"
                        
                        table.AddRow(
                            $"[bold]{model.Name}[/]",
                            $"[dim]{model.ModelId}[/]",
                            $"[green]{size}[/]",
                            downloaded,
                            $"[blue]{filesExist}[/]"
                        ) |> ignore
                    
                    let panel = Panel(table)
                    panel.Header <- PanelHeader("[bold blue]💾 Real Local Model Storage[/]")
                    panel.Border <- BoxBorder.Double
                    AnsiConsole.Write(panel)
                    
                    // Show storage stats
                    let! statsResult = huggingFaceService.GetStorageStatsAsync()
                    match statsResult with
                    | Ok stats ->
                        AnsiConsole.WriteLine()
                        AnsiConsole.MarkupLine($"[dim]📊 Storage: {stats.TotalModels} models, {stats.TotalSize / 1024L} KB total[/]")
                        AnsiConsole.MarkupLine($"[dim]📁 Location: {stats.Directory}[/]")
                    | Error _ -> ()
            | Error error ->
                AnsiConsole.MarkupLine($"[red]❌ Failed to list real local models: {error}[/]")
        }

    member private this.SearchRealModels() =
        task {
            this.ShowRealHuggingFaceHeader()
            
            let query = AnsiConsole.Ask<string>("[cyan]🔍 Enter search query for real HuggingFace models:[/]")
            
            AnsiConsole.WriteLine()
            AnsiConsole.MarkupLine($"[dim]Searching real HuggingFace Hub for: {query}...[/]")
            
            let! result = huggingFaceService.SearchModelsAsync(query, 10)
            match result with
            | Ok models ->
                if models.IsEmpty then
                    AnsiConsole.MarkupLine("[yellow]No real models found for your query.[/]")
                else
                    let table = Table()
                    table.Border <- TableBorder.Rounded
                    table.BorderStyle <- Style.Parse("cyan")
                    
                    table.AddColumn(TableColumn("[bold cyan]Model ID[/]")) |> ignore
                    table.AddColumn(TableColumn("[bold yellow]Description[/]")) |> ignore
                    table.AddColumn(TableColumn("[bold green]Downloads[/]").RightAligned()) |> ignore
                    table.AddColumn(TableColumn("[bold blue]Tags[/]")) |> ignore
                    
                    for model in models |> List.take (Math.Min(10, models.Length)) do
                        let downloads = 
                            model.Downloads 
                            |> Option.map (fun d -> d.ToString("N0"))
                            |> Option.defaultValue "N/A"
                        
                        let tags = 
                            model.Tags 
                            |> List.take (Math.Min(3, model.Tags.Length))
                            |> String.concat ", "
                        
                        table.AddRow(
                            $"[cyan]{model.ModelId}[/]",
                            model.Description,
                            $"[green]{downloads}[/]",
                            $"[dim]{tags}[/]"
                        ) |> ignore
                    
                    let panel = Panel(table)
                    panel.Header <- PanelHeader($"[bold cyan]🔍 Real Search Results: {query}[/]")
                    panel.Border <- BoxBorder.Rounded
                    AnsiConsole.Write(panel)
                    
                    AnsiConsole.WriteLine()
                    AnsiConsole.MarkupLine($"[green]Found {models.Length} real models matching '{query}'[/]")
            | Error error ->
                AnsiConsole.MarkupLine($"[red]❌ Real search failed: {error}[/]")
                AnsiConsole.MarkupLine("[yellow]💡 This might be due to network connectivity issues[/]")
        }

    interface ICommand with
        member _.Name = "realhf"
        member _.Description = "Real HuggingFace Transformers integration with actual downloads"
        member _.Usage = "tars realhf [download|local|search|test]"
        member _.Examples = [
            "tars realhf download"
            "tars realhf local"
            "tars realhf search"
        ]
        member _.ValidateOptions(options) = true

        member this.ExecuteAsync(options) =
            task {
                try
                    match options.Arguments with
                    | "download" :: _ ->
                        do! this.DownloadRealModel()
                        return CommandResult.success("Real model download completed")
                    | "local" :: _ ->
                        do! this.ShowRealLocalModels()
                        return CommandResult.success("Real local models displayed")
                    | "search" :: _ ->
                        do! this.SearchRealModels()
                        return CommandResult.success("Real model search completed")
                    | [] ->
                        do! this.DownloadRealModel()
                        return CommandResult.success("Real HuggingFace integration demo completed")
                    | unknown :: _ ->
                        AnsiConsole.MarkupLine($"[red]❌ Unknown realhf command: {unknown}[/]")
                        return CommandResult.failure($"Unknown command: {unknown}")
                with
                | ex ->
                    logger.LogError(ex, "Error in real huggingface command")
                    AnsiConsole.MarkupLine($"[red]❌ Error: {ex.Message}[/]")
                    return CommandResult.failure(ex.Message)
            }
