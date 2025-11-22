namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open Spectre.Console
open TarsEngine.FSharp.Cli.Services

/// HuggingFace integration command for TARS
type HuggingFaceCommand(logger: ILogger<HuggingFaceCommand>, huggingFaceService: HuggingFaceService) =

    member private self.ShowHuggingFaceHeader() =
        AnsiConsole.Clear()
        
        let figlet = FigletText("TARS + HF")
        figlet.Color <- Color.Green
        AnsiConsole.Write(figlet)
        
        let rule = Rule("[bold yellow]HuggingFace Transformers Integration[/]")
        rule.Style <- Style.Parse("green")
        AnsiConsole.Write(rule)
        AnsiConsole.WriteLine()

    member private self.ShowRecommendedModels() =
        task {
            self.ShowHuggingFaceHeader()
            
            AnsiConsole.MarkupLine("[bold cyan]🤗 Recommended HuggingFace Models for TARS[/]")
            AnsiConsole.WriteLine()
            
            AnsiConsole.MarkupLine("[dim]Fetching recommended models...[/]")
            
            let! result = huggingFaceService.GetRecommendedModelsAsync()
            match result with
            | Ok models ->
                let table = Table()
                table.Border <- TableBorder.Rounded
                table.BorderStyle <- Style.Parse("green")
                
                table.AddColumn(TableColumn("[bold cyan]Model[/]").Centered()) |> ignore
                table.AddColumn(TableColumn("[bold yellow]Capability[/]").Centered()) |> ignore
                table.AddColumn(TableColumn("[bold magenta]Expert Type[/]").Centered()) |> ignore
                table.AddColumn(TableColumn("[bold blue]Status[/]").Centered()) |> ignore
                
                for model in models do
                    let expertType = 
                        model.ExpertType 
                        |> Option.map string 
                        |> Option.defaultValue "General"
                    
                    let status = if model.IsLoaded then "[green]✓ Loaded[/]" else "[yellow]Available[/]"
                    
                    table.AddRow(
                        $"[bold]{model.ModelInfo.Name}[/]",
                        $"[yellow]{model.Capability}[/]",
                        $"[magenta]{expertType}[/]",
                        status
                    ) |> ignore
                
                let panel = Panel(table)
                panel.Header <- PanelHeader("[bold green]🎯 TARS-Optimized Models[/]")
                panel.Border <- BoxBorder.Double
                AnsiConsole.Write(panel)
                
                AnsiConsole.WriteLine()
                AnsiConsole.MarkupLine("[bold green]✅ These models are optimized for TARS expert system integration![/]")
            | Error error ->
                AnsiConsole.MarkupLine($"[red]❌ Failed to fetch models: {error}[/]")
        }

    member private self.SearchModels() =
        task {
            self.ShowHuggingFaceHeader()
            
            let query = AnsiConsole.Ask<string>("[cyan]🔍 Enter search query for HuggingFace models:[/]")
            
            AnsiConsole.WriteLine()
            AnsiConsole.MarkupLine($"[dim]Searching HuggingFace Hub for: {query}...[/]")
            
            let! result = huggingFaceService.SearchModelsAsync(query, 10)
            match result with
            | Ok models ->
                if models.IsEmpty then
                    AnsiConsole.MarkupLine("[yellow]No models found for your query.[/]")
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
                    panel.Header <- PanelHeader($"[bold cyan]🔍 Search Results: {query}[/]")
                    panel.Border <- BoxBorder.Rounded
                    AnsiConsole.Write(panel)
                    
                    AnsiConsole.WriteLine()
                    AnsiConsole.MarkupLine($"[green]Found {models.Length} models matching '{query}'[/]")
            | Error error ->
                AnsiConsole.MarkupLine($"[red]❌ Search failed: {error}[/]")
        }

    member private self.DownloadModel() =
        task {
            self.ShowHuggingFaceHeader()
            
            let modelId = AnsiConsole.Ask<string>("[cyan]📥 Enter model ID to download (e.g., microsoft/DialoGPT-medium):[/]")
            
            AnsiConsole.WriteLine()
            AnsiConsole.MarkupLine($"[bold yellow]📥 Downloading model: {modelId}[/]")
            
            // Create progress display
            let progressTable = Table()
            progressTable.Border <- TableBorder.None
            progressTable.AddColumn("Status") |> ignore
            progressTable.AddColumn("Progress") |> ignore
            
            let mutable currentProgress = ""
            
            let onProgress (progress: DownloadProgress) =
                let percentageText = progress.Percentage |> Option.map (fun p -> $"{p:F1}%%") |> Option.defaultValue "N/A"
                currentProgress <- $"{progress.Status} - {percentageText}"
                
                progressTable.Rows.Clear()
                progressTable.AddRow(
                    "[yellow]Downloading...[/]",
                    currentProgress
                ) |> ignore
                
                AnsiConsole.Clear()
                self.ShowHuggingFaceHeader()
                AnsiConsole.MarkupLine($"[bold yellow]📥 Downloading model: {modelId}[/]")
                AnsiConsole.Write(progressTable)
            
            let! result = huggingFaceService.DownloadModelAsync(modelId, onProgress)
            
            AnsiConsole.Clear()
            self.ShowHuggingFaceHeader()
            
            match result with
            | Ok modelInfo ->
                let sizeText = modelInfo.Size |> Option.map (fun s -> $"{s / 1024L} KB") |> Option.defaultValue "N/A"
                let pathText = modelInfo.LocalPath |> Option.defaultValue "N/A"
                let dateText = modelInfo.CreatedAt |> Option.map (fun d -> d.ToString("yyyy-MM-dd HH:mm")) |> Option.defaultValue "N/A"

                let successText = $"""[bold green]✅ Download Successful![/]

[cyan]Model:[/] {modelInfo.Name}
[cyan]ID:[/] {modelInfo.ModelId}
[cyan]Local Path:[/] {pathText}
[cyan]Size:[/] {sizeText}
[cyan]Downloaded:[/] {dateText}

[bold yellow]🎯 Model is now ready for TARS integration![/]"""

                let successPanel = Panel(successText)
                successPanel.Header <- PanelHeader("[bold green]Download Complete[/]")
                successPanel.Border <- BoxBorder.Double
                AnsiConsole.Write(successPanel)
            | Error error ->
                AnsiConsole.MarkupLine($"[red]❌ Download failed: {error}[/]")
        }

    member private self.ShowLocalModels() =
        task {
            self.ShowHuggingFaceHeader()
            
            AnsiConsole.MarkupLine("[bold cyan]💾 Local HuggingFace Models[/]")
            AnsiConsole.WriteLine()
            
            let! result = huggingFaceService.GetLocalModelsAsync()
            match result with
            | Ok models ->
                if models.IsEmpty then
                    AnsiConsole.MarkupLine("[yellow]No local models found. Use 'download' command to get models.[/]")
                else
                    let table = Table()
                    table.Border <- TableBorder.Rounded
                    table.BorderStyle <- Style.Parse("blue")
                    
                    table.AddColumn(TableColumn("[bold cyan]Model[/]")) |> ignore
                    table.AddColumn(TableColumn("[bold yellow]ID[/]")) |> ignore
                    table.AddColumn(TableColumn("[bold green]Size[/]").RightAligned()) |> ignore
                    table.AddColumn(TableColumn("[bold magenta]Downloaded[/]")) |> ignore
                    table.AddColumn(TableColumn("[bold blue]Status[/]").Centered()) |> ignore
                    
                    for model in models do
                        let size = 
                            model.Size 
                            |> Option.map (fun s -> $"{s / 1024L / 1024L} MB")
                            |> Option.defaultValue "N/A"
                        
                        let downloaded = 
                            model.CreatedAt 
                            |> Option.map (_.ToString("yyyy-MM-dd"))
                            |> Option.defaultValue "N/A"
                        
                        table.AddRow(
                            $"[bold]{model.Name}[/]",
                            $"[dim]{model.ModelId}[/]",
                            $"[green]{size}[/]",
                            downloaded,
                            "[green]✓ Ready[/]"
                        ) |> ignore
                    
                    let panel = Panel(table)
                    panel.Header <- PanelHeader("[bold blue]💾 Local Model Storage[/]")
                    panel.Border <- BoxBorder.Double
                    AnsiConsole.Write(panel)
                    
                    // Show storage stats
                    let! statsResult = huggingFaceService.GetStorageStatsAsync()
                    match statsResult with
                    | Ok stats ->
                        AnsiConsole.WriteLine()
                        AnsiConsole.MarkupLine($"[dim]📊 Storage: {stats.TotalModels} models, {stats.TotalSize / 1024L / 1024L} MB total[/]")
                        AnsiConsole.MarkupLine($"[dim]📁 Location: {stats.Directory}[/]")
                    | Error _ -> ()
            | Error error ->
                AnsiConsole.MarkupLine($"[red]❌ Failed to list local models: {error}[/]")
        }

    member private self.TestInference() =
        task {
            self.ShowHuggingFaceHeader()
            
            let! localModelsResult = huggingFaceService.GetLocalModelsAsync()
            match localModelsResult with
            | Ok models when not models.IsEmpty ->
                let modelChoices = 
                    models 
                    |> List.map (fun m -> m.ModelId)
                    |> List.toArray
                
                let selectedModelId = AnsiConsole.Prompt(
                    (SelectionPrompt<string>())
                        .Title("[cyan]🧠 Select model for inference test:[/]")
                        .AddChoices(modelChoices)
                )
                
                let selectedModel = models |> List.find (fun m -> m.ModelId = selectedModelId)
                
                let prompt = AnsiConsole.Ask<string>("[cyan]💭 Enter your prompt:[/]")
                
                AnsiConsole.WriteLine()
                AnsiConsole.MarkupLine($"[dim]🧠 Generating response with {selectedModel.Name}...[/]")
                
                let! result = huggingFaceService.GenerateTextAsync(selectedModel, prompt)
                match result with
                | Ok response ->
                    let responsePanel = Panel(response)
                    responsePanel.Header <- PanelHeader($"[bold green]🤖 {selectedModel.Name} Response[/]")
                    responsePanel.Border <- BoxBorder.Rounded
                    AnsiConsole.Write(responsePanel)
                    
                    AnsiConsole.WriteLine()
                    AnsiConsole.MarkupLine("[bold green]✅ Inference completed successfully![/]")
                | Error error ->
                    AnsiConsole.MarkupLine($"[red]❌ Inference failed: {error}[/]")
            | Ok _ ->
                AnsiConsole.MarkupLine("[yellow]No local models available. Download a model first.[/]")
            | Error error ->
                AnsiConsole.MarkupLine($"[red]❌ Failed to list models: {error}[/]")
        }

    interface ICommand with
        member _.Name = "huggingface"
        member _.Description = "HuggingFace Transformers integration for TARS"
        member self.Usage = "tars huggingface [recommended|search|download|local|test]"
        member self.Examples = [
            "tars huggingface recommended"
            "tars huggingface search"
            "tars huggingface download"
            "tars huggingface local"
            "tars huggingface test"
        ]
        member self.ValidateOptions(options) = true

        member self.ExecuteAsync(options) =
            task {
                try
                    match options.Arguments with
                    | "recommended" :: _ ->
                        do! self.ShowRecommendedModels()
                        return CommandResult.success("Recommended models displayed")
                    | "search" :: _ ->
                        do! self.SearchModels()
                        return CommandResult.success("Model search completed")
                    | "download" :: _ ->
                        do! self.DownloadModel()
                        return CommandResult.success("Model download completed")
                    | "local" :: _ ->
                        do! self.ShowLocalModels()
                        return CommandResult.success("Local models displayed")
                    | "test" :: _ ->
                        do! self.TestInference()
                        return CommandResult.success("Inference test completed")
                    | [] ->
                        do! self.ShowRecommendedModels()
                        return CommandResult.success("HuggingFace integration demo completed")
                    | unknown :: _ ->
                        AnsiConsole.MarkupLine($"[red]❌ Unknown huggingface command: {unknown}[/]")
                        return CommandResult.failure($"Unknown command: {unknown}")
                with
                | ex ->
                    logger.LogError(ex, "Error in huggingface command")
                    AnsiConsole.MarkupLine($"[red]❌ Error: {ex.Message}[/]")
                    return CommandResult.failure(ex.Message)
            }
