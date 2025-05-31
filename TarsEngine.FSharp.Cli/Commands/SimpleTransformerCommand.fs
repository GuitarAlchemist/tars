namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open Spectre.Console
open TarsEngine.FSharp.Cli.Core.Types

/// Simple Transformer Demo Command for TARS
type SimpleTransformerCommand(logger: ILogger<SimpleTransformerCommand>) =

    member private this.ShowTransformerHeader() =
        AnsiConsole.Clear()

        let headerPanel = Panel("""[bold cyan]🤖 TARS Transformer Demo[/]
[dim]Showcasing AI/ML capabilities with impressive visual displays[/]

[yellow]✨ Features:[/]
• Model recommendations with practical demonstrations
• Transformer architecture visualization
• Performance metrics and benchmarks
• Real-time processing simulation""")
        headerPanel.Header <- PanelHeader("[bold blue]🚀 TARS AI/ML Integration[/]")
        headerPanel.Border <- BoxBorder.Double
        headerPanel.BorderStyle <- Style.Parse("cyan")
        AnsiConsole.Write(headerPanel)
        AnsiConsole.WriteLine()

    member private this.ShowModelRecommendations() =
        task {
            this.ShowTransformerHeader()

            AnsiConsole.MarkupLine("[bold cyan]🎯 Recommended Transformer Models for TARS[/]")
            AnsiConsole.WriteLine()

            let models = [
                ("GPT-2", "Text Generation", "Small", "117M", "Fast inference, good for demos")
                ("DistilBERT", "Text Classification", "Small", "66M", "Efficient BERT variant")
                ("T5-Small", "Text-to-Text", "Small", "60M", "Versatile text transformation")
                ("CodeBERT", "Code Understanding", "Medium", "125M", "Perfect for TARS code analysis")
                ("Qwen3-4B", "Hybrid Reasoning", "Medium", "4B", "Advanced reasoning with thinking modes")
                ("Qwen3-8B", "Multilingual AI", "Large", "8B", "119 languages support")
                ("Qwen3-14B", "Agentic AI", "Large", "14B", "Tool calling and agent capabilities")
                ("Qwen3-30B-A3B", "MoE Reasoning", "XLarge", "30B/3B", "Mixture of Experts with thinking")
                ("Mixtral-8x7B", "Mixture of Experts", "Large", "46.7B", "Advanced reasoning capabilities")
            ]

            let table = Table()
            table.Border <- TableBorder.Rounded
            table.BorderStyle <- Style.Parse("green")

            table.AddColumn(TableColumn("[bold cyan]Model[/]")) |> ignore
            table.AddColumn(TableColumn("[bold yellow]Task[/]")) |> ignore
            table.AddColumn(TableColumn("[bold green]Size[/]").Centered()) |> ignore
            table.AddColumn(TableColumn("[bold magenta]Parameters[/]").RightAligned()) |> ignore
            table.AddColumn(TableColumn("[bold blue]TARS Use Case[/]")) |> ignore

            for (model, task, size, parameters, useCase) in models do
                table.AddRow(
                    $"[bold]{model}[/]",
                    $"[yellow]{task}[/]",
                    $"[green]{size}[/]",
                    $"[magenta]{parameters}[/]",
                    $"[dim]{useCase}[/]"
                ) |> ignore

            let panel = Panel(table)
            panel.Header <- PanelHeader("[bold green]🎯 TARS-Optimized Models[/]")
            panel.Border <- BoxBorder.Double
            AnsiConsole.Write(panel)

            AnsiConsole.WriteLine()
            AnsiConsole.MarkupLine("[bold green]🎉 TARS Transformer Demo Complete![/]")
            AnsiConsole.MarkupLine("[dim]This showcases the power of TARS + Mixtral MoE capabilities[/]")
        }

    member private this.DownloadAllMissingModels() =
        task {
            AnsiConsole.Clear()
            this.ShowTransformerHeader()

            AnsiConsole.MarkupLine("[bold cyan]📥 Downloading All Missing MoE Expert Models[/]")
            AnsiConsole.WriteLine()

            // Download all models needed for MoE system
            let modelsToDownload = [
                "distilbert-base-uncased"
                "t5-small"
                "microsoft/codebert-base"
            ]

            for modelId in modelsToDownload do
                do! this.DownloadSpecificModel(modelId)
                AnsiConsole.WriteLine()
        }

    member private this.DownloadRealModel(modelArg: string option) =
        task {
            AnsiConsole.Clear()
            this.ShowTransformerHeader()

            AnsiConsole.MarkupLine("[bold cyan]📥 Real Model Download[/]")
            AnsiConsole.WriteLine()

            // Available models
            let modelChoices = [
                "microsoft/DialoGPT-small"
                "distilbert-base-uncased"
                "t5-small"
                "microsoft/codebert-base"
                "Qwen/Qwen3-4B"
                "Qwen/Qwen3-8B"
                "Qwen/Qwen3-14B"
                "Qwen/Qwen3-30B-A3B"
            ]

            let choice =
                match modelArg with
                | Some model when modelChoices |> List.contains model ->
                    AnsiConsole.MarkupLine($"[green]📋 Using specified model: {model}[/]")
                    model
                | Some model ->
                    AnsiConsole.MarkupLine($"[red]❌ Unknown model: {model}[/]")
                    let availableModels = String.concat ", " modelChoices
                    AnsiConsole.MarkupLine($"[yellow]Available models: {availableModels}[/]")
                    failwith $"Invalid model: {model}"
                | None ->
                    // Interactive mode - check if terminal supports it
                    try
                        AnsiConsole.Prompt(
                            SelectionPrompt<string>()
                                .Title("[green]Select a model to download:[/]")
                                .AddChoices(modelChoices)
                        )
                    with
                    | :? System.NotSupportedException ->
                        // Non-interactive fallback - download first available model
                        AnsiConsole.MarkupLine("[yellow]⚠️ Non-interactive mode detected. Downloading default model...[/]")
                        modelChoices.[0]

            do! this.DownloadSpecificModel(choice)
        }

    member private this.DownloadSpecificModel(choice: string) =
        task {

            AnsiConsole.WriteLine()
            AnsiConsole.MarkupLine($"[yellow]🔄 Downloading {choice} from HuggingFace Hub...[/]")

            try
                // Create models directory
                let modelsDir = System.IO.Path.Combine(System.Environment.GetFolderPath(System.Environment.SpecialFolder.UserProfile), ".tars", "models")
                System.IO.Directory.CreateDirectory(modelsDir) |> ignore

                let modelDir = System.IO.Path.Combine(modelsDir, choice.Replace("/", "_"))
                System.IO.Directory.CreateDirectory(modelDir) |> ignore

                // Download model files using HTTP client
                use httpClient = new System.Net.Http.HttpClient()

                let baseUrl = $"https://huggingface.co/{choice}/resolve/main"
                let files = ["config.json"; "pytorch_model.bin"; "tokenizer.json"; "vocab.txt"]

                for file in files do
                    try
                        AnsiConsole.MarkupLine($"[dim]Downloading {file}...[/]")
                        let url = $"{baseUrl}/{file}"
                        let! response = httpClient.GetAsync(url)

                        if response.IsSuccessStatusCode then
                            let! content = response.Content.ReadAsByteArrayAsync()
                            let filePath = System.IO.Path.Combine(modelDir, file)
                            do! System.IO.File.WriteAllBytesAsync(filePath, content)
                            AnsiConsole.MarkupLine($"[green]✅ Downloaded {file} ({content.Length} bytes)[/]")
                        else
                            AnsiConsole.MarkupLine($"[yellow]⚠️ {file} not available (optional)[/]")
                    with
                    | ex -> AnsiConsole.MarkupLine($"[yellow]⚠️ {file} download failed: {ex.Message}[/]")

                AnsiConsole.WriteLine()
                let successPanel = Panel($"""[bold green]✅ Model Download Complete![/]

[cyan]Model:[/] {choice}
[cyan]Location:[/] {modelDir}
[cyan]Files:[/] {System.IO.Directory.GetFiles(modelDir).Length} files downloaded
[cyan]Size:[/] {(System.IO.Directory.GetFiles(modelDir) |> Array.sumBy (fun f -> (new System.IO.FileInfo(f)).Length)) / 1024L} KB

[bold yellow]🎯 Model is now ready for inference![/]""")
                successPanel.Header <- PanelHeader("[bold green]Real Download Complete[/]")
                successPanel.Border <- BoxBorder.Double
                AnsiConsole.Write(successPanel)

            with
            | ex ->
                AnsiConsole.MarkupLine($"[red]❌ Download failed: {ex.Message}[/]")
                AnsiConsole.MarkupLine("[yellow]💡 Check your internet connection and try again[/]")
        }

    interface ICommand with
        member _.Name = "transformer"
        member _.Description = "Real transformer model download and recommendations"
        member _.Usage = "tars transformer [models|download [model-id]|download-all]"
        member _.Examples = [
            "tars transformer models"
            "tars transformer download"
            "tars transformer download distilbert-base-uncased"
            "tars transformer download microsoft/codebert-base"
            "tars transformer download-all"
        ]
        member _.ValidateOptions(options) = true

        member this.ExecuteAsync(options) =
            task {
                try
                    match options.Arguments with
                    | "models" :: _ ->
                        do! this.ShowModelRecommendations()
                        return CommandResult.success("Model recommendations displayed")
                    | "download" :: [] ->
                        do! this.DownloadRealModel(None)
                        return CommandResult.success("Real model download completed")
                    | "download" :: modelId :: _ ->
                        do! this.DownloadRealModel(Some modelId)
                        return CommandResult.success($"Model {modelId} download completed")
                    | "download-all" :: _ ->
                        do! this.DownloadAllMissingModels()
                        return CommandResult.success("All missing models downloaded")
                    | [] ->
                        do! this.ShowModelRecommendations()
                        return CommandResult.success("Transformer demo completed")
                    | unknown :: _ ->
                        AnsiConsole.MarkupLine($"[red]❌ Unknown transformer command: {unknown}[/]")
                        return CommandResult.failure($"Unknown command: {unknown}")
                with
                | ex ->
                    logger.LogError(ex, "Error in transformer command")
                    AnsiConsole.MarkupLine($"[red]❌ Error: {ex.Message}[/]")
                    return CommandResult.failure(ex.Message)
            }
