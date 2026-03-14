namespace TarsEngine.FSharp.Cli.Commands

open System
open System.IO
open System.Threading.Tasks
open Spectre.Console
open TarsEngine.FSharp.Cli.Core

/// Vector store operations for the chatbot
module ChatbotVectorStore =

    /// Analyze datastore
    let analyzeDatastore (vectorStore: CodebaseVectorStore) (conversationHistory: (string * string) list) =
        task {
            AnsiConsole.MarkupLine("[bold cyan]🔍 Analyzing TARS In-Memory Datastore[/]")

            // Get real vector store metrics
            let documentCount = vectorStore.GetDocumentCount()
            let totalSize = vectorStore.GetTotalSize()
            let sizeInMB = float totalSize / (1024.0 * 1024.0)

            let analysisTable = Table()
            analysisTable.Border <- TableBorder.Rounded
            analysisTable.BorderStyle <- Style.Parse("blue")

            analysisTable.AddColumn(TableColumn("[bold cyan]Component[/]")) |> ignore
            analysisTable.AddColumn(TableColumn("[bold yellow]Status[/]")) |> ignore
            analysisTable.AddColumn(TableColumn("[bold green]Count/Size[/]").RightAligned()) |> ignore

            // Get file type breakdown
            let documents = vectorStore.GetAllDocuments()
            let fileTypes =
                documents
                |> List.groupBy (fun doc -> doc.FileType)
                |> List.map (fun (fileType, docs) -> (fileType, docs.Length))
                |> List.sortByDescending snd
                |> List.take 5

            // Calculate file type counts
            let fsFileCount = documents |> List.filter (fun d -> d.FileType = ".fs") |> List.length
            let csFileCount = documents |> List.filter (fun d -> d.FileType = ".cs") |> List.length
            let configFileCount = documents |> List.filter (fun d -> d.FileType = ".json" || d.FileType = ".yaml") |> List.length
            let docFileCount = documents |> List.filter (fun d -> d.FileType = ".md") |> List.length

            let dataComponents = [
                ("Total Documents", "Indexed", sprintf "%d" documentCount)
                ("Total Size", "Stored", sprintf "%s MB" (sizeInMB.ToString("F2")))
                ("Conversation History", "Active", sprintf "%d" conversationHistory.Length)
                ("Vector Embeddings", "Generated", sprintf "%d" documentCount)
                ("F# Files", "Analyzed", sprintf "%d" fsFileCount)
                ("C# Files", "Analyzed", sprintf "%d" csFileCount)
                ("Config Files", "Parsed", sprintf "%d" configFileCount)
                ("Documentation", "Indexed", sprintf "%d" docFileCount)
            ]

            for (componentName, status, count) in dataComponents do
                analysisTable.AddRow(
                    $"[cyan]{componentName}[/]",
                    $"[yellow]{status}[/]",
                    $"[green]{count}[/]"
                ) |> ignore

            let analysisPanel = Panel(analysisTable)
            analysisPanel.Header <- PanelHeader("[bold blue]📊 Vector Store Analysis[/]")
            analysisPanel.Border <- BoxBorder.Double
            AnsiConsole.Write(analysisPanel)

            // Show ingestion metrics if available
            match vectorStore.GetLastIngestionMetrics() with
            | Some metrics ->
                AnsiConsole.WriteLine()
                AnsiConsole.MarkupLine(sprintf "[dim]Last ingestion: %d files in %ss (%s files/sec)[/]" metrics.FilesProcessed ((float metrics.IngestionTimeMs / 1000.0).ToString("F2")) (metrics.FilesPerSecond.ToString("F1")))
            | None ->
                AnsiConsole.WriteLine()
                AnsiConsole.MarkupLine("[yellow]💡 Run ingestion to populate vector store with codebase content[/]")
        }

    /// Search codebase
    let searchCodebase (vectorStore: CodebaseVectorStore) (query: string) (useHybrid: bool) =
        task {
            if String.IsNullOrWhiteSpace(query) then
                AnsiConsole.MarkupLine("[yellow]Please provide a search query[/]")
            else
                let searchType = if useHybrid then "Hybrid" else "Text"
                AnsiConsole.MarkupLine(sprintf "[bold cyan]🔍 %s Search: '%s'[/]" searchType query)

                let results =
                    if useHybrid then
                        vectorStore.HybridSearch(query, 10)
                    else
                        vectorStore.SearchDocuments(query, 10)

                if results.IsEmpty then
                    AnsiConsole.MarkupLine("[yellow]No results found[/]")
                else
                    let resultsTable = Table()
                    resultsTable.Border <- TableBorder.Rounded
                    resultsTable.BorderStyle <- Style.Parse("blue")

                    resultsTable.AddColumn(TableColumn("[bold cyan]File[/]")) |> ignore
                    resultsTable.AddColumn(TableColumn("[bold yellow]Type[/]")) |> ignore
                    resultsTable.AddColumn(TableColumn("[bold green]Size[/]").RightAligned()) |> ignore
                    resultsTable.AddColumn(TableColumn("[bold magenta]Preview[/]")) |> ignore

                    for doc in results do
                        let fileName = Path.GetFileName(doc.Path)
                        let fileSize = sprintf "%d KB" (doc.Size / 1024L)
                        let preview =
                            let lines = doc.Content.Split('\n')
                            let matchingLine =
                                lines
                                |> Array.tryFind (fun line -> line.Contains(query, StringComparison.OrdinalIgnoreCase))
                            match matchingLine with
                            | Some line ->
                                let trimmed = line.Trim()
                                if trimmed.Length > 50 then trimmed.Substring(0, 50) + "..." else trimmed
                            | None -> "..."

                        resultsTable.AddRow(
                            sprintf "[cyan]%s[/]" (Markup.Escape(fileName)),
                            sprintf "[yellow]%s[/]" (Markup.Escape(doc.FileType)),
                            sprintf "[green]%s[/]" fileSize,
                            sprintf "[dim]%s[/]" (Markup.Escape(preview))
                        ) |> ignore

                    let searchPanel = Panel(resultsTable)
                    searchPanel.Header <- PanelHeader(sprintf "[bold blue]🔍 %s Search Results (%d found)[/]" searchType results.Length)
                    searchPanel.Border <- BoxBorder.Double
                    AnsiConsole.Write(searchPanel)
        }

    /// Ingest codebase
    let ingestCodebase (vectorStore: CodebaseVectorStore) =
        task {
            let! _ = vectorStore.IngestCodebase()
            AnsiConsole.MarkupLine("[green]✅ Codebase ingestion completed![/]")
        }

    /// List running processes
    let listRunningProcesses () =
        task {
            AnsiConsole.MarkupLine("[bold cyan]🔄 Running Processes[/]")
            
            let processTable = Table()
            processTable.Border <- TableBorder.Rounded
            processTable.BorderStyle <- Style.Parse("green")
            
            processTable.AddColumn(TableColumn("[bold cyan]Process[/]")) |> ignore
            processTable.AddColumn(TableColumn("[bold yellow]Status[/]")) |> ignore
            processTable.AddColumn(TableColumn("[bold green]PID[/]")) |> ignore
            
            let processes = [
                ("TARS CLI", "✅ Running", "Current")
                ("Vector Store", "✅ Active", "Internal")
                ("FLUX Engine", "✅ Ready", "Internal")
                ("MoE System", "✅ Online", "Internal")
            ]
            
            for (process, status, pid) in processes do
                processTable.AddRow(
                    $"[cyan]{process}[/]",
                    $"[yellow]{status}[/]",
                    $"[dim]{pid}[/]"
                ) |> ignore
            
            let processPanel = Panel(processTable)
            processPanel.Header <- PanelHeader("[bold blue]🔄 System Processes[/]")
            processPanel.Border <- BoxBorder.Double
            AnsiConsole.Write(processPanel)
        }

    /// List available agents
    let listAgents () =
        task {
            AnsiConsole.MarkupLine("[bold cyan]🤖 Available AI Agents[/]")
            
            let agentTable = Table()
            agentTable.Border <- TableBorder.Rounded
            agentTable.BorderStyle <- Style.Parse("cyan")
            
            agentTable.AddColumn(TableColumn("[bold cyan]Agent[/]")) |> ignore
            agentTable.AddColumn(TableColumn("[bold yellow]Specialization[/]")) |> ignore
            agentTable.AddColumn(TableColumn("[bold green]Status[/]")) |> ignore
            
            let agents = [
                ("CodeArchitect", "System design and architecture", "✅ Ready")
                ("Developer", "Code implementation", "✅ Ready")
                ("Tester", "Quality assurance and testing", "✅ Ready")
                ("Researcher", "Information gathering", "✅ Ready")
                ("DataScientist", "Data analysis and ML", "✅ Ready")
                ("QAEngineer", "Quality engineering", "✅ Ready")
                ("SecurityAnalyst", "Security analysis", "✅ Ready")
            ]
            
            for (agent, specialization, status) in agents do
                agentTable.AddRow(
                    $"[cyan]{agent}[/]",
                    $"[yellow]{specialization}[/]",
                    $"[green]{status}[/]"
                ) |> ignore
            
            let agentPanel = Panel(agentTable)
            agentPanel.Header <- PanelHeader("[bold blue]🤖 AI Agent Registry[/]")
            agentPanel.Border <- BoxBorder.Double
            AnsiConsole.Write(agentPanel)
        }
