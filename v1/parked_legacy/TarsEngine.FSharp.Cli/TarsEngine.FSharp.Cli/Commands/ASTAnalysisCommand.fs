namespace TarsEngine.FSharp.Cli.Commands

open System
open System.IO
open System.Threading.Tasks
open TarsEngine.FSharp.Cli.CodeProtection.RAGCodeAnalyzer
open Spectre.Console

/// Command to demonstrate AST-based code analysis
type ASTAnalysisCommand() =
    interface ICommand with
        member _.Name = "ast"
        member _.Description = "Analyze F# code using real AST (Abstract Syntax Tree) analysis"
        member _.Usage = "tars ast [file] [--search <query>]"
        member _.Examples = [
            "tars ast Program.fs"
            "tars ast --search \"context retrieval\""
        ]
        member _.ValidateOptions(_options: CommandOptions) = true
        
        member _.ExecuteAsync(options: CommandOptions) =
            Task.Run(fun () ->
                try
                    let baseDir = Directory.GetCurrentDirectory()
                    
                    AnsiConsole.MarkupLine("[bold cyan]🌳 AST-Based Code Analysis[/]")
                    AnsiConsole.MarkupLine("[dim]Real Abstract Syntax Tree analysis for F# code[/]")
                    AnsiConsole.WriteLine()
                    
                    // Check for specific file analysis
                    let targetFile = 
                        match options.Arguments with
                        | file :: _ when File.Exists(file) -> Some file
                        | _ -> 
                            match options.Options.TryFind("file") with
                            | Some file when File.Exists(file) -> Some file
                            | _ -> None
                    
                    match targetFile with
                    | Some filePath ->
                        // Analyze specific file
                        AnsiConsole.MarkupLine($"[yellow]🔍 Analyzing file: {Path.GetFileName(filePath)}[/]")
                        AnsiConsole.WriteLine()
                        
                        match extractCodeContext filePath with
                        | Some context ->
                            // Display AST analysis results
                            let table = Table()
                            table.AddColumn("Metric") |> ignore
                            table.AddColumn("Value") |> ignore
                            table.AddColumn("Details") |> ignore
                            
                            table.AddRow("Language", context.Language, "") |> ignore
                            table.AddRow("AST Available", (if context.AST.IsSome then "✅ Yes" else "❌ No"), "") |> ignore
                            table.AddRow("Functions", context.Functions.Length.ToString(), String.Join(", ", context.Functions |> List.take (min 5 context.Functions.Length) |> List.map (fun f -> f.Name))) |> ignore
                            table.AddRow("Types", context.Types.Length.ToString(), String.Join(", ", context.Types |> List.take (min 5 context.Types.Length) |> List.map (fun t -> t.Name))) |> ignore
                            table.AddRow("Modules", context.Modules.Length.ToString(), String.Join(", ", context.Modules |> List.take (min 3 context.Modules.Length) |> List.map (fun m -> m.Name))) |> ignore
                            table.AddRow("Dependencies", context.Dependencies.Length.ToString(), String.Join(", ", context.Dependencies |> List.take (min 3 context.Dependencies.Length))) |> ignore
                            table.AddRow("Diagnostics", context.Diagnostics.Length.ToString(), if context.Diagnostics.IsEmpty then "No issues" else "Issues found") |> ignore
                            
                            AnsiConsole.Write(table)
                            AnsiConsole.WriteLine()
                            
                            // Show quality metrics
                            let metrics = calculateQualityMetrics context
                            
                            let metricsTable = Table()
                            metricsTable.AddColumn("Quality Metric") |> ignore
                            metricsTable.AddColumn("Score") |> ignore
                            metricsTable.AddColumn("Assessment") |> ignore
                            
                            let getAssessment score =
                                if score >= 0.8 then "[green]Excellent[/]"
                                elif score >= 0.6 then "[yellow]Good[/]"
                                elif score >= 0.4 then "[orange3]Fair[/]"
                                else "[red]Poor[/]"
                            
                            metricsTable.AddRow("Maintainability", $"{metrics.Maintainability:F2}", getAssessment metrics.Maintainability) |> ignore
                            metricsTable.AddRow("Reliability", $"{metrics.Reliability:F2}", getAssessment metrics.Reliability) |> ignore
                            metricsTable.AddRow("Security", $"{metrics.Security:F2}", getAssessment metrics.Security) |> ignore
                            metricsTable.AddRow("Test Coverage", $"{metrics.TestCoverage:F2}", getAssessment metrics.TestCoverage) |> ignore
                            metricsTable.AddRow("Complexity", metrics.CyclomaticComplexity.ToString(), if metrics.CyclomaticComplexity <= 10 then "[green]Low[/]" elif metrics.CyclomaticComplexity <= 20 then "[yellow]Medium[/]" else "[red]High[/]") |> ignore
                            metricsTable.AddRow("Technical Debt", $"{metrics.TechnicalDebt:F1}", if metrics.TechnicalDebt <= 5.0 then "[green]Low[/]" elif metrics.TechnicalDebt <= 15.0 then "[yellow]Medium[/]" else "[red]High[/]") |> ignore
                            
                            AnsiConsole.MarkupLine("[bold]📊 Quality Metrics (AST-Enhanced)[/]")
                            AnsiConsole.Write(metricsTable)
                            AnsiConsole.WriteLine()
                            
                            // Show function details if available
                            if not context.Functions.IsEmpty then
                                AnsiConsole.MarkupLine("[bold]🔧 Function Analysis[/]")
                                let funcTable = Table()
                                funcTable.AddColumn("Function") |> ignore
                                funcTable.AddColumn("Parameters") |> ignore
                                funcTable.AddColumn("Line") |> ignore
                                funcTable.AddColumn("Async") |> ignore
                                funcTable.AddColumn("Public") |> ignore
                                
                                context.Functions
                                |> List.take (min 10 context.Functions.Length)
                                |> List.iter (fun func ->
                                    funcTable.AddRow(
                                        func.Name,
                                        func.Parameters.Length.ToString(),
                                        func.LineNumber.ToString(),
                                        (if func.IsAsync then "✅" else "❌"),
                                        (if func.IsPublic then "✅" else "❌")
                                    ) |> ignore
                                )
                                
                                AnsiConsole.Write(funcTable)
                                AnsiConsole.WriteLine()
                            
                            // Show diagnostics if any
                            if not context.Diagnostics.IsEmpty then
                                AnsiConsole.MarkupLine("[bold red]⚠️ Diagnostics[/]")
                                context.Diagnostics
                                |> List.take (min 5 context.Diagnostics.Length)
                                |> List.iter (fun diag ->
                                    let severity = 
                                        match diag.Severity with
                                        | FSharp.Compiler.Diagnostics.FSharpDiagnosticSeverity.Error -> "[red]ERROR[/]"
                                        | FSharp.Compiler.Diagnostics.FSharpDiagnosticSeverity.Warning -> "[yellow]WARNING[/]"
                                        | _ -> "[blue]INFO[/]"
                                    AnsiConsole.MarkupLine($"  {severity}: {diag.Message}")
                                )
                                AnsiConsole.WriteLine()
                        
                        | None ->
                            AnsiConsole.MarkupLine("[red]❌ Failed to analyze file (not F# or parse error)[/]")
                    
                    | None ->
                        // Analyze entire codebase
                        AnsiConsole.MarkupLine("[yellow]🔍 Building AST-based knowledge base...[/]")
                        let knowledgeBase = buildKnowledgeBase baseDir
                        
                        AnsiConsole.MarkupLine($"[green]✅ Analyzed {knowledgeBase.Length} files[/]")
                        AnsiConsole.WriteLine()
                        
                        // Summary statistics
                        let totalFunctions = knowledgeBase |> List.sumBy (fun ctx -> ctx.Functions.Length)
                        let totalTypes = knowledgeBase |> List.sumBy (fun ctx -> ctx.Types.Length)
                        let totalModules = knowledgeBase |> List.sumBy (fun ctx -> ctx.Modules.Length)
                        let filesWithAST = knowledgeBase |> List.filter (fun ctx -> ctx.AST.IsSome) |> List.length
                        let filesWithDiagnostics = knowledgeBase |> List.filter (fun ctx -> not ctx.Diagnostics.IsEmpty) |> List.length
                        
                        let summaryTable = Table()
                        summaryTable.AddColumn("Metric") |> ignore
                        summaryTable.AddColumn("Count") |> ignore
                        summaryTable.AddColumn("Details") |> ignore
                        
                        summaryTable.AddRow("Total Files", knowledgeBase.Length.ToString(), $"{filesWithAST} with AST") |> ignore
                        summaryTable.AddRow("Functions", totalFunctions.ToString(), "Across all files") |> ignore
                        summaryTable.AddRow("Types", totalTypes.ToString(), "Records, unions, classes, etc.") |> ignore
                        summaryTable.AddRow("Modules", totalModules.ToString(), "F# modules") |> ignore
                        summaryTable.AddRow("Files with Issues", filesWithDiagnostics.ToString(), "Compilation warnings/errors") |> ignore
                        
                        AnsiConsole.MarkupLine("[bold]📈 Codebase Summary[/]")
                        AnsiConsole.Write(summaryTable)
                        AnsiConsole.WriteLine()
                        
                        // Test semantic search
                        if options.Options.ContainsKey("search") then
                            let query = options.Options.["search"]
                            AnsiConsole.MarkupLine($"[yellow]🔍 Semantic search for: '{query}'[/]")
                            
                            let results = retrieveRelevantContexts knowledgeBase query 5
                            
                            if results.IsEmpty then
                                AnsiConsole.MarkupLine("[red]No relevant results found[/]")
                            else
                                let searchTable = Table()
                                searchTable.AddColumn("File") |> ignore
                                searchTable.AddColumn("Relevance") |> ignore
                                searchTable.AddColumn("Confidence") |> ignore
                                searchTable.AddColumn("Reasoning") |> ignore
                                
                                results
                                |> List.iter (fun result ->
                                    searchTable.AddRow(
                                        Path.GetFileName(result.Context.FilePath),
                                        $"{result.Relevance:F3}",
                                        $"{result.Confidence:F3}",
                                        result.Reasoning
                                    ) |> ignore
                                )
                                
                                AnsiConsole.Write(searchTable)
                        
                        AnsiConsole.WriteLine()
                        AnsiConsole.MarkupLine("[dim]Usage:[/]")
                        AnsiConsole.MarkupLine("[dim]  tars ast Program.fs           # Analyze specific file[/]")
                        AnsiConsole.MarkupLine("[dim]  tars ast --search \"function\" # Semantic search[/]")
                    
                    {
                        Success = true
                        Message = "AST analysis completed successfully"
                        ExitCode = 0
                    }
                with
                | ex ->
                    AnsiConsole.MarkupLine($"[red]❌ Error: {ex.Message}[/]")
                    {
                        Success = false
                        Message = $"AST analysis failed: {ex.Message}"
                        ExitCode = 1
                    }
            )
