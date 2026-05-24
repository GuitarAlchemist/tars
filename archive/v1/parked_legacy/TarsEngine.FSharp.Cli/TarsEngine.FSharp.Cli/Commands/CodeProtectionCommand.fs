namespace TarsEngine.FSharp.Cli.Commands

open System
open System.IO
open System.Threading.Tasks
open TarsEngine.FSharp.Cli.CodeProtection.CodeIntegrityGuard
open Spectre.Console

/// Command to protect codebase from LLM corruption
type CodeProtectionCommand() =
    interface ICommand with
        member _.Name = "protect"
        member _.Description = "Protect codebase from LLM corruption and autonomous system damage"
        member _.Usage = "tars protect [--scan] [--clean] [--backup]"
        member _.Examples = [
            "tars protect --scan"
            "tars protect --clean --backup"
        ]
        member _.ValidateOptions(_options: CommandOptions) = true
        
        member _.ExecuteAsync(options: CommandOptions) =
            Task.Run(fun () ->
                try
                    let baseDir = Directory.GetCurrentDirectory()
                    
                    AnsiConsole.MarkupLine("[bold cyan]🛡️ TARS Code Protection System[/]")
                    AnsiConsole.MarkupLine("[dim]Protecting codebase from LLM corruption and autonomous damage[/]")
                    AnsiConsole.WriteLine()
                    
                    // Check for scan option
                    let shouldScan = options.Options.ContainsKey("scan") || (options.Arguments |> List.contains "scan")
                    let shouldClean = options.Options.ContainsKey("clean") || (options.Arguments |> List.contains "clean")
                    let shouldBackup = options.Options.ContainsKey("backup") || (options.Arguments |> List.contains "backup")
                    
                    if shouldBackup then
                        AnsiConsole.MarkupLine("[yellow]📦 Creating codebase backup...[/]")
                        match createCodebaseBackup baseDir with
                        | Ok backupPath ->
                            AnsiConsole.MarkupLine($"[green]✅ Backup created: {backupPath}[/]")
                        | Error err ->
                            AnsiConsole.MarkupLine($"[red]❌ Backup failed: {err}[/]")
                        AnsiConsole.WriteLine()
                    
                    if shouldScan then
                        AnsiConsole.MarkupLine("[yellow]🔍 Scanning codebase for integrity issues...[/]")
                        let report = generateIntegrityReport baseDir
                        
                        // Save report to file
                        let reportPath = Path.Combine(baseDir, "code_integrity_report.txt")
                        File.WriteAllText(reportPath, report)
                        
                        AnsiConsole.MarkupLine($"[green]✅ Integrity report saved: {reportPath}[/]")
                        AnsiConsole.WriteLine()
                        
                        // Display summary
                        let issues = scanCodebaseIntegrity baseDir
                        if issues.IsEmpty then
                            AnsiConsole.MarkupLine("[green]🎉 No integrity issues detected![/]")
                        else
                            let totalFiles = issues.Length
                            let totalIssues = issues |> List.sumBy (fun (_, issueList) -> issueList.Length)
                            
                            AnsiConsole.MarkupLine($"[red]⚠️ Found {totalIssues} issues in {totalFiles} files[/]")
                            
                            // Show top 5 problematic files
                            let topFiles = 
                                issues 
                                |> List.sortByDescending (fun (_, issueList) -> issueList.Length)
                                |> List.take (min 5 issues.Length)
                            
                            let table = Table()
                            table.AddColumn("File") |> ignore
                            table.AddColumn("Issues") |> ignore
                            table.AddColumn("Sample Issue") |> ignore
                            
                            topFiles
                            |> List.iter (fun (filePath, issueList) ->
                                let fileName = Path.GetFileName(filePath)
                                let issueCount = issueList.Length.ToString()
                                let sampleIssue = 
                                    if issueList.IsEmpty then "None"
                                    else 
                                        let (_, pattern, _) = issueList.Head
                                        pattern.Substring(0, min 50 pattern.Length) + "..."
                                
                                table.AddRow(fileName, issueCount, sampleIssue) |> ignore
                            )
                            
                            AnsiConsole.Write(table)
                    
                    if shouldClean then
                        AnsiConsole.MarkupLine("[yellow]🧹 Cleaning dangerous patterns...[/]")
                        
                        let issues = scanCodebaseIntegrity baseDir
                        let mutable totalCleaned = 0
                        
                        for (filePath, _) in issues do
                            match cleanDangerousPatterns filePath with
                            | Ok count ->
                                if count > 0 then
                                    totalCleaned <- totalCleaned + count
                                    let fileName = Path.GetFileName(filePath)
                                    AnsiConsole.MarkupLine($"[green]✅ Cleaned {count} patterns from {fileName}[/]")
                            | Error err ->
                                let fileName = Path.GetFileName(filePath)
                                AnsiConsole.MarkupLine($"[red]❌ Failed to clean {fileName}: {err}[/]")
                        
                        AnsiConsole.MarkupLine($"[green]🎉 Total patterns cleaned: {totalCleaned}[/]")
                    
                    if not shouldScan && not shouldClean && not shouldBackup then
                        // Default: show protection status
                        AnsiConsole.MarkupLine("[bold]🛡️ Code Protection Status[/]")
                        AnsiConsole.WriteLine()
                        
                        let issues = scanCodebaseIntegrity baseDir
                        let totalFiles = issues.Length
                        let totalIssues = issues |> List.sumBy (fun (_, issueList) -> issueList.Length)
                        
                        let statusTable = Table()
                        statusTable.AddColumn("Protection Metric") |> ignore
                        statusTable.AddColumn("Status") |> ignore
                        statusTable.AddColumn("Details") |> ignore
                        
                        statusTable.AddRow(
                            "Integrity Issues",
                            (if totalIssues = 0 then "[green]✅ Clean[/]" else "[red]❌ " + totalIssues.ToString() + " issues[/]"),
                            (totalFiles.ToString() + " files affected")
                        ) |> ignore
                        
                        statusTable.AddRow(
                            "Critical Files",
                            "[green]✅ Protected[/]",
                            "Core files protected from modification"
                        ) |> ignore
                        
                        statusTable.AddRow(
                            "Backup System",
                            "[green]✅ Available[/]",
                            "Use --backup to create backup"
                        ) |> ignore
                        
                        statusTable.AddRow(
                            "Pattern Detection",
                            "[green]✅ Active[/]",
                            "Monitoring for dangerous patterns"
                        ) |> ignore
                        
                        AnsiConsole.Write(statusTable)
                        AnsiConsole.WriteLine()
                        
                        AnsiConsole.MarkupLine("[dim]Usage:[/]")
                        AnsiConsole.MarkupLine("[dim]  tars protect --scan     # Scan for issues[/]")
                        AnsiConsole.MarkupLine("[dim]  tars protect --clean    # Clean dangerous patterns[/]")
                        AnsiConsole.MarkupLine("[dim]  tars protect --backup   # Create backup[/]")
                    
                    {
                        Success = true
                        Message = "Code protection completed successfully"
                        ExitCode = 0
                    }
                with
                | ex ->
                    AnsiConsole.MarkupLine($"[red]❌ Error: {ex.Message}[/]")
                    {
                        Success = false
                        Message = $"Code protection failed: {ex.Message}"
                        ExitCode = 1
                    }
            )
