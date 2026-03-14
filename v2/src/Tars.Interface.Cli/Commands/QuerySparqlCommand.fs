namespace Tars.Interface.Cli.Commands

open System
open System.IO
open Spectre.Console
open Tars.Core
open Tars.LinkedData
open Tars.Interface.Cli

module QuerySparqlCommand = 

    type QueryOptions = {
        Endpoint: string
        Query: string option
        Preset: string option
        File: string option
    }

    let defaultOptions = {
        Endpoint = QueryTemplates.Endpoints.Wikidata
        Query = None
        Preset = None
        File = None
    }

    let run (args: string array) = 
        async {
            Logging.init LoggingConfig.Development
            let log = Logging.withCategory "CLI"
            
            // Parse args
            let mutable options = defaultOptions
            let mutable i = 0
            
            while i < args.Length do
                match args.[i] with
                | "--endpoint" when i + 1 < args.Length ->
                    options <- { options with Endpoint = args.[i+1] }
                    i <- i + 2
                | "--preset" when i + 1 < args.Length ->
                    options <- { options with Preset = Some args.[i+1] }
                    i <- i + 2
                | "--file" when i + 1 < args.Length ->
                    options <- { options with File = Some args.[i+1] }
                    i <- i + 2
                | arg when not (arg.StartsWith("--")) && options.Query.IsNone ->
                    options <- { options with Query = Some arg }
                    i <- i + 1
                | _ -> i <- i + 1

            // Determine query
            let queryStr = 
                match options.Preset with
                | Some "languages" -> Some QueryTemplates.Queries.getProgrammingLanguages
                | Some "dbpedia-languages" -> 
                    options <- { options with Endpoint = QueryTemplates.Endpoints.DBpedia }
                    Some QueryTemplates.Queries.getDbpediaLanguages
                | Some p -> 
                    log.Warn (sprintf "Unknown preset: %s" p)
                    None
                | None ->
                    match options.File with
                    | Some f -> 
                        if File.Exists f then Some (File.ReadAllText f)
                        else 
                            log.Error(sprintf "Query file not found: %s" f, null)
                            None
                    | None -> options.Query

            match queryStr with
            | None ->
                AnsiConsole.MarkupLine("[red]Error: No query provided.[/]")
                printfn "Usage: tars query-sparql \"SELECT ...\" [--endpoint <url>] [--preset languages]"
                return 1
            | Some q ->
                AnsiConsole.MarkupLine(sprintf "[blue]Querying %s...[/]" options.Endpoint)
                AnsiConsole.MarkupLine(sprintf "[dim]%s[/]" (q.Trim()))
                
                try
                    let endpointUri = Uri(options.Endpoint)
                    let! result = SparqlClient.query endpointUri q
                    
                    match result with
                    | Microsoft.FSharp.Core.Result.Error err -> 
                        log.Error(sprintf "Query failed: %s" err)
                        return 1
                    | Microsoft.FSharp.Core.Result.Ok results ->
                        if results.IsEmpty then
                            AnsiConsole.MarkupLine("[yellow]No results found.[/]")
                        else
                            AnsiConsole.MarkupLine(sprintf "[green]Found %d results[/]" results.Length)
                            
                            // Build table
                            let table = Table()
                            table.Border(TableBorder.Rounded) |> ignore
                            
                            // Add columns from first result keys
                            let headers = results.Head.Keys |> Seq.toList
                            for h in headers do table.AddColumn(h) |> ignore
                            
                            // Add rows
                            for row in results do
                                let rowValues = headers |> List.map (fun h -> 
                                    if row.ContainsKey h then row.[h] else "")
                                table.AddRow(List.toArray rowValues) |> ignore
                                
                            AnsiConsole.Write(table)
                        return 0
                with ex ->
                    log.Error(sprintf "Execution error: %s" ex.Message, ex)
                    return 1
        } |> Async.RunSynchronously
