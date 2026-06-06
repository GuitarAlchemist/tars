namespace Tars.Interface.Cli.Commands

open Spectre.Console
open Tars.LinkedData

module LodCommand =

    let run (args: string array) =
        let subCommand = if args.Length > 0 then args.[0] else "list"
        
        match subCommand with
        | "list" ->
            let datasets = DatasetCatalog.list()
            let table = Table()
            table.Border(TableBorder.Rounded) |> ignore
            table.AddColumn("ID") |> ignore
            table.AddColumn("Name") |> ignore
            table.AddColumn("Domain") |> ignore
            table.AddColumn("Format") |> ignore
            table.AddColumn("Description") |> ignore
            
            for d in datasets do
                table.AddRow(d.Id, d.Name, d.Domain, d.Format, d.Description) |> ignore
                
            AnsiConsole.Write(table)
            0
            
        | "search" when args.Length > 1 ->
            let query = args.[1]
            let datasets = DatasetCatalog.search query
            if datasets.IsEmpty then
                printfn "No datasets found matching '%s'" query
            else
                for d in datasets do
                    printfn "- %s (%s): %s" d.Name d.Id d.Description
            0
            
        | _ ->
            printfn "Usage: tars lod <command>"
            printfn "Commands:"
            printfn "  list          List all known LOD datasets"
            printfn "  search <q>    Search datasets"
            1
