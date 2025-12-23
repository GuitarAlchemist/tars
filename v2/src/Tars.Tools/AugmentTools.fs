namespace Tars.Tools.Augment

open System
open System.Threading.Tasks
open System.Text.Json
open Tars.Tools
open Tars.Connectors

module AugmentTools =

    // Singleton client for connection reuse
    let mutable private clientOpt: AugmentClient option = None
    let private clientLock = obj ()

    let private getOrCreateClient (workspacePath: string) =
        lock clientLock (fun () ->
            match clientOpt with
            | Some c -> c
            | None ->
                let c = new AugmentClient(workspacePath)
                clientOpt <- Some c
                c)

    let private parseCodebaseSearchArgs (args: string) =
        try
            let doc = JsonDocument.Parse(args)
            let root = doc.RootElement

            let mutable queryProp = Unchecked.defaultof<JsonElement>
            let mutable workspaceProp = Unchecked.defaultof<JsonElement>

            let query =
                if root.TryGetProperty("query", &queryProp) then
                    queryProp.GetString()
                elif root.TryGetProperty("information_request", &queryProp) then
                    queryProp.GetString()
                else
                    args.Trim('"')

            let workspace =
                if root.TryGetProperty("workspace", &workspaceProp) then
                    workspaceProp.GetString()
                else
                    Environment.CurrentDirectory

            Ok(query, workspace)
        with ex ->
            // If JSON parsing fails, treat entire args as query
            Ok(args.Trim('"'), Environment.CurrentDirectory)

    [<TarsToolAttribute("augment_codebase_search",
                        "Semantic search over the codebase using Augment Context Engine. Input JSON: { \"query\": \"Where is the BudgetGovernor implemented?\" }")>]
    let codebaseSearch (args: string) =
        task {
            match parseCodebaseSearchArgs args with
            | Error msg -> return $"augment_codebase_search error: {msg}"
            | Ok(query, workspace) ->
                try
                    let client = getOrCreateClient workspace

                    // Connect if not already connected
                    try
                        let! _ = client.ConnectAsync()
                        ()
                    with :? InvalidOperationException ->
                        // Already connected, continue
                        ()

                    let! result = client.CodebaseRetrievalAsync(query)

                    if String.IsNullOrWhiteSpace(result) then
                        return "No results found for the query."
                    else
                        return result
                with ex ->
                    return $"augment_codebase_search error: {ex.Message}"
        }

    [<TarsToolAttribute("augment_disconnect", "Disconnect from Augment Context Engine MCP server.")>]
    let disconnect (_: string) =
        task {
            lock clientLock (fun () ->
                match clientOpt with
                | Some c ->
                    c.CloseAsync() |> Async.AwaitTask |> Async.RunSynchronously
                    clientOpt <- None
                | None -> ())

            return "Disconnected from Augment Context Engine."
        }
