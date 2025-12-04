namespace Tars.Connectors.Mcp

open System
open System.Text.Json
open Tars.Core

module McpToolAdapter =

    let toTarsTool (client: McpClient) (mcpTool: McpTool) : Tool =
        let execute (input: string) : Async<Result<string, string>> =
            async {
                try
                    // Parse input JSON to Map<string, obj>
                    let args =
                        try
                            let doc = JsonDocument.Parse(input)

                            if doc.RootElement.ValueKind = JsonValueKind.Object then
                                doc.RootElement.EnumerateObject()
                                |> Seq.map (fun prop ->
                                    let value =
                                        match prop.Value.ValueKind with
                                        | JsonValueKind.String -> prop.Value.GetString() :> obj
                                        | JsonValueKind.Number ->
                                            if prop.Value.ToString().Contains(".") then
                                                prop.Value.GetDouble() :> obj
                                            else
                                                prop.Value.GetInt64() :> obj
                                        | JsonValueKind.True -> true :> obj
                                        | JsonValueKind.False -> false :> obj
                                        | _ -> prop.Value.GetRawText() :> obj // Fallback

                                    prop.Name, value)
                                |> Map.ofSeq
                            else
                                Map.empty
                        with _ ->
                            Map.empty

                    let! result = client.CallToolAsync(mcpTool.Name, args) |> Async.AwaitTask

                    if result.IsError = Some true then
                        let errorText =
                            result.Content
                            |> List.map (fun c -> c.Text |> Option.defaultValue "")
                            |> String.concat "\n"

                        return Result.Error errorText
                    else
                        let successText =
                            result.Content
                            |> List.map (fun c -> c.Text |> Option.defaultValue "")
                            |> String.concat "\n"

                        return Result.Ok successText
                with ex ->
                    return Result.Error ex.Message
            }

        { Name = mcpTool.Name
          Description = mcpTool.Description |> Option.defaultValue ""
          Version = "1.0.0"
          ParentVersion = None
          CreatedAt = DateTime.UtcNow
          Execute = execute }
