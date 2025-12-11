namespace Tars.Tools

open System
open System.IO
open System.Text.Json
open System.Collections.Concurrent
open Tars.Connectors.Mcp
open Tars.Tools.Standard

// Configuration types
type McpServerConfig =
    { Name: string
      Command: string
      Arguments: string list
      Environment: Map<string, string> }

type McpConfig = { Servers: McpServerConfig list }

/// Manages MCP server configurations and active connections
type McpManager(configPath: string) =
    let mutable config = { Servers = [] }
    let activeClients = ConcurrentDictionary<string, McpClient>()

    let loadConfig () =
        if File.Exists(configPath) then
            try
                let json = File.ReadAllText(configPath)
                config <- JsonSerializer.Deserialize<McpConfig>(json)
            with ex ->
                // Fallback or log
                printfn "Error loading mcp_config.json: %s" ex.Message
                config <- { Servers = [] }
        else
            config <- { Servers = [] }

    let saveConfig () =
        let options = JsonSerializerOptions(WriteIndented = true)
        let json = JsonSerializer.Serialize(config, options)
        File.WriteAllText(configPath, json)

    do loadConfig ()

    // --- Configuration Management ---

    member this.AddServer(name: string, command: string, args: string list, env: Map<string, string>) =
        let server =
            { Name = name
              Command = command
              Arguments = args
              Environment = env }

        let others = config.Servers |> List.filter (fun s -> s.Name <> name)
        config <- { Servers = others @ [ server ] }
        saveConfig ()

    member this.RemoveServer(name: string) =
        let others = config.Servers |> List.filter (fun s -> s.Name <> name)
        config <- { Servers = others }
        saveConfig ()

    member this.GetServers() = config.Servers

    member this.GetServer(name: string) =
        config.Servers |> List.tryFind (fun s -> s.Name = name)

    // --- Active Connection Management ---

    /// Connects to a server by name (if config exists) and returns the client
    member this.ConnectAsync(name: string) =
        task {
            match this.GetServer(name) with
            | Some cfg ->
                // Convert list to space-separated args string for StdioTransport
                // Naive joining - simple for now.
                // The current StdioTransport expects "args string".
                // Ideally, StdioTransport should take string list to avoid quoting hell.
                // For now, we join with spaces.
                let argsStr = String.Join(" ", cfg.Arguments)

                // TODO: update StdioTransport to support environment variables
                let transport = new StdioTransport(cfg.Command, argsStr, None)
                let client = new McpClient(transport)

                let! _ = client.ConnectAsync()

                activeClients.AddOrUpdate(name, client, (fun _ _ -> client)) |> ignore
                return Some client
            | None -> return None
        }

    member this.GetActiveClient(name: string) =
        match activeClients.TryGetValue(name) with
        | true, client -> Some client
        | false, _ -> None

    member this.GetAllActiveClients() =
        activeClients |> Seq.map (fun kv -> kv.Key, kv.Value) |> Seq.toList
