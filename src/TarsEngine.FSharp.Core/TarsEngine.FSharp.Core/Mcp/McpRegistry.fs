namespace TarsEngine.FSharp.Core.Mcp

open System
open System.Collections.Generic
open System.IO
open YamlDotNet.Serialization
open YamlDotNet.Serialization.NamingConventions

type McpServer =
    { Name: string
      Url: string
      Description: string option
      Headers: Map<string, string> }

[<CLIMutable>]
type McpServerYaml =
    { name: string
      url: string
      description: string
      headers: IDictionary<string, string> }

[<CLIMutable>]
type McpConfigYaml =
    { servers: IList<McpServerYaml> }

module McpRegistry =

    let private someIfNotBlank (value: string) =
        if String.IsNullOrWhiteSpace value then None else Some value

    let private configCandidates () =
        [ someIfNotBlank (Environment.GetEnvironmentVariable("TARS_MCP_CONFIG"))
          Some(Path.Combine(Environment.CurrentDirectory, "config", "mcp-servers.yaml"))
          Some(Path.Combine(AppContext.BaseDirectory, "config", "mcp-servers.yaml"))
          Some(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".tars", "mcp-servers.yaml")) ]
        |> List.choose id
        |> List.distinct

    let private deserializer =
        DeserializerBuilder()
            .WithNamingConvention(CamelCaseNamingConvention.Instance)
            .IgnoreUnmatchedProperties()
            .Build()

    let private load () =
        let rec loop paths =
            match paths with
            | [] -> Map.empty, None
            | path :: rest ->
                if File.Exists path then
                    try
                        use reader = new StringReader(File.ReadAllText(path))
                        let config = deserializer.Deserialize<McpConfigYaml>(reader)
                        let rawServers =
                            if obj.ReferenceEquals(config, null) || obj.ReferenceEquals(config.servers, null) then
                                Seq.empty
                            else
                                config.servers :> seq<McpServerYaml>

                        let servers =
                            rawServers
                            |> Seq.choose (fun server ->
                                if obj.ReferenceEquals(server, null) then
                                    None
                                elif String.IsNullOrWhiteSpace server.name || String.IsNullOrWhiteSpace server.url then
                                    None
                                else
                                    let headers =
                                        if obj.ReferenceEquals(server.headers, null) then
                                            Map.empty
                                        else
                                            server.headers
                                            |> Seq.map (fun kv -> (kv.Key, kv.Value))
                                            |> Seq.choose (fun (key, value) ->
                                                let trimmedKey = if isNull key then null else key.Trim()
                                                if String.IsNullOrWhiteSpace trimmedKey then None
                                                else Some (trimmedKey, value))
                                            |> Map.ofSeq

                                    let description =
                                        if String.IsNullOrWhiteSpace server.description then None
                                        else Some (server.description.Trim())

                                    Some (
                                        server.name.Trim().ToLowerInvariant(),
                                        { Name = server.name.Trim()
                                          Url = server.url.Trim()
                                          Description = description
                                          Headers = headers }))
                            |> Map.ofSeq

                        servers, Some path
                    with ex ->
                        raise (InvalidOperationException($"Failed to parse MCP configuration '{path}': {ex.Message}", ex))
                else
                    loop rest

        loop (configCandidates ())

    let mutable private cache : Lazy<Map<string, McpServer> * string option> = lazy (load ())

    let private ensure () = cache.Value

    let TryGetServer (name: string) =
        if String.IsNullOrWhiteSpace name then
            None
        else
            let servers, _ = ensure ()
            servers |> Map.tryFind (name.Trim().ToLowerInvariant())

    let GetServers () =
        ensure () |> fst

    let KnownServerNames () =
        GetServers ()
        |> Map.toSeq
        |> Seq.map (fun (_, server) -> server.Name)
        |> Seq.toList

    let ConfigPath () =
        ensure () |> snd

    let ConfigSearchPaths () = configCandidates ()

    let Refresh () =
        cache <- lazy (load ())
