namespace Tars.Tools

open System
open System.Threading.Tasks
open System.Text.Json
open Tars.Core

module McpTools =

    // We need a way to access the singleton McpManager.
    // WORKAROUND: We will initialize a static instance for tools to use,
    // pointing to the standard config location.

    let private configPath =
        System.IO.Path.Combine(System.Environment.CurrentDirectory, "mcp_config.json")

    let Manager = McpManager(configPath)

    [<TarsToolAttribute("list_mcp_servers", "Lists all configured MCP servers. No input required.")>]
    let listMcpServers (_: string) : Task<string> =
        task {
            let servers = Manager.GetServers()

            if servers.IsEmpty then
                return "No MCP servers configured."
            else
                let lines =
                    servers
                    |> List.mapi (fun i s ->
                        sprintf "%d. %s (Command: %s %s)" (i + 1) s.Name s.Command (String.Join(" ", s.Arguments)))

                return "Configured MCP Servers:\n" + (String.Join("\n", lines))
        }

    [<TarsToolAttribute("configure_mcp_server",
                        "Adds a new MCP server configuration. Input JSON: { \"name\": \"...\", \"command\": \"...\", \"args\": \"...\" }")>]
    let configureMcpServer (args: string) : Task<string> =
        task {
            try
                let name = ToolHelpers.parseStringArg args "name"
                let command = ToolHelpers.parseStringArg args "command"
                let argsStr = ToolHelpers.parseStringArg args "args"

                if String.IsNullOrWhiteSpace(name) || String.IsNullOrWhiteSpace(command) then
                    return "Error: 'name' and 'command' are required."
                else
                    // Naive split of args string by space (imperfect but fits current simple needs)
                    let argList =
                        if String.IsNullOrWhiteSpace(argsStr) then
                            []
                        else
                            argsStr.Split(' ', System.StringSplitOptions.RemoveEmptyEntries) |> Array.toList

                    // Explicitly ignore synchronous unit result
                    // do returns unit, which is compatible with statement in task CE (if not binding)
                    Manager.AddServer(name, command, argList, Map.empty)

                    return
                        $"✅ MCP Server '%s{name}' configured successfully.\nCommand: %s{command} %A{argList}\n\nRestart TARS or use /reload to activate."
            with ex ->
                return $"Error configuring server: %s{ex.Message}"
        }

    [<TarsToolAttribute("install_mcp_server",
                        "Installs an MCP server via NPM. Input JSON: { \"package\": \"@modelcontextprotocol/server-...\" }")>]
    let installMcpServer (args: string) : Task<string> =
        task {
            try
                let pkg = ToolHelpers.parseStringArg args "package"

                if String.IsNullOrWhiteSpace(pkg) then
                    return "Error: 'package' is required."
                else
                    // Use run_command tool logic (or similar) to execute npm install -g
                    // We'll trust 'npm' is in the PATH since TARS environment assumes dev tools.

                    let psi = System.Diagnostics.ProcessStartInfo("npm", $"install -g {pkg}")
                    psi.RedirectStandardOutput <- true
                    psi.RedirectStandardError <- true
                    psi.UseShellExecute <- false

                    let p = System.Diagnostics.Process.Start(psi)

                    // Fallback to sync wait to avoid Task/unit type inference issues in CE
                    p.WaitForExit()

                    if p.ExitCode = 0 then
                        return $"✅ Successfully installed '%s{pkg}' globally."
                    else
                        let! err = p.StandardError.ReadToEndAsync()
                        return $"❌ Failed to install '%s{pkg}'. Exit code: %d{p.ExitCode}. Error: %s{err}"
            with ex ->
                return $"Error installing package: %s{ex.Message}"
        }

    [<TarsToolAttribute("search_mcp_servers", "Searches for available MCP servers via NPM. Input: query string.")>]
    let searchMcpServers (args: string) : Task<string> =
        task {
            try
                let query = (ToolHelpers.parseStringArg args "query").ToLower()
                printfn $"🔍 SEARCHING MCP SERVERS: {query}"

                // Try to search @modelcontextprotocol/server- namespace
                let psi =
                    System.Diagnostics.ProcessStartInfo("npm", $"search @modelcontextprotocol/server-{query} --json")

                psi.RedirectStandardOutput <- true
                psi.RedirectStandardError <- true
                psi.UseShellExecute <- false
                psi.CreateNoWindow <- true

                use p = System.Diagnostics.Process.Start(psi)
                let! stdout = p.StandardOutput.ReadToEndAsync()
                p.WaitForExit()

                if p.ExitCode = 0 && not (String.IsNullOrWhiteSpace stdout) then
                    try
                        let doc = JsonDocument.Parse(stdout)
                        let results = doc.RootElement.EnumerateArray() |> Seq.toList

                        if results.IsEmpty then
                            return $"No MCP servers found for query '{query}'."
                        else
                            let lines =
                                results
                                |> List.truncate 10
                                |> List.map (fun r ->
                                    let name = r.GetProperty("name").GetString()
                                    let desc = r.GetProperty("description").GetString()
                                    $"- {name}: {desc}")

                            return $"Found MCP Servers on NPM:\n" + (String.Join("\n", lines))
                    with _ ->
                        return $"Found MCP Servers (parsed): {stdout.Substring(0, min 200 stdout.Length)}..."
                else
                    // Fallback to a better curated list if npm search fails or returns nothing
                    let curated =
                        [ "filesystem (@modelcontextprotocol/server-filesystem) - Access local files"
                          "github (@modelcontextprotocol/server-github) - Access GitHub API"
                          "postgres (@modelcontextprotocol/server-postgres) - Database access"
                          "memory (@modelcontextprotocol/server-memory) - Ephemeral memory graph"
                          "google-maps (@modelcontextprotocol/server-google-maps) - Geolocation"
                          "slack (@modelcontextprotocol/server-slack) - Slack integration" ]

                    let matches = curated |> List.filter (fun s -> s.ToLower().Contains(query))

                    if matches.IsEmpty then
                        return "No servers found. Try 'filesystem', 'github', or 'postgres'."
                    else
                        return "Matched Curated MCP Servers:\n" + (String.Join("\n", matches))
            with ex ->
                return $"Error searching MCP servers: {ex.Message}"
        }

    /// Helper to get all tools defined in this module
    let getTools () =
        let createTool name desc (exec: string -> Task<string>) =
            { Name = name
              Description = desc
              Version = "1.0.0"
              ParentVersion = None
              CreatedAt = System.DateTime.UtcNow
              Execute =
                fun input ->
                    async {
                        try
                            let! result = exec input |> Async.AwaitTask
                            return Result.Ok result
                        with ex ->
                            return Result.Error ex.Message
                    } }

        let t1 =
            createTool "list_mcp_servers" "Lists all configured MCP servers. No input required." listMcpServers

        let t2 =
            createTool
                "configure_mcp_server"
                "Adds a new MCP server configuration. Input JSON: { \"name\": \"...\", \"command\": \"...\", \"args\": \"...\" }"
                configureMcpServer

        let t3 =
            createTool
                "install_mcp_server"
                "Installs an MCP server via NPM. Input JSON: { \"package\": \"@modelcontextprotocol/server-...\" }"
                installMcpServer

        let t4 =
            createTool
                "search_mcp_servers"
                "Searches for available MCP servers (simulated). Input: query string."
                searchMcpServers

        [ t1; t2; t3; t4 ]
