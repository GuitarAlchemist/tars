namespace Tars.Interface.Cli.Commands

open System
open System.IO
open System.Text.Json
open System.Threading.Tasks
open Spectre.Console
open Tars.Connectors.Mcp

/// Built-in skill catalog with popular MCP servers
module SkillCatalog =

    type SkillDefinition =
        { Name: string
          Description: string
          Command: string
          Arguments: string list
          Source: string }

    let private skills =
        [ { Name = "github"
            Description = "GitHub API access - repos, issues, PRs, code search"
            Command = "npx"
            Arguments = [ "-y"; "@modelcontextprotocol/server-github" ]
            Source = "modelcontextprotocol" }

          { Name = "filesystem"
            Description = "Local filesystem access - read, write, list files"
            Command = "npx"
            Arguments = [ "-y"; "@modelcontextprotocol/server-filesystem"; "." ]
            Source = "modelcontextprotocol" }

          { Name = "fetch"
            Description = "HTTP fetch - retrieve web content"
            Command = "npx"
            Arguments = [ "-y"; "@modelcontextprotocol/server-fetch" ]
            Source = "modelcontextprotocol" }

          { Name = "git"
            Description = "Git operations - status, diff, log, commit"
            Command = "npx"
            Arguments = [ "-y"; "@modelcontextprotocol/server-git" ]
            Source = "modelcontextprotocol" }

          { Name = "postgres"
            Description = "PostgreSQL database access"
            Command = "npx"
            Arguments = [ "-y"; "@modelcontextprotocol/server-postgres" ]
            Source = "modelcontextprotocol" }

          { Name = "sqlite"
            Description = "SQLite database access"
            Command = "npx"
            Arguments = [ "-y"; "@modelcontextprotocol/server-sqlite" ]
            Source = "modelcontextprotocol" }

          { Name = "brave-search"
            Description = "Brave Search API for web search"
            Command = "npx"
            Arguments = [ "-y"; "@modelcontextprotocol/server-brave-search" ]
            Source = "modelcontextprotocol" }

          { Name = "puppeteer"
            Description = "Browser automation with Puppeteer"
            Command = "npx"
            Arguments = [ "-y"; "@modelcontextprotocol/server-puppeteer" ]
            Source = "modelcontextprotocol" } ]

    let getAll () = skills

    let tryFind (name: string) =
        skills
        |> List.tryFind (fun s -> s.Name.Equals(name, StringComparison.OrdinalIgnoreCase))

/// MCP configuration file management
module McpConfig =

    type ServerEntry =
        { Name: string
          Command: string
          Arguments: string list
          Environment: Map<string, string> }

    type McpConfigFile = { Servers: ServerEntry list }

    let private configPath =
        Path.Combine(Environment.CurrentDirectory, "mcp_config.json")

    let private jsonOptions =
        let opts = JsonSerializerOptions(WriteIndented = true)
        opts.PropertyNamingPolicy <- JsonNamingPolicy.CamelCase
        opts

    let load () =
        if File.Exists(configPath) then
            try
                let json = File.ReadAllText(configPath)
                let doc = JsonDocument.Parse(json)
                let serversElem = doc.RootElement.GetProperty("Servers")

                let servers =
                    [ for elem in serversElem.EnumerateArray() do
                          let name = elem.GetProperty("Name").GetString()
                          let command = elem.GetProperty("Command").GetString()

                          let args =
                              [ for arg in elem.GetProperty("Arguments").EnumerateArray() do
                                    yield arg.GetString() ]

                          let env =
                              try
                                  let envElem = elem.GetProperty("Environment")

                                  [ for prop in envElem.EnumerateObject() do
                                        yield prop.Name, prop.Value.GetString() ]
                                  |> Map.ofList
                              with _ ->
                                  Map.empty

                          yield
                              { Name = name
                                Command = command
                                Arguments = args
                                Environment = env } ]

                { Servers = servers }
            with _ ->
                { Servers = [] }
        else
            { Servers = [] }

    let save (config: McpConfigFile) =
        let serverEntries =
            config.Servers
            |> List.map (fun s ->
                let argsJson =
                    s.Arguments |> List.map (fun a -> "\"" + a + "\"") |> String.concat ", "

                "    {\n"
                + "      \"Name\": \""
                + s.Name
                + "\",\n"
                + "      \"Command\": \""
                + s.Command
                + "\",\n"
                + "      \"Arguments\": ["
                + argsJson
                + "],\n"
                + "      \"Environment\": {}\n"
                + "    }")
            |> String.concat ",\n"

        let json = "{\n  \"Servers\": [\n" + serverEntries + "\n  ]\n}"
        File.WriteAllText(configPath, json)

    let addServer (server: ServerEntry) =
        let config = load ()

        let filtered =
            config.Servers
            |> List.filter (fun s -> not (s.Name.Equals(server.Name, StringComparison.OrdinalIgnoreCase)))

        save { Servers = filtered @ [ server ] }

    let removeServer (name: string) =
        let config = load ()

        let filtered =
            config.Servers
            |> List.filter (fun s -> not (s.Name.Equals(name, StringComparison.OrdinalIgnoreCase)))

        if filtered.Length < config.Servers.Length then
            save { Servers = filtered }
            true
        else
            false

    let hasServer (name: string) =
        let config = load ()

        config.Servers
        |> List.exists (fun s -> s.Name.Equals(name, StringComparison.OrdinalIgnoreCase))

/// CLI commands for skill management
module SkillCommand =

    let listInstalled () =
        task {
            let config = McpConfig.load ()

            if config.Servers.IsEmpty then
                AnsiConsole.MarkupLine("[yellow]No skills installed.[/]")
                AnsiConsole.MarkupLine("")
                AnsiConsole.MarkupLine("Run [bold]tars skill catalog[/] to see available skills.")
            else
                let table = Table()
                table.AddColumn("Skill") |> ignore
                table.AddColumn("Command") |> ignore
                table.AddColumn("Status") |> ignore
                table.Border(TableBorder.Rounded) |> ignore

                for server in config.Servers do
                    let args = server.Arguments |> String.concat " "

                    table.AddRow($"[bold cyan]{server.Name}[/]", $"{server.Command} {args}", "[green]Installed[/]")
                    |> ignore

                AnsiConsole.MarkupLine("[bold]Installed Skills[/]")
                AnsiConsole.WriteLine()
                AnsiConsole.Write(table)

            return 0
        }

    let showCatalog () =
        task {
            let skills = SkillCatalog.getAll ()
            let installed = McpConfig.load ()

            let table = Table()
            table.AddColumn("Skill") |> ignore
            table.AddColumn("Description") |> ignore
            table.AddColumn("Status") |> ignore
            table.Border(TableBorder.Rounded) |> ignore

            for skill in skills do
                let isInstalled =
                    installed.Servers
                    |> List.exists (fun s -> s.Name.Equals(skill.Name, StringComparison.OrdinalIgnoreCase))

                let status =
                    if isInstalled then
                        "[green]Installed[/]"
                    else
                        "[dim]Available[/]"

                table.AddRow($"[bold cyan]{skill.Name}[/]", skill.Description, status) |> ignore

            AnsiConsole.MarkupLine("[bold]Available MCP Skills[/] (from agentskills.io / MCP)")
            AnsiConsole.WriteLine()
            AnsiConsole.Write(table)
            AnsiConsole.WriteLine()
            AnsiConsole.MarkupLine("Install with: [bold]tars skill install <name>[/]")

            return 0
        }

    let install (name: string) =
        task {
            match SkillCatalog.tryFind name with
            | None ->
                AnsiConsole.MarkupLine($"[red]Skill '{name}' not found in catalog.[/]")
                AnsiConsole.MarkupLine("")
                AnsiConsole.MarkupLine("Run [bold]tars skill catalog[/] to see available skills.")
                return 1

            | Some skill ->
                if McpConfig.hasServer skill.Name then
                    AnsiConsole.MarkupLine($"[yellow]Skill '{skill.Name}' is already installed.[/]")
                    return 0
                else
                    AnsiConsole.MarkupLine($"[bold blue]Installing skill:[/] {skill.Name}")
                    let argsStr = skill.Arguments |> String.concat " "
                    AnsiConsole.MarkupLine($"[dim]Command: {skill.Command} {argsStr}[/]")

                    let server: McpConfig.ServerEntry =
                        { Name = skill.Name
                          Command = skill.Command
                          Arguments = skill.Arguments
                          Environment = Map.empty }

                    McpConfig.addServer server

                    AnsiConsole.MarkupLine($"[green]✓ Skill '{skill.Name}' installed successfully![/]")
                    AnsiConsole.WriteLine()
                    AnsiConsole.MarkupLine($"[dim]The skill is now available as an MCP server.[/]")
                    AnsiConsole.MarkupLine($"[dim]Use 'tars mcp' commands to interact with it.[/]")

                    return 0
        }

    let remove (name: string) =
        task {
            if McpConfig.removeServer name then
                AnsiConsole.MarkupLine($"[green]✓ Skill '{name}' removed successfully![/]")
                return 0
            else
                AnsiConsole.MarkupLine($"[yellow]Skill '{name}' was not installed.[/]")
                return 1
        }

    let run (subcommand: string) (args: string list) =
        task {
            match subcommand.ToLowerInvariant() with
            | "list"
            | "ls" -> return! listInstalled ()
            | "catalog"
            | "available" -> return! showCatalog ()
            | "install"
            | "add" ->
                match args with
                | name :: _ -> return! install name
                | [] ->
                    AnsiConsole.MarkupLine("[red]Usage: tars skill install <name>[/]")
                    return 1
            | "remove"
            | "uninstall"
            | "rm" ->
                match args with
                | name :: _ -> return! remove name
                | [] ->
                    AnsiConsole.MarkupLine("[red]Usage: tars skill remove <name>[/]")
                    return 1
            | ""
            | "help" ->
                AnsiConsole.MarkupLine("[bold]TARS Skill Management[/]")
                AnsiConsole.WriteLine()
                AnsiConsole.MarkupLine("  [bold]tars skill list[/]           - List installed skills")
                AnsiConsole.MarkupLine("  [bold]tars skill catalog[/]        - Show available skills")
                AnsiConsole.MarkupLine("  [bold]tars skill install <name>[/] - Install a skill")
                AnsiConsole.MarkupLine("  [bold]tars skill remove <name>[/]  - Remove a skill")
                return 0
            | other ->
                AnsiConsole.MarkupLine($"[red]Unknown subcommand: {other}[/]")
                AnsiConsole.MarkupLine("Run [bold]tars skill help[/] for usage.")
                return 1
        }
