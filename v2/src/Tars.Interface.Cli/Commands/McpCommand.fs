namespace Tars.Interface.Cli.Commands

open Spectre.Console
open Tars.Connectors.Mcp

module McpCommand =

    /// Render the configured MCP client backends (from mcp_config.json in CWD).
    /// Shared by `tars mcp list` and the cross-repo demo.
    let renderConfigured () =
        let servers = Tars.Tools.McpTools.Manager.GetServers()
        if List.isEmpty servers then
            AnsiConsole.MarkupLine("[yellow]No MCP servers configured[/] — no mcp_config.json in the current directory.")
        else
            let table = new Table()
            table.AddColumn("Name") |> ignore
            table.AddColumn("Command") |> ignore
            table.Border(TableBorder.Rounded) |> ignore
            for s in servers do
                let cmd = s.Command + " " + System.String.Join(" ", s.Arguments)
                table.AddRow(Markup.Escape s.Name, Markup.Escape cmd) |> ignore
            AnsiConsole.Write(table)
        servers.Length

    /// `tars mcp list` — show configured MCP federation backends.
    let list () =
        task {
            AnsiConsole.MarkupLine("[bold]Configured MCP backends[/] (federated by [bold]tars mcp server[/]):")
            renderConfigured () |> ignore
            return 0
        }

    let run (command: string) (args: string) =
        task {
            AnsiConsole.MarkupLine($"[bold blue]Connecting to MCP Server:[/] {command} {args}")

            let transport = new StdioTransport(command, args, None)
            let client = new McpClient(transport)

            try
                AnsiConsole.MarkupLine("Initializing...")
                let! initResult = client.ConnectAsync()

                AnsiConsole.MarkupLine(
                    $"[green]Connected![/] Server: [bold]{initResult.ServerInfo.Name} v{initResult.ServerInfo.Version}[/]"
                )

                AnsiConsole.MarkupLine("Listing tools...")
                let! toolsResult = client.ListToolsAsync()

                let table = new Table()
                table.AddColumn("Name") |> ignore
                table.AddColumn("Description") |> ignore
                table.Border(TableBorder.Rounded) |> ignore

                for tool in toolsResult.Tools do
                    // Escape: tool names/descriptions may contain '[...]' that Spectre would parse as markup.
                    let name = Markup.Escape tool.Name
                    let desc = Markup.Escape(tool.Description |> Option.defaultValue "")
                    table.AddRow(name, desc) |> ignore

                AnsiConsole.Write(table)

                do! client.CloseAsync()
                return 0
            with ex ->
                AnsiConsole.MarkupLine($"[red]Error:[/] {ex.Message}")
                return 1
        }
