namespace Tars.Interface.Cli.Commands

open Spectre.Console
open Tars.Connectors.Mcp

module McpCommand =

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
                    table.AddRow(tool.Name, tool.Description |> Option.defaultValue "") |> ignore

                AnsiConsole.Write(table)

                do! client.CloseAsync()
                return 0
            with ex ->
                AnsiConsole.MarkupLine($"[red]Error:[/] {ex.Message}")
                return 1
        }
