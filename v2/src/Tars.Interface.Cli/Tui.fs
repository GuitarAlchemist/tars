module Tars.Interface.Cli.Tui

open Spectre.Console

let showSplashScreen () =
    AnsiConsole.Clear()

    let font = FigletFont.Default

    AnsiConsole.Write(FigletText(font, "TARS v2").Centered().Color(Color.Cyan1))

    let panel =
        Panel("[bold white]The Automated Reasoning System[/]\n[dim]v2.0-alpha[/]")
            .Border(BoxBorder.Rounded)
            .BorderStyle(Style(Color.Blue))
            .Padding(2, 1, 2, 1)
            .Expand()

    AnsiConsole.Write(panel)

    AnsiConsole.MarkupLine("[grey]Initializing kernel...[/]")
    // Simulate some loading for effect
    // System.Threading.Thread.Sleep(500)
    AnsiConsole.MarkupLine("[green]Ready.[/]")
    AnsiConsole.WriteLine()
