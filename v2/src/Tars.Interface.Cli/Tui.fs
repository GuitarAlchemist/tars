module Tars.Interface.Cli.Tui

open Spectre.Console

let showSplashScreen () =
    // Skip splash when console is not interactive or explicitly disabled
    let noSplashEnv = System.Environment.GetEnvironmentVariable("TARS_NO_SPLASH")
    let shouldSkip =
        System.Console.IsOutputRedirected
        || System.Console.IsInputRedirected
        || (noSplashEnv |> System.String.IsNullOrEmpty |> not)

    if shouldSkip then
        ()
    else
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
