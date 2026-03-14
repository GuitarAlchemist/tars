open System
open Spectre.Console

let displaySwarmStatus() =
    // Create a beautiful table
    let table = Table()
    table.AddColumn("[bold cyan]Container[/]") |> ignore
    table.AddColumn("[bold green]Status[/]") |> ignore
    table.AddColumn("[bold yellow]Uptime[/]") |> ignore
    table.AddColumn("[bold magenta]Ports[/]") |> ignore
    table.AddColumn("[bold blue]Role[/]") |> ignore

    // Add sample data
    table.AddRow("tars-alpha", "[green]🟢 Running[/]", "2h 15m", "8080-8081", "[bold]🎯 Primary[/]") |> ignore
    table.AddRow("tars-beta", "[green]🟢 Running[/]", "2h 14m", "8082-8083", "[bold]🔄 Secondary[/]") |> ignore
    table.AddRow("tars-gamma", "[green]🟢 Running[/]", "2h 13m", "8084-8085", "[bold]🧪 Experimental[/]") |> ignore
    table.AddRow("tars-delta", "[green]🟢 Running[/]", "2h 12m", "8086-8087", "[bold]🔍 QA[/]") |> ignore
    table.AddRow("tars-postgres", "[green]🟢 Running[/]", "2h 16m", "5432", "[bold]🗄️ Database[/]") |> ignore
    table.AddRow("tars-redis", "[green]🟢 Running[/]", "2h 16m", "6379", "[bold]⚡ Cache[/]") |> ignore

    AnsiConsole.Write(table)

let runSwarmTests() =
    AnsiConsole.MarkupLine("[bold cyan]🧪 Running TARS Swarm Tests...[/]")

    let table = Table()
    table.AddColumn("[bold]Test[/]") |> ignore
    table.AddColumn("[bold]Result[/]") |> ignore

    table.AddRow("Container Health Check", "[green]✅ PASS[/]") |> ignore
    table.AddRow("Network Connectivity", "[green]✅ PASS[/]") |> ignore
    table.AddRow("TARS CLI Availability", "[green]✅ PASS[/]") |> ignore
    table.AddRow("Metascript Execution", "[green]✅ PASS[/]") |> ignore
    table.AddRow("Inter-Container Communication", "[red]❌ FAIL[/]") |> ignore
    table.AddRow("Load Balancing", "[green]✅ PASS[/]") |> ignore

    AnsiConsole.Write(table)

let showDemoHeader() =
    AnsiConsole.Clear()

    // Beautiful header
    let rule = Rule("[bold cyan]🚀 TARS Autonomous Swarm Demo[/]")
    AnsiConsole.Write(rule)
    AnsiConsole.WriteLine()

    // Show ASCII art
    AnsiConsole.MarkupLine("""
[bold cyan]
    ████████╗ █████╗ ██████╗ ███████╗
    ╚══██╔══╝██╔══██╗██╔══██╗██╔════╝
       ██║   ███████║██████╔╝███████╗
       ██║   ██╔══██║██╔══██╗╚════██║
       ██║   ██║  ██║██║  ██║███████║
       ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝
[/]
[bold yellow]The Autonomous Reasoning System - Swarm Mode[/]
""")

let runPerformanceMonitor() =
    AnsiConsole.MarkupLine("[bold cyan]📈 TARS Swarm Performance Monitor[/]")
    let random = Random()

    for i in 1..5 do
        let cpuUsage = random.Next(10, 80)
        let memoryUsage = random.Next(20, 90)
        let networkIO = random.Next(5, 50)
        let diskIO = random.Next(5, 40)

        AnsiConsole.MarkupLine($"[bold]Iteration {i}/5[/]")
        AnsiConsole.MarkupLine($"CPU: [red]{cpuUsage}%%[/] | Memory: [blue]{memoryUsage}%%[/] | Network: [green]{networkIO}%%[/] | Disk: [yellow]{diskIO}%%[/]")

        // Create a simple progress bar effect
        let progressBar = String.replicate (cpuUsage / 10) "█"
        AnsiConsole.MarkupLine($"CPU Load: [red]{progressBar}[/]")

        System.Threading.Thread.Sleep(1500)
        AnsiConsole.WriteLine()

let runContainerCommands() =
    AnsiConsole.MarkupLine("[bold cyan]📋 Executing Commands Across TARS Swarm...[/]")

    let commands = [
        ("tars version", "TARS CLI v1.0.0+412c685b")
        ("docker ps", "6 containers running")
        ("tars metascript list", "Found 12 metascripts")
        ("tars agent status", "4 agents active")
    ]

    for (command, result) in commands do
        AnsiConsole.MarkupLine($"[bold blue]📡 Executing:[/] [cyan]{command}[/]")
        System.Threading.Thread.Sleep(800)
        AnsiConsole.MarkupLine($"[green]✅ Result: {result}[/]")
        AnsiConsole.WriteLine()

let runInteractiveDemo() =
    let mutable continueLoop = true

    while continueLoop do
        showDemoHeader()

        let choice = AnsiConsole.Prompt(
            (SelectionPrompt<string>())
                .Title("[bold green]🎯 What would you like to explore?[/]")
                .AddChoices([
                    "📊 View Swarm Status"
                    "🧪 Run Swarm Tests"
                    "📈 Performance Monitor"
                    "📋 Execute Commands"
                    "🔄 Restart Containers (Simulated)"
                    "🚪 Exit Demo"
                ])
        )

        AnsiConsole.WriteLine()

        match choice with
        | "📊 View Swarm Status" ->
            AnsiConsole.MarkupLine("[bold cyan]🔍 Checking TARS Swarm Status...[/]")
            displaySwarmStatus()

        | "🧪 Run Swarm Tests" ->
            runSwarmTests()

        | "📈 Performance Monitor" ->
            runPerformanceMonitor()

        | "📋 Execute Commands" ->
            runContainerCommands()

        | "🔄 Restart Containers (Simulated)" ->
            AnsiConsole.MarkupLine("[bold yellow]🔄 Restarting TARS Swarm Containers...[/]")
            AnsiConsole.MarkupLine("[green]Stopping containers...[/]")
            System.Threading.Thread.Sleep(1000)
            AnsiConsole.MarkupLine("[blue]Starting containers...[/]")
            System.Threading.Thread.Sleep(1000)
            AnsiConsole.MarkupLine("[bold green]✅ Containers restarted successfully![/]")

        | "🚪 Exit Demo" ->
            AnsiConsole.MarkupLine("[bold green]👋 Thanks for using TARS Swarm Demo![/]")
            continueLoop <- false

        | _ -> ()

        if continueLoop then
            AnsiConsole.WriteLine()
            AnsiConsole.MarkupLine("[dim]Press any key to continue...[/]")
            Console.ReadKey(true) |> ignore

let runSimpleDemo() =
    showDemoHeader()
    AnsiConsole.MarkupLine("[bold green]🔍 Checking TARS Swarm Status...[/]")
    displaySwarmStatus()
    AnsiConsole.WriteLine()
    AnsiConsole.MarkupLine("[bold cyan]🧪 Running Tests...[/]")
    runSwarmTests()
    AnsiConsole.WriteLine()
    AnsiConsole.MarkupLine("[bold green]✅ Demo completed successfully![/]")

[<EntryPoint>]
let main args =
    try
        match args with
        | [| "interactive" |] | [| "-i" |] ->
            runInteractiveDemo()
        | [| "status" |] ->
            showDemoHeader()
            displaySwarmStatus()
        | [| "test" |] ->
            showDemoHeader()
            runSwarmTests()
        | [| "monitor" |] ->
            showDemoHeader()
            runPerformanceMonitor()
        | _ ->
            runSimpleDemo()
        0
    with
    | ex ->
        AnsiConsole.MarkupLine($"[red]❌ Error: {ex.Message}[/]")
        1
