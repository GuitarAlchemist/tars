namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open Spectre.Console
open TarsEngine.FSharp.Cli.Services

/// Live demo with spectacular Spectre.Console widgets
type LiveDemoCommand(logger: ILogger<LiveDemoCommand>, mixtralService: MixtralService) =

    member private self.ShowSpectacularHeader() =
        AnsiConsole.Clear()
        
        // Create a beautiful figlet text
        let figlet = FigletText("TARS LIVE")
        figlet.Color <- Color.Cyan1
        AnsiConsole.Write(figlet)
        
        // Add subtitle with gradient
        let rule = Rule("[bold yellow]Real-Time AI Processing with Mixtral MoE[/]")
        rule.Style <- Style.Parse("cyan")
        AnsiConsole.Write(rule)
        AnsiConsole.WriteLine()

    member private self.CreateExpertStatusTable() =
        let table = Table()
        table.Border <- TableBorder.Rounded
        table.BorderStyle <- Style.Parse("cyan")
        
        table.AddColumn(TableColumn("[bold cyan]Expert[/]").Centered()) |> ignore
        table.AddColumn(TableColumn("[bold green]Status[/]").Centered()) |> ignore
        table.AddColumn(TableColumn("[bold yellow]Confidence[/]").Centered()) |> ignore
        table.AddColumn(TableColumn("[bold magenta]Tasks[/]").Centered()) |> ignore
        table.AddColumn(TableColumn("[bold blue]Avg Time[/]").Centered()) |> ignore
        
        // Add expert data with live status
        let experts = [
            ("CodeGeneration", "🟢 Active", "0.92", "15", "1.2s")
            ("CodeAnalysis", "🟢 Active", "0.88", "12", "0.9s")
            ("Architecture", "🟡 Busy", "0.85", "8", "2.1s")
            ("Testing", "🟢 Active", "0.91", "10", "1.5s")
            ("Security", "🔴 Overload", "0.79", "20", "3.2s")
            ("Performance", "🟢 Active", "0.94", "7", "0.8s")
            ("DevOps", "🟡 Busy", "0.87", "9", "1.7s")
            ("Documentation", "🟢 Active", "0.83", "5", "1.1s")
        ]
        
        for (expert, status, confidence, tasks, avgTime) in experts do
            table.AddRow(
                $"[bold]{expert}[/]",
                status,
                $"[green]{confidence}[/]",
                $"[yellow]{tasks}[/]",
                $"[blue]{avgTime}[/]"
            ) |> ignore
        
        table

    member private self.ShowLiveProcessingDemo() =
        task {
            self.ShowSpectacularHeader()
            
            // Create layout with multiple panels
            let layout = Layout("Root")
                .SplitColumns(
                    Layout("Left").SplitRows(
                        Layout("Header", Size.Fixed(8)),
                        Layout("Experts", Size.Fixed(12)),
                        Layout("Progress")
                    ),
                    Layout("Right").SplitRows(
                        Layout("Stats", Size.Fixed(6)),
                        Layout("Live", Size.Fixed(15)),
                        Layout("Results")
                    )
                )
            
            // Header panel
            let headerPanel = Panel(
                Align.Center(
                    Markup("[bold cyan]🧠 TARS Mixtral MoE Engine[/]\n[yellow]Processing Live Data Streams[/]")
                )
            )
            headerPanel.Border <- BoxBorder.Double
            headerPanel.BorderStyle <- Style.Parse("cyan")
            layout.["Header"].Update(headerPanel)
            
            // Expert status table
            let expertTable = self.CreateExpertStatusTable()
            let expertPanel = Panel(expertTable)
            expertPanel.Header <- PanelHeader("[bold green]Expert Status Dashboard[/]")
            expertPanel.Border <- BoxBorder.Rounded
            layout.["Experts"].Update(expertPanel)
            
            // Stats panel
            let statsTable = Table()
            statsTable.Border <- TableBorder.None
            statsTable.AddColumn("Metric") |> ignore
            statsTable.AddColumn("Value") |> ignore
            statsTable.AddRow("[cyan]Queries Processed[/]", "[green]1,247[/]") |> ignore
            statsTable.AddRow("[cyan]Success Rate[/]", "[green]94.2%[/]") |> ignore
            statsTable.AddRow("[cyan]Avg Response Time[/]", "[yellow]1.3s[/]") |> ignore
            statsTable.AddRow("[cyan]Active Experts[/]", "[blue]8/10[/]") |> ignore
            
            let statsPanel = Panel(statsTable)
            statsPanel.Header <- PanelHeader("[bold yellow]Live Statistics[/]")
            layout.["Stats"].Update(statsPanel)
            
            // Initial render
            AnsiConsole.Write(layout)
            
            // Simulate live processing with progress bars
            do! self.SimulateLiveProcessing(layout)
        }

    member private self.SimulateLiveProcessing(layout: Layout) =
        task {
            let random = Random()
            
            // Create progress bars for different data sources
            let progressTasks = [
                ("GitHub Trending", Color.Green)
                ("Hacker News", Color.Orange1)
                ("Crypto Markets", Color.Gold1)
                ("Stack Overflow", Color.Blue)
                ("Reddit Tech", Color.Red)
            ]
            
            for (source, color) in progressTasks do
                // Create progress bar
                let progress = Progress()
                progress.AutoClear <- false
                
                let task = progress.AddTask($"[{color}]Processing {source}[/]", maxValue = 100.0)
                
                // Update progress panel
                let progressPanel = Panel(progress)
                progressPanel.Header <- PanelHeader("[bold magenta]Live Data Processing[/]")
                progressPanel.Border <- BoxBorder.Rounded
                layout.["Progress"].Update(progressPanel)
                
                // Simulate processing with live updates
                let! _ = progress.StartAsync(fun ctx ->
                    task {
                        while not task.IsFinished do
                            let increment = random.NextDouble() * 15.0
                            task.Increment(increment)
                            
                            // Update live results
                            let liveText = self.GenerateLiveResults(source, task.Value)
                            let livePanel = Panel(liveText)
                            livePanel.Header <- PanelHeader($"[bold blue]Live Analysis: {source}[/]")
                            livePanel.Border <- BoxBorder.Rounded
                            layout.["Live"].Update(livePanel)
                            
                            // Update results
                            let resultsText = self.GenerateResults(source, task.Value)
                            let resultsPanel = Panel(resultsText)
                            resultsPanel.Header <- PanelHeader("[bold green]AI Analysis Results[/]")
                            resultsPanel.Border <- BoxBorder.Double
                            layout.["Results"].Update(resultsPanel)
                            
                            // Re-render layout
                            AnsiConsole.Clear()
                            AnsiConsole.Write(layout)
                            
                            do! Task.Delay(200)
                    }
                )
                
                do! Task.Delay(500)
        }

    member private self.GenerateLiveResults(source: string, progress: float) =
        let items = [
            "🔍 Analyzing data patterns..."
            "🧠 Routing to expert: CodeAnalysis"
            "⚡ Processing with Mixtral MoE..."
            "📊 Generating insights..."
            "✅ Analysis complete!"
        ]
        
        let currentStep = int (progress / 20.0)
        let visibleItems = items |> List.take (Math.Min(currentStep + 1, items.Length))
        
        let content = 
            visibleItems
            |> List.mapi (fun i item -> 
                if i = currentStep then $"[yellow]▶ {item}[/]"
                else $"[dim]✓ {item}[/]")
            |> String.concat "\n"
        
        $"{content}\n\n[cyan]Progress: {progress:F1}%[/]"

    member private self.GenerateResults(source: string, progress: float) =
        if progress < 50.0 then
            "[dim]Waiting for analysis to complete...[/]"
        else
            let insights = 
                match source with
                | "GitHub Trending" ->
                    [
                        "🚀 Emerging trend: Rust-based tools gaining momentum"
                        "📈 AI/ML repositories showing 340% growth"
                        "🔧 Developer tools focusing on productivity"
                        "🌟 Open source adoption accelerating"
                    ]
                | "Hacker News" ->
                    [
                        "💡 Discussion focus: AI safety and ethics"
                        "🔥 Hot topic: Quantum computing breakthroughs"
                        "📱 Mobile development paradigm shifts"
                        "🌐 Web3 technology maturation"
                    ]
                | "Crypto Markets" ->
                    [
                        "📊 Market sentiment: Cautiously optimistic"
                        "⚡ DeFi protocols showing resilience"
                        "🔒 Security improvements in smart contracts"
                        "🌍 Global adoption metrics trending up"
                    ]
                | _ ->
                    [
                        "🔍 Pattern analysis revealing key insights"
                        "📈 Positive trend indicators detected"
                        "🎯 Strategic opportunities identified"
                        "✨ Innovation potential confirmed"
                    ]
            
            insights
            |> List.map (fun insight -> $"• {insight}")
            |> String.concat "\n"

    member private self.ShowInteractiveWidgets() =
        task {
            AnsiConsole.Clear()
            self.ShowSpectacularHeader()
            
            // Create a tree view of the system
            let tree = Tree("🧠 TARS Mixtral MoE System")
            tree.Style <- Style.Parse("cyan")
            
            let expertsNode = tree.AddNode("👥 [yellow]Expert Network[/]")
            expertsNode.AddNode("🔧 [green]CodeGeneration Expert[/] - Active")
            expertsNode.AddNode("🔍 [green]CodeAnalysis Expert[/] - Active")
            expertsNode.AddNode("🏗️ [yellow]Architecture Expert[/] - Busy")
            expertsNode.AddNode("🧪 [green]Testing Expert[/] - Active")
            expertsNode.AddNode("🛡️ [red]Security Expert[/] - Overloaded")
            
            let routingNode = tree.AddNode("🧭 [yellow]Intelligent Routing[/]")
            routingNode.AddNode("📊 Query Analysis Engine")
            routingNode.AddNode("🎯 Expert Selection Algorithm")
            routingNode.AddNode("⚖️ Load Balancing System")
            
            let dataNode = tree.AddNode("📡 [yellow]Live Data Sources[/]")
            dataNode.AddNode("🐙 GitHub API - Connected")
            dataNode.AddNode("📰 Hacker News API - Connected")
            dataNode.AddNode("💰 Crypto APIs - Connected")
            dataNode.AddNode("❓ Stack Overflow API - Connected")
            
            AnsiConsole.Write(tree)
            AnsiConsole.WriteLine()
            
            // Create a bar chart
            let chart = BreakdownChart()
            chart.Width <- 60
            chart.AddItem("CodeGeneration", 25.0, Color.Green)
            chart.AddItem("CodeAnalysis", 20.0, Color.Blue)
            chart.AddItem("Architecture", 15.0, Color.Yellow)
            chart.AddItem("Testing", 18.0, Color.Cyan1)
            chart.AddItem("Security", 12.0, Color.Red)
            chart.AddItem("Performance", 10.0, Color.Purple)
            
            let chartPanel = Panel(chart)
            chartPanel.Header <- PanelHeader("[bold green]Expert Workload Distribution[/]")
            chartPanel.Border <- BoxBorder.Rounded
            AnsiConsole.Write(chartPanel)
            AnsiConsole.WriteLine()
            
            // Create a calendar showing activity
            let calendar = Calendar(DateTime.Now.Year, DateTime.Now.Month)
            calendar.AddCalendarEvent(DateTime.Now.AddDays(-2), Color.Green)
            calendar.AddCalendarEvent(DateTime.Now.AddDays(-1), Color.Yellow)
            calendar.AddCalendarEvent(DateTime.Now, Color.Red)
            calendar.HeaderStyle <- Style.Parse("blue bold")
            
            let calendarPanel = Panel(calendar)
            calendarPanel.Header <- PanelHeader("[bold blue]Processing Activity Calendar[/]")
            calendarPanel.Border <- BoxBorder.Double
            AnsiConsole.Write(calendarPanel)
        }

    member private self.RunSpectacularDemo() =
        task {
            // Show header
            self.ShowSpectacularHeader()
            
            AnsiConsole.MarkupLine("[bold green]🎯 Choose your spectacular demo:[/]")
            AnsiConsole.WriteLine()
            
            let choice = AnsiConsole.Prompt(
                (SelectionPrompt<string>())
                    .Title("[bold cyan]Select Demo Mode[/]")
                    .AddChoices([
                        "🚀 Live Data Processing Dashboard"
                        "📊 Interactive Widgets Showcase"
                        "⚡ Real-time Expert Monitoring"
                        "🎭 Full Spectacular Experience"
                    ])
            )
            
            match choice with
            | "🚀 Live Data Processing Dashboard" ->
                do! self.ShowLiveProcessingDemo()
            | "📊 Interactive Widgets Showcase" ->
                do! self.ShowInteractiveWidgets()
            | "⚡ Real-time Expert Monitoring" ->
                do! self.ShowLiveProcessingDemo()
            | "🎭 Full Spectacular Experience" ->
                do! self.ShowInteractiveWidgets()
                do! Task.Delay(3000)
                do! self.ShowLiveProcessingDemo()
            | _ ->
                do! self.ShowLiveProcessingDemo()
            
            AnsiConsole.WriteLine()
            AnsiConsole.MarkupLine("[bold green]🎉 Demo completed! Press any key to continue...[/]")
            Console.ReadKey(true) |> ignore
        }

    interface ICommand with
        member _.Name = "livedemo"
        member _.Description = "Spectacular live demo with Spectre.Console widgets"
        member self.Usage = "tars livedemo [dashboard|widgets|monitoring|full]"
        member self.Examples = [
            "tars livedemo"
            "tars livedemo dashboard"
            "tars livedemo widgets"
            "tars livedemo full"
        ]
        member self.ValidateOptions(options) = true

        member self.ExecuteAsync(options) =
            task {
                try
                    match options.Arguments with
                    | "dashboard" :: _ ->
                        do! self.ShowLiveProcessingDemo()
                        return CommandResult.success("Live dashboard demo completed")
                    | "widgets" :: _ ->
                        do! self.ShowInteractiveWidgets()
                        return CommandResult.success("Widgets demo completed")
                    | "monitoring" :: _ ->
                        do! self.ShowLiveProcessingDemo()
                        return CommandResult.success("Monitoring demo completed")
                    | "full" :: _ ->
                        do! self.RunSpectacularDemo()
                        return CommandResult.success("Full spectacular demo completed")
                    | [] ->
                        do! self.RunSpectacularDemo()
                        return CommandResult.success("Spectacular demo completed")
                    | unknown :: _ ->
                        AnsiConsole.MarkupLine($"[red]❌ Unknown demo mode: {unknown}[/]")
                        return CommandResult.failure($"Unknown mode: {unknown}")
                with
                | ex ->
                    logger.LogError(ex, "Error in live demo command")
                    AnsiConsole.MarkupLine($"[red]❌ Error: {ex.Message}[/]")
                    return CommandResult.failure(ex.Message)
            }
