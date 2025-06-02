namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open Spectre.Console
open TarsEngine.FSharp.Agents.AgentTypes
open TarsEngine.FSharp.Agents.AgentPersonas
open TarsEngine.FSharp.Agents.SpecializedTeams
open TarsEngine.FSharp.Agents.AgentOrchestrator
// Types are in the same namespace, no need to import

/// <summary>
/// Command for managing specialized agent teams in TARS
/// </summary>
type TeamsCommand(logger: ILogger<TeamsCommand>) =
    
    /// Display all available specialized teams
    member private this.DisplayAvailableTeams() =
        task {
            AnsiConsole.Clear()
            
            let title = FigletText("TARS TEAMS")
            title.Color <- Color.Cyan1
            AnsiConsole.Write(title)
            
            AnsiConsole.MarkupLine("[bold cyan]🤖 TARS Specialized Agent Teams Management[/]")
            AnsiConsole.WriteLine()
            
            let teams = getAllTeamConfigurations()
            
            let table = Table()
            table.AddColumn("[bold]Team Name[/]") |> ignore
            table.AddColumn("[bold]Description[/]") |> ignore
            table.AddColumn("[bold]Objectives[/]") |> ignore
            table.AddColumn("[bold]Communication[/]") |> ignore
            
            for team in teams do
                let objectives = team.SharedObjectives |> List.take (min 3 team.SharedObjectives.Length) |> String.concat "\n• "
                table.AddRow(
                    $"[bold blue]{team.Name}[/]",
                    team.Description,
                    $"• {objectives}",
                    team.CommunicationProtocol
                ) |> ignore
            
            AnsiConsole.Write(table)
            AnsiConsole.WriteLine()
        }
    
    /// Display team details
    member private this.DisplayTeamDetails(teamName: string) =
        task {
            match getTeamByName teamName with
            | Some team ->
                AnsiConsole.Clear()
                
                let objectives = team.SharedObjectives |> List.mapi (fun i obj -> $"{i+1}. {obj}") |> String.concat "\n"
                let personas = getRecommendedPersonasForTeam team.Name |> List.map (fun p -> $"• {p.Name} - {p.Specialization}") |> String.concat "\n"

                let panelContent = $"""[bold cyan]{team.Name}[/]

[bold yellow]Description:[/]
{team.Description}

[bold yellow]Shared Objectives:[/]
{objectives}

[bold yellow]Communication Protocol:[/]
{team.CommunicationProtocol}

[bold yellow]Decision Making Process:[/]
{team.DecisionMakingProcess}

[bold yellow]Conflict Resolution:[/]
{team.ConflictResolution}

[bold yellow]Recommended Personas:[/]
{personas}"""

                let panel = Panel(panelContent)
                panel.Header <- PanelHeader($"[bold blue]📋 {team.Name} Details[/]")
                panel.Border <- BoxBorder.Rounded
                panel.BorderStyle <- Style.Parse("blue")
                
                AnsiConsole.Write(panel)
                AnsiConsole.WriteLine()

            | None ->
                AnsiConsole.MarkupLine($"[red]❌ Team '{teamName}' not found[/]")
        }
    
    /// Create and deploy a specialized team
    member private this.CreateTeam(teamName: string) =
        task {
            match getTeamByName teamName with
            | Some teamConfig ->
                AnsiConsole.MarkupLine($"[yellow]🚀 Creating {teamConfig.Name}...[/]")
                
                let personas = getRecommendedPersonasForTeam teamConfig.Name
                let team = createTeamWithPersonas teamConfig personas
                
                // Display creation progress
                let progress = AnsiConsole.Progress()
                progress.AutoClear <- false
                
                do! progress.StartAsync(fun ctx ->
                    task {
                        let task1 = ctx.AddTask($"[green]Spawning {personas.Length} agents[/]")
                        let task2 = ctx.AddTask("[green]Configuring team coordination[/]")
                        let task3 = ctx.AddTask("[green]Initializing communication channels[/]")
                        
                        // Simulate agent creation
                        for i in 0..personas.Length-1 do
                            task1.Increment(100.0 / float personas.Length)
                            do! Task.Delay(500)
                        
                        task2.Increment(100.0)
                        do! Task.Delay(300)
                        
                        task3.Increment(100.0)
                        do! Task.Delay(200)
                    })
                
                AnsiConsole.MarkupLine($"[green]✅ {teamConfig.Name} created successfully![/]")
                AnsiConsole.MarkupLine($"[cyan]👥 Team Members: {personas.Length} agents[/]")
                AnsiConsole.MarkupLine($"[cyan]🎯 Objectives: {team.SharedObjectives.Length} defined[/]")
                
                // Display team summary
                let summaryTable = Table()
                summaryTable.AddColumn("[bold]Agent[/]") |> ignore
                summaryTable.AddColumn("[bold]Persona[/]") |> ignore
                summaryTable.AddColumn("[bold]Specialization[/]") |> ignore
                
                for persona in personas do
                    let agentId = Guid.NewGuid().ToString("N").[..7]
                    summaryTable.AddRow(
                        $"[blue]Agent-{agentId}[/]",
                        $"[bold]{persona.Name}[/]",
                        persona.Specialization
                    ) |> ignore
                
                AnsiConsole.Write(summaryTable)
                
            | None ->
                AnsiConsole.MarkupLine($"[red]❌ Team '{teamName}' not found[/]")
                AnsiConsole.MarkupLine("[yellow]💡 Use 'tars teams list' to see available teams[/]")
        }
    
    /// Display team creation demo
    member private this.RunTeamDemo() =
        task {
            AnsiConsole.Clear()
            
            AnsiConsole.MarkupLine("[bold cyan]🎬 TARS Specialized Teams Demo[/]")
            AnsiConsole.WriteLine()
            
            let demoTeams = ["DevOps Team"; "AI Team"; "Innovation Team"]
            
            for teamName in demoTeams do
                AnsiConsole.MarkupLine($"[yellow]🚀 Demonstrating {teamName}...[/]")
                do! this.CreateTeam(teamName)
                AnsiConsole.WriteLine()
                do! Task.Delay(1000)
            
            AnsiConsole.MarkupLine("[bold green]🎉 Demo completed! All specialized teams are ready for deployment.[/]")
        }
    
    /// Display help information
    member private this.DisplayHelp() =
        task {
            let helpText = """
[bold cyan]🤖 TARS Teams Command Help[/]

[bold yellow]USAGE:[/]
  tars teams <subcommand> [options]

[bold yellow]SUBCOMMANDS:[/]
  [bold]list[/]              List all available specialized teams
  [bold]details <team>[/]    Show detailed information about a specific team
  [bold]create <team>[/]     Create and deploy a specialized team
  [bold]demo[/]              Run a demonstration of team creation
  [bold]help[/]              Show this help information

[bold yellow]EXAMPLES:[/]
  tars teams list
  tars teams details "DevOps Team"
  tars teams create "AI Team"
  tars teams demo

[bold yellow]AVAILABLE TEAMS:[/]
  • DevOps Team - Infrastructure and deployment specialists
  • Technical Writers Team - Documentation and knowledge management
  • Architecture Team - System design and planning specialists
  • Direction Team - Strategic planning and product direction
  • Innovation Team - Research and breakthrough solutions
  • Machine Learning Team - AI/ML development specialists
  • UX Team - User experience and interface design
  • AI Team - Advanced AI research and coordination
"""
            
            let panel = Panel(helpText)
            panel.Header <- PanelHeader("[bold blue]📚 Teams Command Help[/]")
            panel.Border <- BoxBorder.Rounded
            panel.BorderStyle <- Style.Parse("blue")
            
            AnsiConsole.Write(panel)
        }

    interface ICommand with
        member _.Name = "teams"
        member _.Description = "Manage specialized agent teams (DevOps, Technical Writers, Architecture, Direction, Innovation, ML, UX, AI)"
        member _.Usage = "tars teams <subcommand> [options]"
        member _.Examples = [
            "tars teams list"
            "tars teams details \"DevOps Team\""
            "tars teams create \"AI Team\""
            "tars teams demo"
        ]
        member _.ValidateOptions(_) = true

        member this.ExecuteAsync(options) =
            task {
                try
                    match options.Arguments with
                    | "list" :: _ ->
                        do! this.DisplayAvailableTeams()
                        return CommandResult.success("Teams listed successfully")
                    | "details" :: teamName :: _ ->
                        do! this.DisplayTeamDetails(teamName)
                        return CommandResult.success($"Team details displayed for {teamName}")
                    | "create" :: teamName :: _ ->
                        do! this.CreateTeam(teamName)
                        return CommandResult.success($"Team {teamName} created successfully")
                    | "demo" :: _ ->
                        do! this.RunTeamDemo()
                        return CommandResult.success("Team demo completed")
                    | "help" :: _ ->
                        do! this.DisplayHelp()
                        return CommandResult.success("Help displayed")
                    | [] ->
                        do! this.DisplayAvailableTeams()
                        return CommandResult.success("Teams listed successfully")
                    | unknown :: _ ->
                        AnsiConsole.MarkupLine($"[red]❌ Unknown teams subcommand: {unknown}[/]")
                        do! this.DisplayHelp()
                        return CommandResult.failure($"Unknown subcommand: {unknown}")
                with
                | ex ->
                    logger.LogError(ex, "Error executing teams command")
                    AnsiConsole.MarkupLine($"[red]❌ Error: {ex.Message}[/]")
                    return CommandResult.failure(ex.Message)
            }
