namespace TarsEngine.FSharp.DynamicUI

open System
open Fable.Core
open Fable.React
open Fable.React.Props
open Browser.Dom
open Types

module AgentTeams =
    
    /// Initialize the dynamic UI agent teams
    let createAgentTeams () = [
        {
            Id = "requirements-analyzer"
            Name = "Requirements Analyzer"
            Role = "Continuously analyzes DSL requirements and user needs"
            Expertise = ["DSL parsing"; "User behavior analysis"; "Requirement prioritization"]
            CycleIntervalMs = 2000
            IsActive = true
            LastExecution = None
            ComponentsGenerated = 0
            CurrentTask = Some "Analyzing current requirements"
        }
        {
            Id = "3d-scene-architect"
            Name = "3D Scene Architect"
            Role = "Builds and evolves Three.js WebGPU scenes"
            Expertise = ["Three.js"; "WebGPU"; "3D mathematics"; "Scene optimization"]
            CycleIntervalMs = 3000
            IsActive = true
            LastExecution = None
            ComponentsGenerated = 0
            CurrentTask = Some "Designing agent tree structure"
        }
        {
            Id = "component-factory"
            Name = "Component Factory"
            Role = "Generates React/Elmish components dynamically"
            Expertise = ["React"; "Elmish"; "Component architecture"; "State management"]
            CycleIntervalMs = 2500
            IsActive = true
            LastExecution = None
            ComponentsGenerated = 0
            CurrentTask = Some "Creating agent status displays"
        }
        {
            Id = "ux-optimizer"
            Name = "UX Optimizer"
            Role = "Continuously improves user experience and interactions"
            Expertise = ["User experience"; "Accessibility"; "Interaction design"; "Usability"]
            CycleIntervalMs = 4000
            IsActive = true
            LastExecution = None
            ComponentsGenerated = 0
            CurrentTask = Some "Optimizing navigation patterns"
        }
        {
            Id = "performance-monitor"
            Name = "Performance Monitor"
            Role = "Monitors and optimizes rendering performance"
            Expertise = ["WebGPU optimization"; "Memory management"; "FPS optimization"; "Profiling"]
            CycleIntervalMs = 1500
            IsActive = true
            LastExecution = None
            ComponentsGenerated = 0
            CurrentTask = Some "Monitoring WebGPU performance"
        }
        {
            Id = "style-generator"
            Name = "Style Generator"
            Role = "Generates and evolves CSS styles and themes"
            Expertise = ["CSS"; "Theming"; "Visual design"; "Animation"]
            CycleIntervalMs = 3500
            IsActive = true
            LastExecution = None
            ComponentsGenerated = 0
            CurrentTask = Some "Creating Interstellar theme"
        }
    ]
    
    /// Execute an agent team cycle
    let executeAgentCycle (agent: AgentTeam) (currentState: obj) : AgentTeam * DynamicComponent list * AgentMessage list =
        let updatedAgent = { agent with LastExecution = Some DateTime.Now }
        
        match agent.Id with
        | "requirements-analyzer" ->
            // Analyze requirements and generate requests
            let newRequirements = {
                AgentTreeVisible = true
                ShowConnections = true
                VisualizationStyle = Interstellar
                InteractionMode = Orbit
                PerformanceMode = HighQuality
                UpdateFrequency = 60
            }
            
            let component = {
                Id = $"req-analysis-{DateTime.Now.Ticks}"
                Type = "RequirementsPanel"
                GeneratedBy = agent.Name
                CreatedAt = DateTime.Now
                Properties = Map ["requirements", box newRequirements]
                ReactElement = Some (
                    div [ 
                        Style [ 
                            Position "absolute"
                            Top "20px"
                            Left "20px"
                            Background "rgba(0, 0, 0, 0.9)"
                            Color "#00ff88"
                            Padding "15px"
                            BorderRadius "10px"
                            Border "1px solid #00ff88"
                            FontFamily "monospace"
                            FontSize "12px"
                        ] 
                    ] [
                        h4 [] [ str "📋 Requirements Analysis" ]
                        div [] [ str $"Agent Tree: {newRequirements.AgentTreeVisible}" ]
                        div [] [ str $"Style: {newRequirements.VisualizationStyle}" ]
                        div [] [ str $"Mode: {newRequirements.InteractionMode}" ]
                        div [] [ str $"Updated: {DateTime.Now.ToString("HH:mm:ss")}" ]
                    ]
                )
                ThreeJSObject = None
                IsVisible = true
            }
            
            let messages = [ RequirementsChanged newRequirements ]
            ({ updatedAgent with ComponentsGenerated = agent.ComponentsGenerated + 1 }, [component], messages)
            
        | "3d-scene-architect" ->
            // Generate 3D scene components
            let component = {
                Id = $"3d-scene-{DateTime.Now.Ticks}"
                Type = "AgentNode3D"
                GeneratedBy = agent.Name
                CreatedAt = DateTime.Now
                Properties = Map [
                    "position", box (Random().NextDouble() * 10.0 - 5.0, Random().NextDouble() * 10.0 - 5.0, Random().NextDouble() * 10.0 - 5.0)
                    "color", box "#00ff88"
                    "size", box 0.3
                ]
                ReactElement = None
                ThreeJSObject = Some (obj()) // Would be actual Three.js object
                IsVisible = true
            }
            
            ({ updatedAgent with ComponentsGenerated = agent.ComponentsGenerated + 1 }, [component], [])
            
        | "component-factory" ->
            // Generate React/Elmish components
            let agentCount = Random().Next(5, 15)
            let component = {
                Id = $"agent-status-{DateTime.Now.Ticks}"
                Type = "AgentStatusPanel"
                GeneratedBy = agent.Name
                CreatedAt = DateTime.Now
                Properties = Map ["agentCount", box agentCount]
                ReactElement = Some (
                    div [ 
                        Style [ 
                            Position "absolute"
                            Top "20px"
                            Right "20px"
                            Background "rgba(0, 0, 0, 0.9)"
                            Color "#00ff88"
                            Padding "15px"
                            BorderRadius "10px"
                            Border "1px solid #00ff88"
                            FontFamily "monospace"
                            FontSize "12px"
                            Width "250px"
                        ] 
                    ] [
                        h4 [] [ str "🤖 Agent Status" ]
                        div [] [ str $"Active Agents: {agentCount}" ]
                        div [] [ str $"Generated: {DateTime.Now.ToString("HH:mm:ss")}" ]
                        div [ Style [ MarginTop "10px" ] ] [
                            for i in 1..agentCount ->
                                div [ Style [ Margin "3px 0"; Padding "3px"; BorderLeft "2px solid #00ff88"; PaddingLeft "8px" ] ] [
                                    str $"Agent-{i}: Running"
                                ]
                        ]
                    ]
                )
                ThreeJSObject = None
                IsVisible = true
            }
            
            ({ updatedAgent with ComponentsGenerated = agent.ComponentsGenerated + 1 }, [component], [])
            
        | "performance-monitor" ->
            // Monitor and report performance
            let metrics = {
                FPS = 60.0 + Random().NextDouble() * 10.0 - 5.0
                ComponentCount = Random().Next(10, 50)
                ActiveAgents = Random().Next(5, 15)
                MemoryUsage = Random().NextDouble() * 100.0
                RenderTime = Random().NextDouble() * 16.0
                WebGPUActive = true
            }
            
            let component = {
                Id = $"perf-monitor-{DateTime.Now.Ticks}"
                Type = "PerformancePanel"
                GeneratedBy = agent.Name
                CreatedAt = DateTime.Now
                Properties = Map ["metrics", box metrics]
                ReactElement = Some (
                    div [ 
                        Style [ 
                            Position "absolute"
                            Bottom "20px"
                            Right "20px"
                            Background "rgba(0, 0, 0, 0.9)"
                            Color "#00ff88"
                            Padding "15px"
                            BorderRadius "10px"
                            Border "1px solid #00ff88"
                            FontFamily "monospace"
                            FontSize "12px"
                        ] 
                    ] [
                        h4 [] [ str "⚡ Performance" ]
                        div [] [ str $"FPS: {metrics.FPS:F1}" ]
                        div [] [ str $"Components: {metrics.ComponentCount}" ]
                        div [] [ str $"Memory: {metrics.MemoryUsage:F1}MB" ]
                        div [] [ str $"WebGPU: {if metrics.WebGPUActive then "Active" else "Inactive"}" ]
                    ]
                )
                ThreeJSObject = None
                IsVisible = true
            }
            
            let messages = [ PerformanceUpdate metrics ]
            ({ updatedAgent with ComponentsGenerated = agent.ComponentsGenerated + 1 }, [component], messages)
            
        | _ ->
            // Default agent behavior
            ({ updatedAgent with ComponentsGenerated = agent.ComponentsGenerated + 1 }, [], [])
    
    /// Get agent team status display
    let getAgentTeamStatusComponent (teams: AgentTeam list) =
        div [ 
            Style [ 
                Position "absolute"
                Top "20px"
                Left "20px"
                Background "rgba(0, 0, 0, 0.9)"
                Color "#00ff88"
                Padding "20px"
                BorderRadius "10px"
                Border "1px solid #00ff88"
                FontFamily "monospace"
                FontSize "12px"
                MaxWidth "400px"
            ] 
        ] [
            h3 [] [ str "🏗️ Agent Teams Building Interface" ]
            for team in teams ->
                div [ 
                    Style [ 
                        Margin "10px 0"
                        Padding "10px"
                        BorderLeft "3px solid #00ff88"
                        PaddingLeft "12px"
                        Background "rgba(0, 255, 136, 0.1)"
                        BorderRadius "5px"
                    ] 
                ] [
                    strong [] [ str team.Name ]
                    br []
                    small [] [ str team.Role ]
                    br []
                    small [] [ str $"Generated: {team.ComponentsGenerated} components" ]
                    br []
                    small [] [ str $"Last: {team.LastExecution |> Option.map (fun d -> d.ToString("HH:mm:ss")) |> Option.defaultValue "Never"}" ]
                    match team.CurrentTask with
                    | Some task -> 
                        br []
                        small [ Style [ Color "#ffaa00" ] ] [ str $"Task: {task}" ]
                    | None -> ()
                ]
        ]
