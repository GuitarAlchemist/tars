namespace TarsEngine.FSharp.DynamicUI

open System
open Types
open AgentTeams

module State =
    
    /// Main application state
    type Model = {
        AgentTeams: AgentTeam list
        DynamicComponents: DynamicComponent list
        TarsAgents: TarsAgent list
        SceneRequirements: SceneRequirements
        PerformanceMetrics: PerformanceMetrics
        IsGenerating: bool
        GenerationLog: string list
        LastUpdate: DateTime
    }
    
    /// Application messages
    type Msg =
        | StartDynamicGeneration
        | StopDynamicGeneration
        | AgentCycleCompleted of string * DynamicComponent list * AgentMessage list
        | UpdateRequirements of SceneRequirements
        | UpdatePerformance of PerformanceMetrics
        | AddLogEntry of string
        | ClearOldComponents
        | ToggleAgent of string
    
    /// Initialize the application state
    let init () =
        let initialAgents = [
            { Id = "core-1"; Name = "Core Engine"; Type = "core"; Position = (0.0, 0.0, 0.0); Status = Running; TaskCount = 15; Department = "Core" }
            { Id = "ui-1"; Name = "UI Dev Team"; Type = "ui"; Position = (-3.0, 2.0, 0.0); Status = Running; TaskCount = 8; Department = "UI" }
            { Id = "ui-2"; Name = "3D Graphics Team"; Type = "ui"; Position = (-2.0, 3.0, 1.0); Status = Running; TaskCount = 5; Department = "UI" }
            { Id = "qa-1"; Name = "QA Agent Alpha"; Type = "qa"; Position = (3.0, 1.0, 0.0); Status = Idle; TaskCount = 0; Department = "QA" }
            { Id = "meta-1"; Name = "Metascript Engine"; Type = "meta"; Position = (0.0, -2.0, 2.0); Status = Running; TaskCount = 7; Department = "Core" }
        ]
        
        let initialRequirements = {
            AgentTreeVisible = true
            ShowConnections = true
            VisualizationStyle = Interstellar
            InteractionMode = Orbit
            PerformanceMode = HighQuality
            UpdateFrequency = 60
        }
        
        let initialMetrics = {
            FPS = 60.0
            ComponentCount = 0
            ActiveAgents = 0
            MemoryUsage = 0.0
            RenderTime = 0.0
            WebGPUActive = false
        }
        
        {
            AgentTeams = createAgentTeams ()
            DynamicComponents = []
            TarsAgents = initialAgents
            SceneRequirements = initialRequirements
            PerformanceMetrics = initialMetrics
            IsGenerating = false
            GenerationLog = ["🚀 TARS Dynamic UI System Initialized"]
            LastUpdate = DateTime.Now
        }, Cmd.none
    
    /// Add a log entry with timestamp
    let addLogEntry (message: string) (model: Model) =
        let timestamp = DateTime.Now.ToString("HH:mm:ss")
        let entry = $"[{timestamp}] {message}"
        let newLog = entry :: model.GenerationLog |> List.take 50 // Keep only last 50 entries
        { model with GenerationLog = newLog; LastUpdate = DateTime.Now }
    
    /// Update agent team after cycle completion
    let updateAgentTeam (teamId: string) (newComponents: DynamicComponent list) (model: Model) =
        let updatedTeams = 
            model.AgentTeams 
            |> List.map (fun team -> 
                if team.Id = teamId then 
                    { team with 
                        LastExecution = Some DateTime.Now
                        ComponentsGenerated = team.ComponentsGenerated + newComponents.Length }
                else team)
        
        let updatedComponents = 
            // Remove old components of the same type to prevent accumulation
            let filteredComponents = 
                model.DynamicComponents 
                |> List.filter (fun comp -> 
                    not (newComponents |> List.exists (fun newComp -> newComp.Type = comp.Type)))
            
            filteredComponents @ newComponents
        
        { model with 
            AgentTeams = updatedTeams
            DynamicComponents = updatedComponents }
    
    /// Process agent messages
    let processAgentMessages (messages: AgentMessage list) (model: Model) =
        messages |> List.fold (fun acc message ->
            match message with
            | RequirementsChanged newReqs ->
                addLogEntry $"Requirements updated: {newReqs.VisualizationStyle}" 
                    { acc with SceneRequirements = newReqs }
            | PerformanceUpdate metrics ->
                addLogEntry $"Performance: {metrics.FPS:F1} FPS, {metrics.ComponentCount} components"
                    { acc with PerformanceMetrics = metrics }
            | AgentStatusUpdate (agentId, status) ->
                let updatedAgents = 
                    acc.TarsAgents 
                    |> List.map (fun agent -> 
                        if agent.Id = agentId then { agent with Status = status } else agent)
                addLogEntry $"Agent {agentId} status: {status}"
                    { acc with TarsAgents = updatedAgents }
            | _ -> acc
        ) model
    
    /// Clean up old components to prevent memory issues
    let cleanupOldComponents (model: Model) =
        let cutoffTime = DateTime.Now.AddMinutes(-5.0)
        let recentComponents = 
            model.DynamicComponents 
            |> List.filter (fun comp -> comp.CreatedAt > cutoffTime)
        
        if recentComponents.Length < model.DynamicComponents.Length then
            let removedCount = model.DynamicComponents.Length - recentComponents.Length
            addLogEntry $"Cleaned up {removedCount} old components"
                { model with DynamicComponents = recentComponents }
        else
            model
