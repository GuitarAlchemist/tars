namespace TarsEngine.FSharp.DynamicUI

open System
open Elmish
open Types
open AgentTeams
open State

module Update =
    
    /// Update function for the Elmish application
    let update (msg: Msg) (model: Model) : Model * Cmd<Msg> =
        match msg with
        | StartDynamicGeneration ->
            if not model.IsGenerating then
                let updatedModel = 
                    addLogEntry "🚀 Starting dynamic UI generation with agent teams" 
                        { model with IsGenerating = true }
                
                // Start agent cycles
                let agentCmds = 
                    model.AgentTeams 
                    |> List.map (fun agent ->
                        Cmd.ofSub (fun dispatch ->
                            let rec agentLoop () =
                                async {
                                    if updatedModel.IsGenerating then
                                        try
                                            // Execute agent cycle
                                            let updatedAgent, newComponents, messages = 
                                                executeAgentCycle agent (box model)
                                            
                                            dispatch (AgentCycleCompleted (agent.Id, newComponents, messages))
                                            
                                            // Wait for next cycle
                                            do! Async.Sleep agent.CycleIntervalMs
                                            return! agentLoop ()
                                        with
                                        | ex ->
                                            dispatch (AddLogEntry $"❌ Agent {agent.Name} error: {ex.Message}")
                                            do! Async.Sleep (agent.CycleIntervalMs * 2) // Wait longer on error
                                            return! agentLoop ()
                                }
                            agentLoop () |> Async.StartImmediate
                        )
                    )
                
                // Start cleanup timer
                let cleanupCmd = 
                    Cmd.ofSub (fun dispatch ->
                        let rec cleanupLoop () =
                            async {
                                do! Async.Sleep 30000 // Every 30 seconds
                                dispatch ClearOldComponents
                                return! cleanupLoop ()
                            }
                        cleanupLoop () |> Async.StartImmediate
                    )
                
                updatedModel, Cmd.batch (cleanupCmd :: agentCmds)
            else
                model, Cmd.none
        
        | StopDynamicGeneration ->
            addLogEntry "⏹️ Stopping dynamic UI generation" 
                { model with IsGenerating = false }, Cmd.none
        
        | AgentCycleCompleted (teamId, newComponents, messages) ->
            let updatedModel = updateAgentTeam teamId newComponents model
            let modelWithMessages = processAgentMessages messages updatedModel
            let finalModel = 
                addLogEntry $"🔄 {teamId} generated {newComponents.Length} components" modelWithMessages
            
            finalModel, Cmd.none
        
        | UpdateRequirements newReqs ->
            addLogEntry $"📋 Requirements updated: {newReqs.VisualizationStyle}" 
                { model with SceneRequirements = newReqs }, Cmd.none
        
        | UpdatePerformance metrics ->
            { model with PerformanceMetrics = metrics }, Cmd.none
        
        | AddLogEntry message ->
            addLogEntry message model, Cmd.none
        
        | ClearOldComponents ->
            cleanupOldComponents model, Cmd.none
        
        | ToggleAgent agentId ->
            let updatedAgents = 
                model.TarsAgents 
                |> List.map (fun agent -> 
                    if agent.Id = agentId then 
                        let newStatus = if agent.Status = Running then Idle else Running
                        { agent with Status = newStatus }
                    else agent)
            
            addLogEntry $"🔄 Toggled agent {agentId}" 
                { model with TarsAgents = updatedAgents }, Cmd.none
    
    /// Subscribe to external events
    let subscription (model: Model) =
        if model.IsGenerating then
            // Subscribe to performance updates
            Cmd.ofSub (fun dispatch ->
                let rec performanceLoop () =
                    async {
                        do! Async.Sleep 1000 // Every second
                        let metrics = {
                            FPS = 60.0 + (Random().NextDouble() - 0.5) * 10.0
                            ComponentCount = model.DynamicComponents.Length
                            ActiveAgents = model.AgentTeams |> List.filter (fun a -> a.IsActive) |> List.length
                            MemoryUsage = Random().NextDouble() * 100.0
                            RenderTime = Random().NextDouble() * 16.0
                            WebGPUActive = true
                        }
                        dispatch (UpdatePerformance metrics)
                        return! performanceLoop ()
                    }
                performanceLoop () |> Async.StartImmediate
            )
        else
            Cmd.none
