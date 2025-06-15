namespace TarsEngine.FSharp.Core.Services

open System
open System.Collections.Concurrent
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Types
open TarsEngine.FSharp.Core.Closures.UnifiedEvolutionaryClosureFactory

/// Service that integrates the unified closure factory with TARS metascript system
module MetascriptClosureIntegrationService =
    
    /// Metascript closure command
    type MetascriptClosureCommand = {
        Command: string
        ClosureType: string
        Name: string
        Parameters: Map<string, obj>
        Context: Map<string, obj>
        ExecutionMode: ExecutionMode
    }
    
    and ExecutionMode =
        | Synchronous
        | Asynchronous
        | Scheduled of DateTime
        | Reactive of trigger: string
    
    /// Closure execution result for metascripts
    type MetascriptClosureResult = {
        CommandId: string
        Success: bool
        Output: string
        Artifacts: string list
        ExecutionTime: TimeSpan
        EvolutionData: Map<string, obj>
        NextSteps: string list
        Error: string option
    }
    
    /// Metascript closure integration service
    type MetascriptClosureIntegrationService(
        logger: ILogger<MetascriptClosureIntegrationService>,
        closureFactory: UnifiedEvolutionaryClosureFactory) =
        
        let activeCommands = ConcurrentDictionary<string, MetascriptClosureCommand>()
        let executionResults = ConcurrentQueue<MetascriptClosureResult>()
        
        /// Parse metascript closure command
        member this.ParseClosureCommand(metascriptLine: string) : MetascriptClosureCommand option =
            try
                // Parse commands like:
                // CLOSURE_CREATE GRAMMAR_EVOLUTION "TeamEvolution" team_name="University Research Team" evolution_goal="Optimize coordination patterns"
                // CLOSURE_CREATE AGENT_3D_VISUALIZATION "AgentViz" agent_types="QRE,CH,NoRegret" count=5
                // CLOSURE_CREATE FRACTAL_METASCRIPT "FractalGen" depth=3 pattern="team_coordination"
                
                let parts = metascriptLine.Split(' ', StringSplitOptions.RemoveEmptyEntries)
                if parts.Length >= 4 && parts.[0] = "CLOSURE_CREATE" then
                    let closureType = parts.[1]
                    let name = parts.[2].Trim('"')
                    
                    // Parse parameters
                    let parameters = 
                        parts.[3..]
                        |> Array.choose (fun part ->
                            if part.Contains('=') then
                                let keyValue = part.Split('=', 2)
                                if keyValue.Length = 2 then
                                    let key = keyValue.[0]
                                    let value = keyValue.[1].Trim('"')
                                    Some (key, box value)
                                else None
                            else None)
                        |> Map.ofArray
                    
                    Some {
                        Command = "CREATE"
                        ClosureType = closureType
                        Name = name
                        Parameters = parameters
                        Context = Map.empty
                        ExecutionMode = Synchronous
                    }
                else
                    None
            with
            | ex ->
                logger.LogError(ex, "Error parsing closure command: {MetascriptLine}", metascriptLine)
                None
        
        /// Execute closure command from metascript
        member this.ExecuteClosureCommand(command: MetascriptClosureCommand) : Async<MetascriptClosureResult> =
            async {
                try
                    let commandId = Guid.NewGuid().ToString("N")[..7]
                    activeCommands.[commandId] <- command
                    
                    logger.LogInformation("🔧 Executing closure command: {ClosureType} '{Name}'", command.ClosureType, command.Name)
                    
                    let startTime = DateTime.UtcNow
                    
                    // Map metascript closure type to evolutionary closure type
                    let evolutionaryClosureType = this.MapToEvolutionaryClosureType(command)
                    
                    match evolutionaryClosureType with
                    | Some closureType ->
                        // Create and execute closure
                        let closure = closureFactory.CreateClosure(closureType, command.Name, command.Context)
                        let! result = closure command.Parameters
                        
                        let endTime = DateTime.UtcNow
                        let executionTime = endTime - startTime
                        
                        let metascriptResult = {
                            CommandId = commandId
                            Success = result.Success
                            Output = this.FormatOutputForMetascript(result.Output)
                            Artifacts = result.GeneratedArtifacts
                            ExecutionTime = executionTime
                            EvolutionData = result.EvolutionData
                            NextSteps = result.NextEvolutionSteps
                            Error = result.Error
                        }
                        
                        executionResults.Enqueue(metascriptResult)
                        return metascriptResult
                    
                    | None ->
                        let errorResult = {
                            CommandId = commandId
                            Success = false
                            Output = ""
                            Artifacts = []
                            ExecutionTime = TimeSpan.Zero
                            EvolutionData = Map.empty
                            NextSteps = []
                            Error = Some $"Unknown closure type: {command.ClosureType}"
                        }
                        executionResults.Enqueue(errorResult)
                        return errorResult
                
                with
                | ex ->
                    logger.LogError(ex, "Error executing closure command")
                    let errorResult = {
                        CommandId = Guid.NewGuid().ToString("N")[..7]
                        Success = false
                        Output = ""
                        Artifacts = []
                        ExecutionTime = TimeSpan.Zero
                        EvolutionData = Map.empty
                        NextSteps = []
                        Error = Some ex.Message
                    }
                    executionResults.Enqueue(errorResult)
                    return errorResult
            }
        
        /// Map metascript closure type to evolutionary closure type
        member private this.MapToEvolutionaryClosureType(command: MetascriptClosureCommand) : EvolutionaryClosureType option =
            match command.ClosureType.ToUpperInvariant() with
            | "GRAMMAR_EVOLUTION" ->
                let teamName = command.Parameters.TryFind("team_name") |> Option.map string |> Option.defaultValue "Default Team"
                let evolutionGoal = command.Parameters.TryFind("evolution_goal") |> Option.map string |> Option.defaultValue "Evolve grammars"
                Some (GrammarEvolutionClosure(teamName, evolutionGoal))
            
            | "AGENT_3D_VISUALIZATION" ->
                let agentTypesStr = command.Parameters.TryFind("agent_types") |> Option.map string |> Option.defaultValue "QRE,CH,NoRegret"
                let agentTypes = this.ParseAgentTypes(agentTypesStr)
                Some (Agent3DVisualizationClosure(agentTypes))
            
            | "FRACTAL_METASCRIPT" ->
                let depth = command.Parameters.TryFind("depth") |> Option.map (fun x -> int (string x)) |> Option.defaultValue 3
                let pattern = command.Parameters.TryFind("pattern") |> Option.map string |> Option.defaultValue "team_coordination"
                Some (FractalMetascriptGeneratorClosure(depth, pattern))
            
            | "UNIVERSITY_TEAM_INTEGRATION" ->
                let teamPath = command.Parameters.TryFind("team_path") |> Option.map string |> Option.defaultValue ".tars/university"
                let grammarPath = command.Parameters.TryFind("grammar_path") |> Option.map string |> Option.defaultValue ".tars/grammars"
                Some (UniversityTeamIntegrationClosure(teamPath, grammarPath))
            
            | "MULTI_AGENT_COORDINATION" ->
                let strategyStr = command.Parameters.TryFind("strategy") |> Option.map string |> Option.defaultValue "Hierarchical"
                let strategy = this.ParseCoordinationStrategy(strategyStr)
                Some (MultiAgentCoordinationClosure(strategy))
            
            | "EVOLUTION_MONITOR" ->
                let sessionIdsStr = command.Parameters.TryFind("session_ids") |> Option.map string |> Option.defaultValue ""
                let sessionIds = if String.IsNullOrEmpty(sessionIdsStr) then [] else sessionIdsStr.Split(',') |> Array.map (fun s -> s.Trim()) |> Array.toList
                Some (RealTimeEvolutionMonitorClosure(sessionIds))
            
            | "WEB_INTERFACE" ->
                let featuresStr = command.Parameters.TryFind("features") |> Option.map string |> Option.defaultValue "3d_visualization,evolution_monitor,team_management"
                let features = featuresStr.Split(',') |> Array.map (fun s -> s.Trim()) |> Array.toList
                Some (EvolutionaryWebInterfaceClosure(features))
            
            | _ -> None
        
        /// Parse agent types from string
        member private this.ParseAgentTypes(agentTypesStr: string) : GameTheoryAgentType list =
            agentTypesStr.Split(',')
            |> Array.map (fun typeStr ->
                match typeStr.Trim().ToUpperInvariant() with
                | "QRE" -> QuantalResponseEquilibrium(1.2)
                | "CH" -> CognitiveHierarchy(4)
                | "NOREGRET" -> NoRegretLearning(0.95)
                | "EGT" -> EvolutionaryGameTheory(0.05)
                | "CE" -> CorrelatedEquilibrium([|"signal1"; "signal2"|])
                | "ML" -> MachineLearningAgent("neural_net")
                | _ -> QuantalResponseEquilibrium(1.0))
            |> Array.toList
        
        /// Parse coordination strategy from string
        member private this.ParseCoordinationStrategy(strategyStr: string) : CoordinationStrategy =
            match strategyStr.ToUpperInvariant() with
            | "HIERARCHICAL" -> Hierarchical("leader_agent")
            | "DEMOCRATIC" -> Democratic
            | "SPECIALIZED" -> Specialized
            | "SWARM" -> Swarm
            | "FRACTAL" -> FractalSelfOrganizing
            | _ -> Hierarchical("default_leader")
        
        /// Format output for metascript consumption
        member private this.FormatOutputForMetascript(output: obj option) : string =
            match output with
            | Some obj ->
                match obj with
                | :? string as str -> str
                | :? Map<string, obj> as map ->
                    map 
                    |> Map.toList 
                    |> List.map (fun (k, v) -> $"{k}: {v}")
                    |> String.concat "; "
                | _ -> obj.ToString()
            | None -> ""
        
        /// Generate metascript closure documentation
        member this.GenerateClosureDocumentation() : string =
            [
                "# TARS Metascript Closure Integration"
                ""
                "## Available Closure Types"
                ""
                "### GRAMMAR_EVOLUTION"
                "Creates a grammar evolution session for a university team."
                "```"
                "CLOSURE_CREATE GRAMMAR_EVOLUTION \"TeamEvolution\" team_name=\"University Research Team\" evolution_goal=\"Optimize coordination patterns\""
                "```"
                ""
                "### AGENT_3D_VISUALIZATION"
                "Creates 3D visualization of game theory agents."
                "```"
                "CLOSURE_CREATE AGENT_3D_VISUALIZATION \"AgentViz\" agent_types=\"QRE,CH,NoRegret\" count=5"
                "```"
                ""
                "### FRACTAL_METASCRIPT"
                "Generates fractal metascripts with recursive patterns."
                "```"
                "CLOSURE_CREATE FRACTAL_METASCRIPT \"FractalGen\" depth=3 pattern=\"team_coordination\""
                "```"
                ""
                "### UNIVERSITY_TEAM_INTEGRATION"
                "Integrates existing university teams with evolution system."
                "```"
                "CLOSURE_CREATE UNIVERSITY_TEAM_INTEGRATION \"TeamIntegration\" team_path=\".tars/university\" grammar_path=\".tars/grammars\""
                "```"
                ""
                "### MULTI_AGENT_COORDINATION"
                "Creates coordinated multi-agent teams."
                "```"
                "CLOSURE_CREATE MULTI_AGENT_COORDINATION \"Coordination\" strategy=\"Hierarchical\" agent_count=5"
                "```"
                ""
                "### EVOLUTION_MONITOR"
                "Creates real-time evolution monitoring dashboard."
                "```"
                "CLOSURE_CREATE EVOLUTION_MONITOR \"Monitor\" session_ids=\"session1,session2,session3\""
                "```"
                ""
                "### WEB_INTERFACE"
                "Generates evolutionary web interface with specified features."
                "```"
                "CLOSURE_CREATE WEB_INTERFACE \"WebUI\" features=\"3d_visualization,evolution_monitor,team_management\""
                "```"
                ""
                "## Usage in Metascripts"
                ""
                "```trsx"
                "meta {"
                "  name: \"Evolutionary Closure Demo\""
                "  version: \"1.0\""
                "}"
                ""
                "reasoning {"
                "  This metascript demonstrates the integration of evolutionary"
                "  closures with the TARS metascript system."
                "}"
                ""
                "# Create grammar evolution session"
                "CLOSURE_CREATE GRAMMAR_EVOLUTION \"UniversityEvolution\" team_name=\"Research Team\" evolution_goal=\"Optimize DSL patterns\""
                ""
                "# Create 3D visualization"
                "CLOSURE_CREATE AGENT_3D_VISUALIZATION \"TeamViz\" agent_types=\"QRE,CH,NoRegret,EGT,CE\" count=5"
                ""
                "# Generate fractal coordination metascript"
                "CLOSURE_CREATE FRACTAL_METASCRIPT \"CoordinationFractal\" depth=4 pattern=\"dynamic_formation\""
                ""
                "# Create web interface"
                "CLOSURE_CREATE WEB_INTERFACE \"EvolutionUI\" features=\"3d_visualization,evolution_monitor,grammar_browser,metascript_executor\""
                ""
                "FSHARP {"
                "  // Access closure results"
                "  let evolutionResult = getClosureResult \"UniversityEvolution\""
                "  let vizResult = getClosureResult \"TeamViz\""
                "  "
                "  printfn \"Evolution session: %A\" evolutionResult"
                "  printfn \"3D visualization: %A\" vizResult"
                "}"
                "```"
            ] |> String.concat "\n"
        
        /// Get execution results
        member this.GetExecutionResults() : MetascriptClosureResult list =
            executionResults |> Seq.toList
        
        /// Get active commands
        member this.GetActiveCommands() : Map<string, MetascriptClosureCommand> =
            activeCommands
            |> Seq.map (fun kvp -> kvp.Key, kvp.Value)
            |> Map.ofSeq
        
        /// Create sample metascript with closure integration
        member this.GenerateSampleMetascript() : string =
            [
                "#!/usr/bin/env trsx"
                "# TARS Evolutionary Closure Integration Demo"
                $"# Generated: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss}"
                ""
                "meta {"
                "  name: \"Evolutionary Closure Demo\""
                "  version: \"1.0\""
                "  description: \"Demonstrates integrated evolutionary closures\""
                "  author: \"TARS Evolutionary System\""
                "}"
                ""
                "reasoning {"
                "  This metascript showcases the power of integrating evolutionary"
                "  grammar generation, 3D agent visualization, and university team"
                "  coordination through the unified closure factory system."
                "  "
                "  The closures work together to create a comprehensive evolutionary"
                "  environment where teams can autonomously evolve their grammars"
                "  while being visualized in real-time 3D space."
                "}"
                ""
                "# Initialize university team integration"
                "CLOSURE_CREATE UNIVERSITY_TEAM_INTEGRATION \"TeamSetup\" team_path=\".tars/university\" grammar_path=\".tars/grammars\""
                ""
                "# Create grammar evolution session"
                "CLOSURE_CREATE GRAMMAR_EVOLUTION \"MainEvolution\" team_name=\"University Research Team\" evolution_goal=\"Develop advanced DSL patterns for AI coordination\""
                ""
                "# Spawn 3D agent visualization"
                "CLOSURE_CREATE AGENT_3D_VISUALIZATION \"TeamVisualization\" agent_types=\"QRE,CH,NoRegret,EGT,CE,ML\" count=6"
                ""
                "# Generate fractal coordination metascript"
                "CLOSURE_CREATE FRACTAL_METASCRIPT \"FractalCoordination\" depth=3 pattern=\"team_coordination\""
                ""
                "# Create multi-agent coordination"
                "CLOSURE_CREATE MULTI_AGENT_COORDINATION \"TeamCoordination\" strategy=\"FractalSelfOrganizing\" agent_count=6"
                ""
                "# Set up evolution monitoring"
                "CLOSURE_CREATE EVOLUTION_MONITOR \"EvolutionDashboard\" session_ids=\"MainEvolution\""
                ""
                "# Generate web interface"
                "CLOSURE_CREATE WEB_INTERFACE \"EvolutionInterface\" features=\"3d_visualization,evolution_monitor,team_management,grammar_browser,metascript_executor\""
                ""
                "FSHARP {"
                "  // Process closure results"
                "  let teamSetupResult = getClosureResult \"TeamSetup\""
                "  let evolutionResult = getClosureResult \"MainEvolution\""
                "  let visualizationResult = getClosureResult \"TeamVisualization\""
                "  let webInterfaceResult = getClosureResult \"EvolutionInterface\""
                "  "
                "  printfn \"🎓 Team Setup: %s\" (if teamSetupResult.Success then \"✅ Success\" else \"❌ Failed\")"
                "  printfn \"🧬 Evolution: %s\" (if evolutionResult.Success then \"✅ Active\" else \"❌ Inactive\")"
                "  printfn \"🎯 Visualization: %s\" (if visualizationResult.Success then \"✅ Running\" else \"❌ Stopped\")"
                "  printfn \"🌐 Web Interface: %s\" (if webInterfaceResult.Success then \"✅ Available\" else \"❌ Unavailable\")"
                "  "
                "  // Display evolution metrics"
                "  match evolutionResult.EvolutionData.TryFind \"session_id\" with"
                "  | Some sessionId -> printfn \"📊 Evolution Session: %A\" sessionId"
                "  | None -> printfn \"⚠️ No evolution session found\""
                "  "
                "  // Show generated artifacts"
                "  let allArtifacts = [teamSetupResult; evolutionResult; visualizationResult; webInterfaceResult]"
                "                    |> List.collect (fun result -> result.Artifacts)"
                "  "
                "  printfn \"📁 Generated Artifacts:\""
                "  allArtifacts |> List.iter (fun artifact -> printfn \"  - %s\" artifact)"
                "}"
                ""
                "# Execute fractal coordination metascript"
                "EXECUTE_METASCRIPT \"FractalCoordination\""
                ""
                "# Monitor evolution progress"
                "WHILE evolution_active DO"
                "  UPDATE_EVOLUTION_METRICS \"MainEvolution\""
                "  UPDATE_3D_VISUALIZATION \"TeamVisualization\""
                "  WAIT 5000ms"
                "END"
                ""
                "# Export results"
                "EXPORT_EVOLUTION_RESULTS \"MainEvolution\" \"output/evolution_results.json\""
                "EXPORT_GRAMMARS \"MainEvolution\" \"output/evolved_grammars/\""
                "EXPORT_3D_SCENE \"TeamVisualization\" \"output/3d_scene.json\""
            ] |> String.concat "\n"
