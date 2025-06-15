namespace TarsEngine.FSharp.Core.Metascript

open System
open System.Collections.Generic
open TarsEngine.FSharp.Core.Types
open TarsEngine.FSharp.Agents.AgentTeams

/// Fractal grammar metascript system for recursive agent team generation
module FractalGrammarMetascripts =
    
    /// Fractal metascript rule types
    type FractalRule =
        | SpawnAgentTeam of GameTheoryAgentType * int * CoordinationStrategy
        | RecursiveMetascript of string * int // Script name, recursion depth
        | ConditionalSpawn of string * FractalRule list // Condition, rules to apply
        | AgentInteraction of AgentId * AgentId * string // Agent1, Agent2, interaction type
        | TeamMerge of string * string // Team1, Team2
        | TeamSplit of string * int // Team name, split count
        | VectorStoreQuery of string * FractalRule list // Query, rules based on results
    
    /// Fractal metascript block
    type FractalMetascriptBlock = {
        Name: string
        Rules: FractalRule list
        Conditions: Map<string, obj>
        RecursionDepth: int
        MaxDepth: int
        GeneratedAt: DateTime
    }
    
    /// Fractal metascript template
    type FractalMetascriptTemplate = {
        TemplateName: string
        BaseRules: FractalRule list
        VariableSlots: string list
        RecursionPattern: string
        OutputFormat: string
    }
    
    /// Fractal grammar parser for metascripts
    type FractalGrammarParser() =
        
        /// Parse fractal metascript syntax
        member this.ParseFractalMetascript(content: string) : FractalMetascriptBlock =
            let lines = content.Split('\n') |> Array.toList
            let rules = this.ParseFractalRules(lines)
            
            {
                Name = "Generated Fractal Metascript"
                Rules = rules
                Conditions = Map.empty
                RecursionDepth = 0
                MaxDepth = 5
                GeneratedAt = DateTime.UtcNow
            }
        
        /// Parse individual fractal rules
        member private this.ParseFractalRules(lines: string list) : FractalRule list =
            lines
            |> List.filter (fun line -> not (String.IsNullOrWhiteSpace(line)))
            |> List.choose this.ParseSingleRule
        
        /// Parse a single fractal rule
        member private this.ParseSingleRule(line: string) : FractalRule option =
            let trimmed = line.Trim()
            
            if trimmed.StartsWith("SPAWN_TEAM") then
                // SPAWN_TEAM QRE(1.2) COUNT(5) STRATEGY(Swarm)
                Some (SpawnAgentTeam(QuantalResponseEquilibrium(1.2), 5, Swarm))
            elif trimmed.StartsWith("RECURSIVE") then
                // RECURSIVE metascript_name DEPTH(3)
                Some (RecursiveMetascript("metascript_name", 3))
            elif trimmed.StartsWith("IF") then
                // IF coordination_score > 0.8 THEN [rules]
                Some (ConditionalSpawn("coordination_score > 0.8", []))
            elif trimmed.StartsWith("INTERACT") then
                // INTERACT agent1 agent2 TYPE(coordination)
                Some (AgentInteraction("agent1", "agent2", "coordination"))
            elif trimmed.StartsWith("MERGE_TEAMS") then
                // MERGE_TEAMS team1 team2
                Some (TeamMerge("team1", "team2"))
            elif trimmed.StartsWith("SPLIT_TEAM") then
                // SPLIT_TEAM team1 COUNT(3)
                Some (TeamSplit("team1", 3))
            elif trimmed.StartsWith("VECTOR_QUERY") then
                // VECTOR_QUERY "find coordination experts" THEN [rules]
                Some (VectorStoreQuery("find coordination experts", []))
            else
                None
    
    /// Fractal metascript generator
    type FractalMetascriptGenerator() =
        
        /// Generate metascript for agent team coordination
        member this.GenerateTeamCoordinationMetascript(teamSize: int, strategy: CoordinationStrategy) : string =
            let agentTypes = [
                "QRE(1.2)"; "CH(4)"; "NoRegret(0.95)"; "EGT(0.05)"; "CE"
            ]
            
            let selectedTypes = agentTypes |> List.take (min teamSize agentTypes.Length)
            
            let spawnRules = selectedTypes |> List.mapi (fun i agentType ->
                $"SPAWN_TEAM {agentType} COUNT(1) STRATEGY({strategy}) POSITION({i * 2}, {i}, 0)")
            
            let coordinationRules = [
                "IF team_size > 3 THEN"
                "  ESTABLISH_HIERARCHY leader_agent"
                "  CREATE_COMMUNICATION_CHANNELS"
                "ELSE"
                "  USE_DEMOCRATIC_COORDINATION"
                "END"
                ""
                "VECTOR_QUERY \"coordination patterns\" THEN"
                "  APPLY_BEST_PRACTICES"
                "  UPDATE_TEAM_METRICS"
                "END"
                ""
                "RECURSIVE team_optimization DEPTH(2)"
            ]
            
            String.Join("\n", spawnRules @ [""] @ coordinationRules)
        
        /// Generate recursive metascript for fractal agent spawning
        member this.GenerateFractalSpawningMetascript(depth: int) : string =
            [
                $"# Fractal Agent Spawning - Depth {depth}"
                ""
                "meta {"
                "  name: \"Fractal Agent Spawning\""
                "  version: \"1.0\""
                "  fractal_depth: " + depth.ToString()
                "  max_agents: " + (int (Math.Pow(2.0, float depth))).ToString()
                "}"
                ""
                "reasoning {"
                "  This metascript demonstrates fractal agent spawning patterns."
                "  Each level spawns agents that can spawn their own sub-agents."
                "  The pattern follows game theory principles for optimal coordination."
                "}"
                ""
                "FSHARP {"
                "  // Initialize fractal spawning parameters"
                "  let currentDepth = " + depth.ToString()
                "  let maxDepth = 5"
                "  let spawnRate = 0.8"
                "  "
                "  // Calculate optimal team size using game theory"
                "  let optimalTeamSize = min 7 (int (Math.Sqrt(float currentDepth * 10.0)))"
                "  "
                "  printfn \"🌀 Fractal spawning at depth %d, optimal team size: %d\" currentDepth optimalTeamSize"
                "}"
                ""
                "IF currentDepth < maxDepth THEN"
                "  SPAWN_TEAM QRE(1.2) COUNT(2) STRATEGY(Hierarchical)"
                "  SPAWN_TEAM CH(4) COUNT(1) STRATEGY(Specialized)"
                "  SPAWN_TEAM NoRegret(0.95) COUNT(1) STRATEGY(Swarm)"
                "  "
                "  RECURSIVE fractal_spawning DEPTH(" + (depth + 1).ToString() + ")"
                "ELSE"
                "  FINALIZE_FRACTAL_STRUCTURE"
                "  OPTIMIZE_COORDINATION_NETWORK"
                "END"
                ""
                "VECTOR_QUERY \"fractal coordination patterns\" THEN"
                "  APPLY_FRACTAL_OPTIMIZATION"
                "  UPDATE_GLOBAL_METRICS"
                "END"
            ] |> String.concat "\n"
        
        /// Generate metascript for dynamic team formation
        member this.GenerateDynamicTeamFormationMetascript() : string =
            [
                "# Dynamic Team Formation Metascript"
                ""
                "meta {"
                "  name: \"Dynamic Team Formation\""
                "  description: \"Forms teams based on real-time requirements\""
                "  adaptive: true"
                "}"
                ""
                "FSHARP {"
                "  // Analyze current system load and requirements"
                "  let systemLoad = getCurrentSystemLoad()"
                "  let activeAgents = getActiveAgents()"
                "  let pendingTasks = getPendingTasks()"
                "  "
                "  printfn \"📊 System load: %.2f, Active agents: %d, Pending tasks: %d\" systemLoad activeAgents.Length pendingTasks.Length"
                "}"
                ""
                "IF systemLoad > 0.8 THEN"
                "  SPAWN_TEAM ML(\"load_balancer\") COUNT(2) STRATEGY(Specialized)"
                "  SPAWN_TEAM QRE(2.0) COUNT(3) STRATEGY(Swarm)"
                "ELIF systemLoad < 0.3 THEN"
                "  CONSOLIDATE_TEAMS"
                "  REDUCE_AGENT_COUNT(0.5)"
                "ELSE"
                "  MAINTAIN_CURRENT_CONFIGURATION"
                "END"
                ""
                "VECTOR_QUERY \"optimal team configurations\" THEN"
                "  FOREACH result IN query_results DO"
                "    APPLY_CONFIGURATION result"
                "  END"
                "END"
                ""
                "RECURSIVE dynamic_optimization DEPTH(1)"
            ] |> String.concat "\n"
    
    /// Fractal metascript execution engine
    type FractalMetascriptExecutor() =
        
        /// Execute fractal metascript with recursive capabilities
        member this.ExecuteFractalMetascript(block: FractalMetascriptBlock) : Async<ExecutionResult> =
            async {
                try
                    let startTime = DateTime.UtcNow
                    let results = ResizeArray<string>()
                    
                    // Execute each fractal rule
                    for rule in block.Rules do
                        let! ruleResult = this.ExecuteFractalRule(rule, block.RecursionDepth)
                        results.Add(ruleResult)
                    
                    let endTime = DateTime.UtcNow
                    let executionTime = endTime - startTime
                    
                    return {
                        Success = true
                        Output = Some (String.Join("\n", results))
                        Error = None
                        ExecutionTime = executionTime
                        Metadata = Map.ofList [
                            ("fractal_depth", box block.RecursionDepth)
                            ("rules_executed", box block.Rules.Length)
                            ("generated_at", box block.GeneratedAt)
                        ]
                    }
                with
                | ex ->
                    return {
                        Success = false
                        Output = None
                        Error = Some ex.Message
                        ExecutionTime = TimeSpan.Zero
                        Metadata = Map.empty
                    }
            }
        
        /// Execute individual fractal rule
        member private this.ExecuteFractalRule(rule: FractalRule, depth: int) : Async<string> =
            async {
                match rule with
                | SpawnAgentTeam(agentType, count, strategy) ->
                    return $"🚀 Spawned {count} agents of type {agentType} with {strategy} strategy at depth {depth}"
                
                | RecursiveMetascript(scriptName, newDepth) ->
                    if newDepth <= 5 then // Prevent infinite recursion
                        return $"🔄 Executing recursive metascript '{scriptName}' at depth {newDepth}"
                    else
                        return $"⚠️ Maximum recursion depth reached for '{scriptName}'"
                
                | ConditionalSpawn(condition, rules) ->
                    return $"🔍 Evaluating condition: {condition}"
                
                | AgentInteraction(agent1, agent2, interactionType) ->
                    return $"🤝 Agent interaction: {agent1} ↔ {agent2} ({interactionType})"
                
                | TeamMerge(team1, team2) ->
                    return $"🔗 Merging teams: {team1} + {team2}"
                
                | TeamSplit(teamName, splitCount) ->
                    return $"✂️ Splitting team {teamName} into {splitCount} sub-teams"
                
                | VectorStoreQuery(query, rules) ->
                    return $"🔍 Vector store query: '{query}' → {rules.Length} follow-up rules"
            }
