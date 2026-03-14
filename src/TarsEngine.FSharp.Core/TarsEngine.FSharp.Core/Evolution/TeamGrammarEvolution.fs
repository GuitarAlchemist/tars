namespace TarsEngine.FSharp.Core.Evolution

open System
open System.IO
open System.Text.Json
open System.Collections.Concurrent
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Types
open TarsEngine.FSharp.Core.Services.Agent3DIntegrationService

/// Team grammar evolution system that integrates existing teams with grammars
module TeamGrammarEvolution =
    
    /// Existing grammar definition
    type ExistingGrammar = {
        Id: string
        File: string
        Origin: string
        Version: string
        Content: string
        Language: string
        Tags: string list
        LastModified: DateTime
    }
    
    /// Team agent from university configuration
    type UniversityAgent = {
        Name: string
        Specialization: string
        Capabilities: string list
        OutputFormats: string list
        GrammarAffinity: string list // Which grammars this agent works with
        EvolutionRole: EvolutionRole
    }
    
    and EvolutionRole =
        | GrammarCreator // Creates new grammar rules
        | GrammarMutator // Modifies existing rules
        | GrammarValidator // Tests and validates grammars
        | GrammarSynthesizer // Combines multiple grammars
        | GrammarOptimizer // Improves performance
    
    /// Evolved grammar rule
    type EvolvedGrammarRule = {
        RuleId: string
        OriginalGrammar: string
        RulePattern: string
        RuleBody: string
        CreatedBy: string // Agent name
        FitnessScore: float
        GenerationNumber: int
        ParentRules: string list
        EvolutionHistory: string list
        UsageCount: int
        SuccessRate: float
    }
    
    /// Grammar evolution session
    type GrammarEvolutionSession = {
        SessionId: string
        TeamName: string
        ParticipatingAgents: UniversityAgent list
        BaseGrammars: ExistingGrammar list
        EvolvedRules: EvolvedGrammarRule list
        CurrentGeneration: int
        EvolutionGoal: string
        StartTime: DateTime
        LastActivity: DateTime
        IsActive: bool
        PerformanceMetrics: Map<string, float>
    }
    
    /// Grammar evolution engine
    type TeamGrammarEvolutionEngine(logger: ILogger<TeamGrammarEvolutionEngine>, integrationService: Agent3DIntegrationService) =
        
        let activeSessions = ConcurrentDictionary<string, GrammarEvolutionSession>()
        let grammarLibrary = ConcurrentDictionary<string, ExistingGrammar>()
        
        /// Load existing grammars from .tars/grammars directory
        member this.LoadExistingGrammars(grammarPath: string) : ExistingGrammar list =
            try
                let indexPath = Path.Combine(grammarPath, "grammar_index.json")
                if File.Exists(indexPath) then
                    let indexContent = File.ReadAllText(indexPath)
                    let grammarEntries = JsonSerializer.Deserialize<JsonElement[]>(indexContent)
                    
                    grammarEntries
                    |> Array.choose (fun entry ->
                        try
                            let id = entry.GetProperty("Id").GetString()
                            let file = entry.GetProperty("File").GetString()
                            let origin = entry.GetProperty("Origin").GetString()
                            let version = entry.GetProperty("Version").GetString()
                            let lastModified = DateTime.Parse(entry.GetProperty("LastModified").GetString())
                            
                            let grammarFilePath = Path.Combine(grammarPath, file)
                            if File.Exists(grammarFilePath) then
                                let content = File.ReadAllText(grammarFilePath)
                                let grammar = {
                                    Id = id
                                    File = file
                                    Origin = origin
                                    Version = version
                                    Content = content
                                    Language = this.ExtractLanguageFromContent(content)
                                    Tags = this.ExtractTagsFromContent(content)
                                    LastModified = lastModified
                                }
                                grammarLibrary.[id] <- grammar
                                Some grammar
                            else
                                None
                        with
                        | ex ->
                            logger.LogWarning(ex, "Error loading grammar entry")
                            None)
                    |> Array.toList
                else
                    logger.LogWarning("Grammar index not found at {IndexPath}", indexPath)
                    []
            with
            | ex ->
                logger.LogError(ex, "Error loading existing grammars from {GrammarPath}", grammarPath)
                []
        
        /// Extract language from grammar content
        member private this.ExtractLanguageFromContent(content: string) : string =
            if content.Contains("LANG(\"EBNF\")") then "EBNF"
            elif content.Contains("LANG(\"FLUX\")") then "FLUX"
            elif content.Contains("LANG(\"TRSX\")") then "TRSX"
            else "Unknown"
        
        /// Extract tags from grammar content
        member private this.ExtractTagsFromContent(content: string) : string list =
            try
                let lines = content.Split('\n')
                let tagLine = lines |> Array.tryFind (fun line -> line.Trim().StartsWith("tags:"))
                match tagLine with
                | Some line ->
                    let tagsPart = line.Substring(line.IndexOf('[') + 1, line.IndexOf(']') - line.IndexOf('[') - 1)
                    tagsPart.Split(',') 
                    |> Array.map (fun tag -> tag.Trim().Trim('"'))
                    |> Array.toList
                | None -> []
            with
            | _ -> []
        
        /// Load university team and assign evolution roles
        member this.LoadUniversityTeam(teamConfigPath: string) : UniversityAgent list =
            try
                let configContent = File.ReadAllText(teamConfigPath)
                let config = JsonSerializer.Deserialize<JsonElement>(configContent)
                let agents = config.GetProperty("agents")
                
                agents.EnumerateArray()
                |> Seq.mapi (fun i agentElement ->
                    let name = agentElement.GetProperty("name").GetString()
                    let specialization = agentElement.GetProperty("specialization").GetString()
                    let capabilities = 
                        agentElement.GetProperty("capabilities").EnumerateArray()
                        |> Seq.map (fun cap -> cap.GetString())
                        |> Seq.toList
                    let outputFormats = 
                        agentElement.GetProperty("output_formats").EnumerateArray()
                        |> Seq.map (fun fmt -> fmt.GetString())
                        |> Seq.toList
                    
                    // Assign evolution roles based on specialization
                    let evolutionRole = 
                        match specialization.ToLowerInvariant() with
                        | s when s.Contains("research director") -> GrammarSynthesizer
                        | s when s.Contains("cs researcher") -> GrammarCreator
                        | s when s.Contains("data scientist") -> GrammarOptimizer
                        | s when s.Contains("academic writer") -> GrammarMutator
                        | s when s.Contains("peer reviewer") -> GrammarValidator
                        | s when s.Contains("knowledge synthesizer") -> GrammarSynthesizer
                        | s when s.Contains("ethics officer") -> GrammarValidator
                        | _ -> GrammarMutator
                    
                    // Assign grammar affinity based on capabilities
                    let grammarAffinity = 
                        capabilities
                        |> List.choose (fun cap ->
                            if cap.ToLowerInvariant().Contains("algorithm") then Some "MiniQuery"
                            elif cap.ToLowerInvariant().Contains("mathematical") || cap.ToLowerInvariant().Contains("statistical") then Some "Wolfram"
                            elif cap.ToLowerInvariant().Contains("uri") || cap.ToLowerInvariant().Contains("web") then Some "RFC3986_URI"
                            else None)
                        |> List.distinct
                    
                    {
                        Name = name
                        Specialization = specialization
                        Capabilities = capabilities
                        OutputFormats = outputFormats
                        GrammarAffinity = if grammarAffinity.IsEmpty then ["MiniQuery"] else grammarAffinity
                        EvolutionRole = evolutionRole
                    })
                |> Seq.toList
            with
            | ex ->
                logger.LogError(ex, "Error loading university team from {TeamConfigPath}", teamConfigPath)
                []
        
        /// Start grammar evolution session for a team
        member this.StartEvolutionSession(teamName: string, agents: UniversityAgent list, grammars: ExistingGrammar list, goal: string) : string =
            let sessionId = $"evolution_{teamName.Replace(" ", "_").ToLowerInvariant()}_{DateTime.UtcNow:yyyyMMdd_HHmmss}"
            
            let session = {
                SessionId = sessionId
                TeamName = teamName
                ParticipatingAgents = agents
                BaseGrammars = grammars
                EvolvedRules = []
                CurrentGeneration = 0
                EvolutionGoal = goal
                StartTime = DateTime.UtcNow
                LastActivity = DateTime.UtcNow
                IsActive = true
                PerformanceMetrics = Map.ofList [
                    ("grammar_diversity", 0.5)
                    ("rule_effectiveness", 0.6)
                    ("team_coordination", 0.7)
                    ("innovation_rate", 0.4)
                ]
            }
            
            activeSessions.[sessionId] <- session
            
            // Spawn 3D agents for each team member
            agents |> List.iter (fun agent ->
                let agentType = this.MapEvolutionRoleToGameTheory(agent.EvolutionRole)
                let agentId = integrationService.SpawnAgent(agentType)
                logger.LogInformation("🧬 Spawned 3D agent {AgentId} for {AgentName} ({Role})", 
                                     agentId, agent.Name, agent.EvolutionRole)
            )
            
            logger.LogInformation("🧬 Started grammar evolution session {SessionId} for team {TeamName} with {AgentCount} agents",
                                 sessionId, teamName, agents.Length)
            
            sessionId
        
        /// Map evolution role to game theory agent type
        member private this.MapEvolutionRoleToGameTheory(role: EvolutionRole) : GameTheoryAgentType =
            match role with
            | GrammarCreator -> QuantalResponseEquilibrium(1.5) // High creativity
            | GrammarMutator -> NoRegretLearning(0.9) // Adaptive learning
            | GrammarValidator -> CognitiveHierarchy(5) // Deep analysis
            | GrammarSynthesizer -> CorrelatedEquilibrium([|"synthesis"; "integration"|]) // Coordination
            | GrammarOptimizer -> EvolutionaryGameTheory(0.02) // Optimization focus
        
        /// Evolve grammars for one generation
        member this.EvolveGeneration(sessionId: string) : Async<bool> =
            async {
                match activeSessions.TryGetValue(sessionId) with
                | true, session when session.IsActive ->
                    try
                        logger.LogInformation("🧬 Evolving generation {Generation} for session {SessionId}",
                                             session.CurrentGeneration + 1, sessionId)
                        
                        let newRules = ResizeArray<EvolvedGrammarRule>()
                        
                        // Each agent contributes based on their role
                        for agent in session.ParticipatingAgents do
                            let! agentContribution = this.AgentContributeToEvolution(agent, session)
                            newRules.AddRange(agentContribution)
                        
                        // Evaluate and select best rules
                        let evaluatedRules = this.EvaluateRules(newRules |> Seq.toList, session)
                        let selectedRules = this.SelectBestRules(evaluatedRules, 10) // Keep top 10
                        
                        // Update session
                        let updatedSession = {
                            session with
                                EvolvedRules = session.EvolvedRules @ selectedRules
                                CurrentGeneration = session.CurrentGeneration + 1
                                LastActivity = DateTime.UtcNow
                                PerformanceMetrics = this.UpdatePerformanceMetrics(session, selectedRules)
                        }
                        
                        activeSessions.[sessionId] <- updatedSession
                        
                        // Generate evolved grammar file
                        let evolvedGrammar = this.GenerateEvolvedGrammar(updatedSession)
                        let outputPath = $".tars/evolution/grammars/{sessionId}_gen{updatedSession.CurrentGeneration}.tars"
                        this.SaveEvolvedGrammar(outputPath, evolvedGrammar)
                        
                        // Update 3D visualization
                        this.Update3DVisualization(updatedSession)
                        
                        logger.LogInformation("✅ Generation {Generation} complete. {RuleCount} new rules evolved",
                                             updatedSession.CurrentGeneration, selectedRules.Length)
                        
                        return true
                    with
                    | ex ->
                        logger.LogError(ex, "Error during grammar evolution for session {SessionId}", sessionId)
                        return false
                | _ ->
                    logger.LogWarning("Session {SessionId} not found or not active", sessionId)
                    return false
            }
        
        /// Agent contributes to evolution based on role
        member private this.AgentContributeToEvolution(agent: UniversityAgent, session: GrammarEvolutionSession) : Async<EvolvedGrammarRule list> =
            async {
                let rules = ResizeArray<EvolvedGrammarRule>()
                
                match agent.EvolutionRole with
                | GrammarCreator ->
                    // Create new grammar rules
                    for grammar in session.BaseGrammars do
                        if agent.GrammarAffinity |> List.contains grammar.Id then
                            let newRule = {
                                RuleId = $"created_{agent.Name.Replace(" ", "_")}_{Guid.NewGuid().ToString("N")[..7]}"
                                OriginalGrammar = grammar.Id
                                RulePattern = $"evolved_{grammar.Id.ToLowerInvariant()}_pattern"
                                RuleBody = this.GenerateNewRuleBody(grammar, agent)
                                CreatedBy = agent.Name
                                FitnessScore = 0.6 + Random().NextDouble() * 0.3
                                GenerationNumber = session.CurrentGeneration + 1
                                ParentRules = []
                                EvolutionHistory = [$"Created by {agent.Name} in generation {session.CurrentGeneration + 1}"]
                                UsageCount = 0
                                SuccessRate = 0.5
                            }
                            rules.Add(newRule)
                
                | GrammarMutator ->
                    // Mutate existing rules
                    let existingRules = session.EvolvedRules |> List.take (min 3 session.EvolvedRules.Length)
                    for existingRule in existingRules do
                        let mutatedRule = {
                            existingRule with
                                RuleId = $"mutated_{agent.Name.Replace(" ", "_")}_{Guid.NewGuid().ToString("N")[..7]}"
                                RuleBody = this.MutateRuleBody(existingRule.RuleBody, agent)
                                CreatedBy = agent.Name
                                FitnessScore = existingRule.FitnessScore + (Random().NextDouble() - 0.5) * 0.2
                                GenerationNumber = session.CurrentGeneration + 1
                                ParentRules = [existingRule.RuleId]
                                EvolutionHistory = $"Mutated by {agent.Name}" :: existingRule.EvolutionHistory
                        }
                        rules.Add(mutatedRule)
                
                | GrammarValidator ->
                    // Validate and improve existing rules
                    let recentRules = session.EvolvedRules |> List.filter (fun r -> r.GenerationNumber >= session.CurrentGeneration - 1)
                    for rule in recentRules do
                        let validatedRule = {
                            rule with
                                FitnessScore = min 1.0 (rule.FitnessScore + 0.1) // Boost validated rules
                                SuccessRate = min 1.0 (rule.SuccessRate + 0.15)
                                EvolutionHistory = $"Validated by {agent.Name}" :: rule.EvolutionHistory
                        }
                        rules.Add(validatedRule)
                
                | GrammarSynthesizer ->
                    // Combine multiple rules
                    if session.EvolvedRules.Length >= 2 then
                        let rule1 = session.EvolvedRules.[Random().Next(session.EvolvedRules.Length)]
                        let rule2 = session.EvolvedRules.[Random().Next(session.EvolvedRules.Length)]
                        
                        let synthesizedRule = {
                            RuleId = $"synthesized_{agent.Name.Replace(" ", "_")}_{Guid.NewGuid().ToString("N")[..7]}"
                            OriginalGrammar = rule1.OriginalGrammar
                            RulePattern = $"synthesized_{rule1.RulePattern}_{rule2.RulePattern}"
                            RuleBody = this.SynthesizeRuleBodies(rule1.RuleBody, rule2.RuleBody, agent)
                            CreatedBy = agent.Name
                            FitnessScore = (rule1.FitnessScore + rule2.FitnessScore) / 2.0 + 0.1
                            GenerationNumber = session.CurrentGeneration + 1
                            ParentRules = [rule1.RuleId; rule2.RuleId]
                            EvolutionHistory = [$"Synthesized from {rule1.RuleId} and {rule2.RuleId} by {agent.Name}"]
                            UsageCount = 0
                            SuccessRate = 0.7
                        }
                        rules.Add(synthesizedRule)
                
                | GrammarOptimizer ->
                    // Optimize existing rules for performance
                    let topRules = session.EvolvedRules |> List.sortByDescending (fun r -> r.FitnessScore) |> List.take (min 2 session.EvolvedRules.Length)
                    for rule in topRules do
                        let optimizedRule = {
                            rule with
                                RuleId = $"optimized_{agent.Name.Replace(" ", "_")}_{Guid.NewGuid().ToString("N")[..7]}"
                                RuleBody = this.OptimizeRuleBody(rule.RuleBody, agent)
                                CreatedBy = agent.Name
                                FitnessScore = min 1.0 (rule.FitnessScore + 0.15)
                                GenerationNumber = session.CurrentGeneration + 1
                                ParentRules = [rule.RuleId]
                                EvolutionHistory = $"Optimized by {agent.Name}" :: rule.EvolutionHistory
                        }
                        rules.Add(optimizedRule)
                
                return rules |> Seq.toList
            }
        
        /// Generate new rule body based on grammar and agent capabilities
        member private this.GenerateNewRuleBody(grammar: ExistingGrammar, agent: UniversityAgent) : string =
            let capabilities = String.Join(" | ", agent.Capabilities)
            match grammar.Id with
            | "MiniQuery" ->
                $"enhanced_query = \"find\" , ws , {capabilities.Replace(" ", "_").ToLowerInvariant()} , ws , \"in\" , ws , target ;"
            | "Wolfram" ->
                $"enhanced_function = \"{agent.Specialization.Replace(" ", "")}Function\" , \"[\" , argument_list , \"]\" ;"
            | "RFC3986_URI" ->
                $"enhanced_uri = scheme , \":\" , hier_part , [ \"?\" , {agent.Name.Replace(" ", "_").ToLowerInvariant()}_query ] ;"
            | _ ->
                $"evolved_rule = {agent.EvolutionRole.ToString().ToLowerInvariant()}_pattern , ws , expression ;"
        
        /// Mutate existing rule body
        member private this.MutateRuleBody(ruleBody: string, agent: UniversityAgent) : string =
            let mutations = [
                ruleBody.Replace("=", "::=") // Change assignment operator
                ruleBody + $" | {agent.Name.Replace(" ", "_").ToLowerInvariant()}_extension" // Add extension
                ruleBody.Replace(";", $" , {agent.EvolutionRole.ToString().ToLowerInvariant()}_modifier ;") // Add modifier
            ]
            mutations.[Random().Next(mutations.Length)]
        
        /// Synthesize two rule bodies
        member private this.SynthesizeRuleBodies(body1: string, body2: string, agent: UniversityAgent) : string =
            $"({body1.TrimEnd(';')}) | ({body2.TrimEnd(';')}) | {agent.Name.Replace(" ", "_").ToLowerInvariant()}_synthesis ;"
        
        /// Optimize rule body for performance
        member private this.OptimizeRuleBody(ruleBody: string, agent: UniversityAgent) : string =
            ruleBody
                .Replace(" , ", ",") // Remove spaces for efficiency
                .Replace("{ ", "{") // Compact braces
                .Replace(" }", "}") // Compact braces
                + $" (* Optimized by {agent.Name} *)"
        
        /// Evaluate rules for fitness
        member private this.EvaluateRules(rules: EvolvedGrammarRule list, session: GrammarEvolutionSession) : EvolvedGrammarRule list =
            rules
            |> List.map (fun rule ->
                // Calculate fitness based on various factors
                let complexityScore = min 1.0 (float rule.RuleBody.Length / 200.0)
                let innovationScore = if rule.ParentRules.IsEmpty then 0.3 else 0.1
                let agentScore = if session.ParticipatingAgents |> List.exists (fun a -> a.Name = rule.CreatedBy) then 0.2 else 0.0
                let grammarAffinityScore = 
                    session.ParticipatingAgents 
                    |> List.tryFind (fun a -> a.Name = rule.CreatedBy)
                    |> Option.map (fun a -> if a.GrammarAffinity |> List.contains rule.OriginalGrammar then 0.2 else 0.0)
                    |> Option.defaultValue 0.0
                
                let newFitness = min 1.0 (complexityScore + innovationScore + agentScore + grammarAffinityScore)
                { rule with FitnessScore = newFitness })
        
        /// Select best rules from evaluated set
        member private this.SelectBestRules(rules: EvolvedGrammarRule list, count: int) : EvolvedGrammarRule list =
            rules
            |> List.sortByDescending (fun r -> r.FitnessScore)
            |> List.take (min count rules.Length)
        
        /// Update performance metrics
        member private this.UpdatePerformanceMetrics(session: GrammarEvolutionSession, newRules: EvolvedGrammarRule list) : Map<string, float> =
            let avgFitness = if newRules.IsEmpty then 0.5 else newRules |> List.averageBy (fun r -> r.FitnessScore)
            let diversity = float (newRules |> List.map (fun r -> r.OriginalGrammar) |> List.distinct |> List.length) / float session.BaseGrammars.Length
            let innovation = float (newRules |> List.filter (fun r -> r.ParentRules.IsEmpty) |> List.length) / float newRules.Length
            let coordination = float (newRules |> List.filter (fun r -> r.ParentRules.Length > 1) |> List.length) / float newRules.Length
            
            Map.ofList [
                ("grammar_diversity", diversity)
                ("rule_effectiveness", avgFitness)
                ("team_coordination", coordination)
                ("innovation_rate", innovation)
            ]
        
        /// Generate evolved grammar file
        member private this.GenerateEvolvedGrammar(session: GrammarEvolutionSession) : string =
            let header = [
                $"# Evolved Grammar - {session.TeamName}"
                $"# Generation: {session.CurrentGeneration}"
                $"# Evolution Goal: {session.EvolutionGoal}"
                $"# Session: {session.SessionId}"
                ""
                "meta {"
                $"  name: \"Evolved_{session.TeamName.Replace(" ", "_")}\""
                $"  version: \"Gen{session.CurrentGeneration}.0\""
                $"  source: \"autonomous_evolution\""
                $"  language: \"EBNF\""
                $"  created: \"{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss}\""
                $"  description: \"Autonomously evolved grammar by {session.TeamName}\""
                $"  evolution_session: \"{session.SessionId}\""
                $"  participating_agents: [{String.Join(", ", session.ParticipatingAgents |> List.map (fun a -> $"\"{a.Name}\""))}]"
                "  tags: [\"evolved\", \"autonomous\", \"team_generated\"]"
                "}"
                ""
                "grammar {"
                "  LANG(\"EBNF\") {"
            ]
            
            let rules = 
                session.EvolvedRules
                |> List.sortByDescending (fun r -> r.FitnessScore)
                |> List.take (min 20 session.EvolvedRules.Length)
                |> List.map (fun rule ->
                    $"    (* {rule.RuleId} - Fitness: {rule.FitnessScore:F3} - Created by: {rule.CreatedBy} *)")
                |> List.map (fun comment -> [comment; $"    {session.EvolvedRules |> List.find (fun r -> comment.Contains(r.RuleId)) |> fun r -> r.RuleBody}"; ""])
                |> List.concat
            
            let footer = [
                "  }"
                "}"
                ""
                $"(* Evolution Statistics:"
                $"   Total Rules: {session.EvolvedRules.Length}"
                $"   Best Fitness: {session.EvolvedRules |> List.map (fun r -> r.FitnessScore) |> List.max}"
                $"   Avg Fitness: {session.EvolvedRules |> List.averageBy (fun r -> r.FitnessScore)}"
                $"   Innovation Rate: {session.PerformanceMetrics.["innovation_rate"]:F3}"
                $"   Team Coordination: {session.PerformanceMetrics.["team_coordination"]:F3}"
                "*)"
            ]
            
            String.Join("\n", header @ rules @ footer)
        
        /// Save evolved grammar to file
        member private this.SaveEvolvedGrammar(outputPath: string, grammarContent: string) =
            try
                let directory = Path.GetDirectoryName(outputPath)
                if not (Directory.Exists(directory)) then
                    Directory.CreateDirectory(directory) |> ignore
                
                File.WriteAllText(outputPath, grammarContent)
                logger.LogInformation("💾 Saved evolved grammar to {OutputPath}", outputPath)
            with
            | ex ->
                logger.LogError(ex, "Error saving evolved grammar to {OutputPath}", outputPath)
        
        /// Update 3D visualization with evolution progress
        member private this.Update3DVisualization(session: GrammarEvolutionSession) =
            // Update agent performance based on their contributions
            session.ParticipatingAgents |> List.iter (fun agent ->
                let agentRules = session.EvolvedRules |> List.filter (fun r -> r.CreatedBy = agent.Name)
                let avgFitness = if agentRules.IsEmpty then 0.5 else agentRules |> List.averageBy (fun r -> r.FitnessScore)
                
                // This would update the 3D agent representation
                logger.LogDebug("🎯 Agent {AgentName} performance: {Performance:F3} ({RuleCount} rules)",
                               agent.Name, avgFitness, agentRules.Length)
            )
        
        /// Get evolution status for all active sessions
        member this.GetEvolutionStatus() : Map<string, GrammarEvolutionSession> =
            activeSessions
            |> Seq.map (fun kvp -> kvp.Key, kvp.Value)
            |> Map.ofSeq
        
        /// Stop evolution session
        member this.StopEvolutionSession(sessionId: string) =
            match activeSessions.TryGetValue(sessionId) with
            | true, session ->
                let updatedSession = { session with IsActive = false }
                activeSessions.[sessionId] <- updatedSession
                logger.LogInformation("⏹️ Stopped evolution session {SessionId}", sessionId)
            | false, _ ->
                logger.LogWarning("Session {SessionId} not found for stopping", sessionId)
