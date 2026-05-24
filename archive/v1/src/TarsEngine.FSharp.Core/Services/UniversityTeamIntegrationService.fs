namespace TarsEngine.FSharp.Core.Services

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Evolution.TeamGrammarEvolution
open TarsEngine.FSharp.Core.Services.Agent3DIntegrationService

/// Service to integrate existing university teams with grammar evolution and 3D visualization
module UniversityTeamIntegrationService =
    
    /// Integration configuration
    type IntegrationConfig = {
        UniversityPath: string
        GrammarPath: string
        EvolutionOutputPath: string
        AutoEvolutionEnabled: bool
        EvolutionInterval: TimeSpan
        MaxGenerations: int
    }
    
    /// Team integration status
    type TeamIntegrationStatus = {
        TeamName: string
        IsLoaded: bool
        AgentCount: int
        GrammarCount: int
        EvolutionSessionId: string option
        CurrentGeneration: int
        LastEvolutionTime: DateTime option
        PerformanceMetrics: Map<string, float>
        IsActive: bool
    }
    
    /// University team integration service
    type UniversityTeamIntegrationService(
        logger: ILogger<UniversityTeamIntegrationService>,
        evolutionEngine: TeamGrammarEvolutionEngine,
        integrationService: Agent3DIntegrationService) =
        
        let mutable integrationConfig = {
            UniversityPath = ".tars/university"
            GrammarPath = ".tars/grammars"
            EvolutionOutputPath = ".tars/evolution"
            AutoEvolutionEnabled = true
            EvolutionInterval = TimeSpan.FromMinutes(5.0)
            MaxGenerations = 50
        }
        
        let teamStatuses = System.Collections.Concurrent.ConcurrentDictionary<string, TeamIntegrationStatus>()
        
        /// Initialize integration with existing teams and grammars
        member this.InitializeIntegration(config: IntegrationConfig) : Async<bool> =
            async {
                try
                    integrationConfig <- config
                    logger.LogInformation("🎓 Initializing university team integration...")
                    
                    // Load existing grammars
                    let grammars = evolutionEngine.LoadExistingGrammars(config.GrammarPath)
                    logger.LogInformation("📚 Loaded {GrammarCount} existing grammars", grammars.Length)
                    
                    // Discover and load university teams
                    let! teamLoadResults = this.DiscoverAndLoadTeams(config.UniversityPath, grammars)
                    
                    let successfulTeams = teamLoadResults |> List.filter (fun (_, success) -> success) |> List.length
                    logger.LogInformation("🎯 Successfully integrated {SuccessfulTeams}/{TotalTeams} university teams",
                                         successfulTeams, teamLoadResults.Length)
                    
                    // Start auto-evolution if enabled
                    if config.AutoEvolutionEnabled then
                        this.StartAutoEvolution()
                    
                    return successfulTeams > 0
                with
                | ex ->
                    logger.LogError(ex, "Error during university team integration initialization")
                    return false
            }
        
        /// Discover and load teams from university directory
        member private this.DiscoverAndLoadTeams(universityPath: string, grammars: ExistingGrammar list) : Async<(string * bool) list> =
            async {
                try
                    if not (Directory.Exists(universityPath)) then
                        logger.LogWarning("University directory not found: {UniversityPath}", universityPath)
                        return []
                    
                    let teamConfigPath = Path.Combine(universityPath, "team-config.json")
                    if File.Exists(teamConfigPath) then
                        // Load the main university team
                        let! result = this.LoadAndIntegrateTeam("University Research Team", teamConfigPath, grammars)
                        return [("University Research Team", result)]
                    else
                        // Look for individual team directories
                        let teamDirectories = Directory.GetDirectories(universityPath)
                        let loadTasks = 
                            teamDirectories
                            |> Array.map (fun teamDir ->
                                async {
                                    let teamName = Path.GetFileName(teamDir)
                                    let configPath = Path.Combine(teamDir, "team-config.json")
                                    if File.Exists(configPath) then
                                        let! result = this.LoadAndIntegrateTeam(teamName, configPath, grammars)
                                        return (teamName, result)
                                    else
                                        logger.LogWarning("No team-config.json found in {TeamDir}", teamDir)
                                        return (teamName, false)
                                })
                        
                        let! results = Async.Parallel(loadTasks)
                        return results |> Array.toList
                with
                | ex ->
                    logger.LogError(ex, "Error discovering teams in {UniversityPath}", universityPath)
                    return []
            }
        
        /// Load and integrate a single team
        member private this.LoadAndIntegrateTeam(teamName: string, configPath: string, grammars: ExistingGrammar list) : Async<bool> =
            async {
                try
                    logger.LogInformation("🎓 Loading team: {TeamName} from {ConfigPath}", teamName, configPath)
                    
                    // Load team agents
                    let agents = evolutionEngine.LoadUniversityTeam(configPath)
                    if agents.IsEmpty then
                        logger.LogWarning("No agents found for team {TeamName}", teamName)
                        return false
                    
                    // Filter grammars based on agent affinities
                    let relevantGrammars = 
                        grammars
                        |> List.filter (fun grammar ->
                            agents |> List.exists (fun agent -> agent.GrammarAffinity |> List.contains grammar.Id))
                    
                    if relevantGrammars.IsEmpty then
                        logger.LogInformation("Using all grammars for team {TeamName} (no specific affinities)", teamName)
                    
                    let grammarsToUse = if relevantGrammars.IsEmpty then grammars else relevantGrammars
                    
                    // Start evolution session
                    let evolutionGoal = $"Evolve grammars for {teamName} specializing in {String.Join(", ", agents |> List.map (fun a -> a.Specialization) |> List.distinct)}"
                    let sessionId = evolutionEngine.StartEvolutionSession(teamName, agents, grammarsToUse, evolutionGoal)
                    
                    // Create team status
                    let teamStatus = {
                        TeamName = teamName
                        IsLoaded = true
                        AgentCount = agents.Length
                        GrammarCount = grammarsToUse.Length
                        EvolutionSessionId = Some sessionId
                        CurrentGeneration = 0
                        LastEvolutionTime = None
                        PerformanceMetrics = Map.ofList [
                            ("team_cohesion", 0.7)
                            ("grammar_mastery", 0.6)
                            ("innovation_potential", 0.8)
                        ]
                        IsActive = true
                    }
                    
                    teamStatuses.[teamName] <- teamStatus
                    
                    logger.LogInformation("✅ Successfully integrated team {TeamName} with {AgentCount} agents and {GrammarCount} grammars",
                                         teamName, agents.Length, grammarsToUse.Length)
                    
                    return true
                with
                | ex ->
                    logger.LogError(ex, "Error loading team {TeamName} from {ConfigPath}", teamName, configPath)
                    return false
            }
        
        /// Start automatic evolution for all active teams
        member private this.StartAutoEvolution() =
            logger.LogInformation("🔄 Starting automatic evolution with {Interval} interval", integrationConfig.EvolutionInterval)
            
            let evolutionTimer = new System.Timers.Timer(integrationConfig.EvolutionInterval.TotalMilliseconds)
            evolutionTimer.Elapsed.Add(fun _ ->
                async {
                    try
                        let! _ = this.RunEvolutionCycle()
                        ()
                    with
                    | ex -> logger.LogError(ex, "Error during automatic evolution cycle")
                } |> Async.Start
            )
            evolutionTimer.Start()
        
        /// Run one evolution cycle for all active teams
        member this.RunEvolutionCycle() : Async<Map<string, bool>> =
            async {
                logger.LogInformation("🧬 Running evolution cycle for {TeamCount} active teams", teamStatuses.Count)
                
                let results = System.Collections.Concurrent.ConcurrentDictionary<string, bool>()
                
                let evolutionTasks = 
                    teamStatuses.Values
                    |> Seq.filter (fun status -> status.IsActive && status.EvolutionSessionId.IsSome)
                    |> Seq.map (fun status ->
                        async {
                            try
                                let sessionId = status.EvolutionSessionId.Value
                                let! success = evolutionEngine.EvolveGeneration(sessionId)
                                
                                if success then
                                    // Update team status
                                    let updatedStatus = {
                                        status with
                                            CurrentGeneration = status.CurrentGeneration + 1
                                            LastEvolutionTime = Some DateTime.UtcNow
                                    }
                                    teamStatuses.[status.TeamName] <- updatedStatus
                                    
                                    logger.LogInformation("🎯 Team {TeamName} evolved to generation {Generation}",
                                                         status.TeamName, updatedStatus.CurrentGeneration)
                                
                                results.[status.TeamName] <- success
                            with
                            | ex ->
                                logger.LogError(ex, "Error evolving team {TeamName}", status.TeamName)
                                results.[status.TeamName] <- false
                        })
                
                do! Async.Parallel(evolutionTasks) |> Async.Ignore
                
                return results |> Seq.map (fun kvp -> kvp.Key, kvp.Value) |> Map.ofSeq
            }
        
        /// Get integration status for all teams
        member this.GetIntegrationStatus() : Map<string, TeamIntegrationStatus> =
            teamStatuses
            |> Seq.map (fun kvp -> kvp.Key, kvp.Value)
            |> Map.ofSeq
        
        /// Get detailed team information
        member this.GetTeamDetails(teamName: string) : TeamIntegrationStatus option =
            match teamStatuses.TryGetValue(teamName) with
            | true, status -> Some status
            | false, _ -> None
        
        /// Manually trigger evolution for a specific team
        member this.TriggerTeamEvolution(teamName: string) : Async<bool> =
            async {
                match teamStatuses.TryGetValue(teamName) with
                | true, status when status.IsActive && status.EvolutionSessionId.IsSome ->
                    try
                        let sessionId = status.EvolutionSessionId.Value
                        let! success = evolutionEngine.EvolveGeneration(sessionId)
                        
                        if success then
                            let updatedStatus = {
                                status with
                                    CurrentGeneration = status.CurrentGeneration + 1
                                    LastEvolutionTime = Some DateTime.UtcNow
                            }
                            teamStatuses.[teamName] <- updatedStatus
                            
                            logger.LogInformation("🚀 Manually triggered evolution for team {TeamName} - Generation {Generation}",
                                                 teamName, updatedStatus.CurrentGeneration)
                        
                        return success
                    with
                    | ex ->
                        logger.LogError(ex, "Error during manual evolution trigger for team {TeamName}", teamName)
                        return false
                | _ ->
                    logger.LogWarning("Team {TeamName} not found or not active for evolution", teamName)
                    return false
            }
        
        /// Stop evolution for a specific team
        member this.StopTeamEvolution(teamName: string) =
            match teamStatuses.TryGetValue(teamName) with
            | true, status ->
                if status.EvolutionSessionId.IsSome then
                    evolutionEngine.StopEvolutionSession(status.EvolutionSessionId.Value)
                
                let updatedStatus = { status with IsActive = false }
                teamStatuses.[teamName] <- updatedStatus
                
                logger.LogInformation("⏹️ Stopped evolution for team {TeamName}", teamName)
            | false, _ ->
                logger.LogWarning("Team {TeamName} not found for evolution stop", teamName)
        
        /// Generate comprehensive integration report
        member this.GenerateIntegrationReport() : string =
            let activeTeams = teamStatuses.Values |> Seq.filter (fun s -> s.IsActive) |> Seq.toList
            let totalGenerations = activeTeams |> List.sumBy (fun s -> s.CurrentGeneration)
            let avgPerformance = 
                if activeTeams.IsEmpty then 0.0
                else
                    activeTeams 
                    |> List.collect (fun s -> s.PerformanceMetrics |> Map.values |> Seq.toList)
                    |> List.average
            
            let report = [
                "# TARS University Team Integration Report"
                $"Generated: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC"
                ""
                "## Integration Summary"
                $"- **Active Teams**: {activeTeams.Length}"
                $"- **Total Agents**: {activeTeams |> List.sumBy (fun s -> s.AgentCount)}"
                $"- **Total Grammars**: {activeTeams |> List.sumBy (fun s -> s.GrammarCount)}"
                $"- **Total Generations Evolved**: {totalGenerations}"
                $"- **Average Performance**: {avgPerformance:F3}"
                ""
                "## Team Details"
            ]
            
            let teamDetails = 
                activeTeams
                |> List.map (fun team ->
                    let lastEvolution = 
                        team.LastEvolutionTime 
                        |> Option.map (fun dt -> dt.ToString("yyyy-MM-dd HH:mm:ss"))
                        |> Option.defaultValue "Never"
                    
                    [
                        $"### {team.TeamName}"
                        $"- **Agents**: {team.AgentCount}"
                        $"- **Grammars**: {team.GrammarCount}"
                        $"- **Current Generation**: {team.CurrentGeneration}"
                        $"- **Last Evolution**: {lastEvolution}"
                        $"- **Status**: {if team.IsActive then "Active" else "Inactive"}"
                        ""
                    ])
                |> List.concat
            
            let evolutionStatus = [
                "## Evolution Configuration"
                $"- **Auto Evolution**: {integrationConfig.AutoEvolutionEnabled}"
                $"- **Evolution Interval**: {integrationConfig.EvolutionInterval}"
                $"- **Max Generations**: {integrationConfig.MaxGenerations}"
                $"- **Output Path**: {integrationConfig.EvolutionOutputPath}"
                ""
                "## Next Steps"
                "1. Monitor team evolution progress"
                "2. Review generated grammars in evolution output directory"
                "3. Analyze performance metrics for optimization opportunities"
                "4. Consider adjusting evolution parameters based on results"
            ]
            
            String.Join("\n", report @ teamDetails @ evolutionStatus)
        
        /// Export evolved grammars for a team
        member this.ExportTeamGrammars(teamName: string, outputPath: string) : Async<bool> =
            async {
                try
                    match teamStatuses.TryGetValue(teamName) with
                    | true, status when status.EvolutionSessionId.IsSome ->
                        let evolutionStatus = evolutionEngine.GetEvolutionStatus()
                        match evolutionStatus.TryFind(status.EvolutionSessionId.Value) with
                        | Some session ->
                            let grammarContent = $"# Exported Grammars for {teamName}\n# Generation: {session.CurrentGeneration}\n\n"
                            
                            if not (Directory.Exists(Path.GetDirectoryName(outputPath))) then
                                Directory.CreateDirectory(Path.GetDirectoryName(outputPath)) |> ignore
                            
                            File.WriteAllText(outputPath, grammarContent)
                            
                            logger.LogInformation("📤 Exported grammars for team {TeamName} to {OutputPath}", teamName, outputPath)
                            return true
                        | None ->
                            logger.LogWarning("Evolution session not found for team {TeamName}", teamName)
                            return false
                    | _ ->
                        logger.LogWarning("Team {TeamName} not found or has no evolution session", teamName)
                        return false
                with
                | ex ->
                    logger.LogError(ex, "Error exporting grammars for team {TeamName}", teamName)
                    return false
            }
