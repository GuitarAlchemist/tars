namespace TarsEngine.FSharp.Core.Evolution

open System
open System.IO
open System.Collections.Concurrent
open System.Text.Json
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Types
open TarsEngine.FSharp.Core.Metascript.FractalGrammarMetascripts
open TarsEngine.FSharp.Agents.AgentTeams

/// Evolutionary grammar generation system for autonomous team evolution
module EvolutionaryGrammarSystem =
    
    /// Existing team configuration from .tars directory
    type ExistingTeamConfig = {
        TeamName: string
        Institution: string
        EstablishedDate: DateTime
        Agents: ExistingAgent list
        ResearchAreas: string list
        AcademicStandards: Map<string, obj>
    }
    
    and ExistingAgent = {
        Name: string
        Specialization: string
        Capabilities: string list
        OutputFormats: string list
    }
    
    /// Grammar evolution parameters
    type GrammarEvolutionConfig = {
        MutationRate: float
        CrossoverRate: float
        SelectionPressure: float
        PopulationSize: int
        GenerationLimit: int
        FitnessThreshold: float
        EvolutionStrategy: EvolutionStrategy
    }
    
    and EvolutionStrategy =
        | GeneticAlgorithm
        | DifferentialEvolution
        | ParticleSwarmOptimization
        | HybridEvolution
    
    /// Evolved grammar gene
    type GrammarGene = {
        RulePattern: string
        ActionSequence: string list
        ConditionLogic: string
        FitnessScore: float
        GenerationCreated: int
        ParentGenes: string list
    }
    
    /// Grammar chromosome (collection of genes)
    type GrammarChromosome = {
        Id: string
        Genes: GrammarGene list
        OverallFitness: float
        TeamOrigin: string
        EvolutionHistory: string list
        SpecializationFocus: string
    }
    
    /// Team evolution state
    type TeamEvolutionState = {
        TeamId: string
        OriginalConfig: ExistingTeamConfig
        CurrentGeneration: int
        BestChromosome: GrammarChromosome option
        PopulationPool: GrammarChromosome list
        EvolutionMetrics: Map<string, float>
        LastEvolutionTime: DateTime
        IsActive: bool
    }
    
    /// Evolutionary grammar system
    type EvolutionaryGrammarSystem(logger: ILogger<EvolutionaryGrammarSystem>) =
        
        let activeEvolutions = ConcurrentDictionary<string, TeamEvolutionState>()
        let grammarTemplates = ResizeArray<GrammarChromosome>()
        
        /// Load existing team configuration from .tars directory
        member this.LoadExistingTeam(teamPath: string) : ExistingTeamConfig option =
            try
                let configPath = Path.Combine(teamPath, "team-config.json")
                if File.Exists(configPath) then
                    let jsonContent = File.ReadAllText(configPath)
                    let jsonDoc = JsonDocument.Parse(jsonContent)
                    let root = jsonDoc.RootElement
                    
                    let agents = 
                        root.GetProperty("agents").EnumerateArray()
                        |> Seq.map (fun agentElement ->
                            {
                                Name = agentElement.GetProperty("name").GetString()
                                Specialization = agentElement.GetProperty("specialization").GetString()
                                Capabilities = 
                                    agentElement.GetProperty("capabilities").EnumerateArray()
                                    |> Seq.map (fun cap -> cap.GetString())
                                    |> Seq.toList
                                OutputFormats = 
                                    agentElement.GetProperty("output_formats").EnumerateArray()
                                    |> Seq.map (fun fmt -> fmt.GetString())
                                    |> Seq.toList
                            })
                        |> Seq.toList
                    
                    let researchAreas = 
                        root.GetProperty("research_areas").EnumerateArray()
                        |> Seq.map (fun area -> area.GetString())
                        |> Seq.toList
                    
                    Some {
                        TeamName = root.GetProperty("team_name").GetString()
                        Institution = root.GetProperty("institution").GetString()
                        EstablishedDate = DateTime.Parse(root.GetProperty("established_date").GetString())
                        Agents = agents
                        ResearchAreas = researchAreas
                        AcademicStandards = Map.empty // Simplified for now
                    }
                else
                    None
            with
            | ex ->
                logger.LogError(ex, "Error loading team configuration from {TeamPath}", teamPath)
                None
        
        /// Generate initial grammar chromosome from team configuration
        member this.GenerateInitialChromosome(teamConfig: ExistingTeamConfig) : GrammarChromosome =
            let teamId = teamConfig.TeamName.Replace(" ", "_").ToLowerInvariant()
            
            // Generate genes based on agent capabilities
            let genes = 
                teamConfig.Agents
                |> List.mapi (fun i agent ->
                    let capabilities = String.Join(", ", agent.Capabilities)
                    {
                        RulePattern = $"WHEN {agent.Specialization.ToUpperInvariant()}_TASK_REQUIRED"
                        ActionSequence = [
                            $"ACTIVATE_AGENT \"{agent.Name}\""
                            $"APPLY_CAPABILITIES [{capabilities}]"
                            $"GENERATE_OUTPUT {String.Join("|", agent.OutputFormats)}"
                        ]
                        ConditionLogic = $"task_type == \"{agent.Specialization}\" AND agent_available"
                        FitnessScore = 0.5 + (float i * 0.1) // Initial fitness
                        GenerationCreated = 0
                        ParentGenes = []
                    })
            
            // Add collaboration genes
            let collaborationGenes = [
                {
                    RulePattern = "WHEN INTERDISCIPLINARY_COLLABORATION_NEEDED"
                    ActionSequence = [
                        "FORM_RESEARCH_TEAM"
                        "ESTABLISH_COMMUNICATION_PROTOCOLS"
                        "COORDINATE_RESEARCH_ACTIVITIES"
                        "SYNTHESIZE_RESULTS"
                    ]
                    ConditionLogic = "research_complexity > 0.7 AND multiple_disciplines_required"
                    FitnessScore = 0.8
                    GenerationCreated = 0
                    ParentGenes = []
                }
                {
                    RulePattern = "WHEN QUALITY_ASSURANCE_REQUIRED"
                    ActionSequence = [
                        "INITIATE_PEER_REVIEW"
                        "VALIDATE_METHODOLOGY"
                        "CHECK_ETHICS_COMPLIANCE"
                        "ENSURE_ACADEMIC_STANDARDS"
                    ]
                    ConditionLogic = "output_ready AND quality_check_needed"
                    FitnessScore = 0.9
                    GenerationCreated = 0
                    ParentGenes = []
                }
            ]
            
            {
                Id = $"{teamId}_gen0_chr1"
                Genes = genes @ collaborationGenes
                OverallFitness = 0.6
                TeamOrigin = teamConfig.TeamName
                EvolutionHistory = ["Initial generation from team config"]
                SpecializationFocus = String.Join(", ", teamConfig.ResearchAreas)
            }
        
        /// Start evolution for an existing team
        member this.StartTeamEvolution(teamPath: string, evolutionConfig: GrammarEvolutionConfig) : string option =
            match this.LoadExistingTeam(teamPath) with
            | Some teamConfig ->
                let teamId = teamConfig.TeamName.Replace(" ", "_").ToLowerInvariant()
                
                // Generate initial population
                let initialChromosome = this.GenerateInitialChromosome(teamConfig)
                let initialPopulation = this.GenerateInitialPopulation(initialChromosome, evolutionConfig.PopulationSize)
                
                let evolutionState = {
                    TeamId = teamId
                    OriginalConfig = teamConfig
                    CurrentGeneration = 0
                    BestChromosome = Some initialChromosome
                    PopulationPool = initialPopulation
                    EvolutionMetrics = Map.ofList [
                        ("avg_fitness", 0.6)
                        ("diversity_score", 0.8)
                        ("innovation_rate", 0.3)
                    ]
                    LastEvolutionTime = DateTime.UtcNow
                    IsActive = true
                }
                
                activeEvolutions.[teamId] <- evolutionState
                
                logger.LogInformation("🧬 Started evolution for team {TeamName} with {AgentCount} agents",
                                     teamConfig.TeamName, teamConfig.Agents.Length)
                
                Some teamId
            | None ->
                logger.LogWarning("⚠️ Could not load team configuration from {TeamPath}", teamPath)
                None
        
        /// Generate initial population from base chromosome
        member private this.GenerateInitialPopulation(baseChromosome: GrammarChromosome, populationSize: int) : GrammarChromosome list =
            [1..populationSize]
            |> List.map (fun i ->
                let mutatedGenes = 
                    baseChromosome.Genes
                    |> List.map (fun gene ->
                        // Add slight mutations for diversity
                        { gene with 
                            FitnessScore = gene.FitnessScore + (Random().NextDouble() - 0.5) * 0.2
                            RulePattern = if Random().NextDouble() < 0.1 then 
                                            gene.RulePattern + $"_VARIANT_{i}" 
                                          else gene.RulePattern
                        })
                
                { baseChromosome with
                    Id = $"{baseChromosome.TeamOrigin.Replace(" ", "_").ToLowerInvariant()}_gen0_chr{i}"
                    Genes = mutatedGenes
                    OverallFitness = baseChromosome.OverallFitness + (Random().NextDouble() - 0.5) * 0.1
                })
        
        /// Evolve team grammar for one generation
        member this.EvolveTeamGeneration(teamId: string) : Async<bool> =
            async {
                match activeEvolutions.TryGetValue(teamId) with
                | true, evolutionState when evolutionState.IsActive ->
                    try
                        logger.LogInformation("🧬 Evolving generation {Generation} for team {TeamId}",
                                             evolutionState.CurrentGeneration + 1, teamId)
                        
                        // Selection: Choose best chromosomes
                        let selectedParents = this.SelectParents(evolutionState.PopulationPool)
                        
                        // Crossover: Create offspring
                        let offspring = this.PerformCrossover(selectedParents)
                        
                        // Mutation: Introduce variations
                        let mutatedOffspring = this.PerformMutation(offspring)
                        
                        // Evaluation: Calculate fitness
                        let evaluatedOffspring = this.EvaluateFitness(mutatedOffspring, evolutionState.OriginalConfig)
                        
                        // Create new population
                        let newPopulation = this.CreateNewPopulation(evolutionState.PopulationPool, evaluatedOffspring)
                        
                        // Find best chromosome
                        let bestChromosome = 
                            newPopulation 
                            |> List.maxBy (fun chr -> chr.OverallFitness)
                        
                        // Update evolution state
                        let updatedState = {
                            evolutionState with
                                CurrentGeneration = evolutionState.CurrentGeneration + 1
                                BestChromosome = Some bestChromosome
                                PopulationPool = newPopulation
                                LastEvolutionTime = DateTime.UtcNow
                                EvolutionMetrics = Map.ofList [
                                    ("avg_fitness", newPopulation |> List.averageBy (fun chr -> chr.OverallFitness))
                                    ("best_fitness", bestChromosome.OverallFitness)
                                    ("diversity_score", this.CalculateDiversityScore(newPopulation))
                                ]
                        }
                        
                        activeEvolutions.[teamId] <- updatedState
                        
                        // Generate evolved metascript
                        let evolvedMetascript = this.GenerateMetascriptFromChromosome(bestChromosome)
                        let outputPath = $".tars/evolution/{teamId}_gen{updatedState.CurrentGeneration}.trsx"
                        this.SaveEvolvedMetascript(outputPath, evolvedMetascript)
                        
                        logger.LogInformation("✅ Evolution generation {Generation} complete. Best fitness: {Fitness:F3}",
                                             updatedState.CurrentGeneration, bestChromosome.OverallFitness)
                        
                        return true
                    with
                    | ex ->
                        logger.LogError(ex, "Error during evolution for team {TeamId}", teamId)
                        return false
                | _ ->
                    logger.LogWarning("Team {TeamId} not found or not active for evolution", teamId)
                    return false
            }
        
        /// Select parent chromosomes for reproduction
        member private this.SelectParents(population: GrammarChromosome list) : GrammarChromosome list =
            population
            |> List.sortByDescending (fun chr -> chr.OverallFitness)
            |> List.take (min 4 population.Length) // Select top 4 as parents
        
        /// Perform crossover between parent chromosomes
        member private this.PerformCrossover(parents: GrammarChromosome list) : GrammarChromosome list =
            let random = Random()
            
            [for i in 0..1 do
                if parents.Length >= 2 then
                    let parent1 = parents.[random.Next(parents.Length)]
                    let parent2 = parents.[random.Next(parents.Length)]
                    
                    // Crossover genes
                    let crossoverPoint = random.Next(min parent1.Genes.Length parent2.Genes.Length)
                    let newGenes = 
                        (parent1.Genes |> List.take crossoverPoint) @
                        (parent2.Genes |> List.skip crossoverPoint)
                    
                    yield {
                        Id = $"crossover_{parent1.Id}_{parent2.Id}_{i}"
                        Genes = newGenes
                        OverallFitness = (parent1.OverallFitness + parent2.OverallFitness) / 2.0
                        TeamOrigin = parent1.TeamOrigin
                        EvolutionHistory = $"Crossover of {parent1.Id} and {parent2.Id}" :: parent1.EvolutionHistory
                        SpecializationFocus = parent1.SpecializationFocus
                    }]
        
        /// Perform mutation on chromosomes
        member private this.PerformMutation(chromosomes: GrammarChromosome list) : GrammarChromosome list =
            let random = Random()
            let mutationRate = 0.1
            
            chromosomes
            |> List.map (fun chromosome ->
                if random.NextDouble() < mutationRate then
                    let mutatedGenes = 
                        chromosome.Genes
                        |> List.map (fun gene ->
                            if random.NextDouble() < 0.3 then
                                // Mutate action sequence
                                let newAction = $"EVOLVED_ACTION_{random.Next(1000)}"
                                { gene with 
                                    ActionSequence = newAction :: gene.ActionSequence
                                    GenerationCreated = chromosome.EvolutionHistory.Length
                                }
                            else
                                gene)
                    
                    { chromosome with
                        Id = chromosome.Id + "_mutated"
                        Genes = mutatedGenes
                        EvolutionHistory = "Mutation applied" :: chromosome.EvolutionHistory
                    }
                else
                    chromosome)
        
        /// Evaluate fitness of chromosomes
        member private this.EvaluateFitness(chromosomes: GrammarChromosome list, teamConfig: ExistingTeamConfig) : GrammarChromosome list =
            chromosomes
            |> List.map (fun chromosome ->
                // Calculate fitness based on various factors
                let geneComplexity = float chromosome.Genes.Length / 10.0
                let specialization = if chromosome.SpecializationFocus.Contains("AI") then 0.2 else 0.0
                let collaboration = if chromosome.Genes |> List.exists (fun g -> g.RulePattern.Contains("COLLABORATION")) then 0.3 else 0.0
                let innovation = float chromosome.EvolutionHistory.Length * 0.1
                
                let newFitness = min 1.0 (geneComplexity + specialization + collaboration + innovation)
                
                { chromosome with OverallFitness = newFitness })
        
        /// Create new population from parents and offspring
        member private this.CreateNewPopulation(parents: GrammarChromosome list, offspring: GrammarChromosome list) : GrammarChromosome list =
            (parents @ offspring)
            |> List.sortByDescending (fun chr -> chr.OverallFitness)
            |> List.take (min 10 (parents.Length + offspring.Length)) // Keep top 10
        
        /// Calculate diversity score of population
        member private this.CalculateDiversityScore(population: GrammarChromosome list) : float =
            if population.Length <= 1 then 0.0
            else
                let uniquePatterns = 
                    population 
                    |> List.collect (fun chr -> chr.Genes |> List.map (fun g -> g.RulePattern))
                    |> List.distinct
                    |> List.length
                
                let totalPatterns = population |> List.sumBy (fun chr -> chr.Genes.Length)
                float uniquePatterns / float totalPatterns
        
        /// Generate metascript from evolved chromosome
        member private this.GenerateMetascriptFromChromosome(chromosome: GrammarChromosome) : string =
            let header = [
                $"# Evolved Grammar for {chromosome.TeamOrigin}"
                $"# Generation: {chromosome.EvolutionHistory.Length}"
                $"# Fitness: {chromosome.OverallFitness:F3}"
                $"# Specialization: {chromosome.SpecializationFocus}"
                ""
                "meta {"
                $"  name: \"Evolved Grammar - {chromosome.TeamOrigin}\""
                $"  version: \"Gen{chromosome.EvolutionHistory.Length}.0\""
                $"  fitness_score: {chromosome.OverallFitness:F3}"
                $"  evolution_id: \"{chromosome.Id}\""
                "  autonomous_generation: true"
                "}"
                ""
                "reasoning {"
                "  This metascript was autonomously evolved by the TARS evolutionary"
                "  grammar system. It represents optimized coordination patterns"
                "  discovered through genetic algorithm evolution."
                "}"
                ""
            ]
            
            let rules = 
                chromosome.Genes
                |> List.mapi (fun i gene ->
                    [
                        $"# Rule {i + 1}: {gene.RulePattern}"
                        $"IF {gene.ConditionLogic} THEN"
                    ] @ 
                    (gene.ActionSequence |> List.map (fun action -> $"  {action}")) @
                    ["END"; ""])
                |> List.concat
            
            String.Join("\n", header @ rules)
        
        /// Save evolved metascript to file
        member private this.SaveEvolvedMetascript(outputPath: string, metascriptContent: string) =
            try
                let directory = Path.GetDirectoryName(outputPath)
                if not (Directory.Exists(directory)) then
                    Directory.CreateDirectory(directory) |> ignore
                
                File.WriteAllText(outputPath, metascriptContent)
                logger.LogInformation("💾 Saved evolved metascript to {OutputPath}", outputPath)
            with
            | ex ->
                logger.LogError(ex, "Error saving evolved metascript to {OutputPath}", outputPath)
        
        /// Get evolution status for all active teams
        member this.GetEvolutionStatus() : Map<string, TeamEvolutionState> =
            activeEvolutions
            |> Seq.map (fun kvp -> kvp.Key, kvp.Value)
            |> Map.ofSeq
        
        /// Stop evolution for a team
        member this.StopTeamEvolution(teamId: string) =
            match activeEvolutions.TryGetValue(teamId) with
            | true, state ->
                let updatedState = { state with IsActive = false }
                activeEvolutions.[teamId] <- updatedState
                logger.LogInformation("⏹️ Stopped evolution for team {TeamId}", teamId)
            | false, _ ->
                logger.LogWarning("Team {TeamId} not found for evolution stop", teamId)
