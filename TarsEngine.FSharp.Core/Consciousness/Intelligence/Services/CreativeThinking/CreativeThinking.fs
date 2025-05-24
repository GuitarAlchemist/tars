namespace TarsEngine.FSharp.Core.Consciousness.Intelligence.Services

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Consciousness.Intelligence
open TarsEngine.FSharp.Core.Consciousness.Intelligence.Services.CreativeThinking

/// <summary>
/// Implementation of the creative thinking capabilities.
/// </summary>
type CreativeThinking(logger: ILogger<CreativeThinking>) =
    inherit CreativeThinkingBase(logger)
    
    let random = System.Random()
    let mutable lastIdeaGenerationTime = DateTime.MinValue
    
    /// <summary>
    /// Generates a creative idea.
    /// </summary>
    /// <returns>The generated creative idea.</returns>
    member this.GenerateCreativeIdeaAsync() =
        task {
            if not this.IsInitialized || not this.IsActive then
                return None
            
            // Only generate ideas periodically
            if (DateTime.UtcNow - lastIdeaGenerationTime).TotalSeconds < 30 then
                return None
            
            try
                logger.LogDebug("Generating creative idea")
                
                // Choose a creative process based on current levels
                let processType = CreativeIdeaGeneration.chooseCreativeProcess 
                                    this.DivergentThinkingLevel 
                                    this.ConvergentThinkingLevel 
                                    this.CombinatorialCreativityLevel 
                                    random
                
                // Generate idea based on process type
                let idea = CreativeIdeaGeneration.generateIdeaByProcess 
                            processType 
                            this.DivergentThinkingLevel 
                            this.ConvergentThinkingLevel 
                            this.CombinatorialCreativityLevel 
                            random
                
                // Add to ideas list
                this.AddCreativeIdea(idea)
                
                // Add creative process
                let process = {
                    Id = Guid.NewGuid().ToString()
                    Type = processType
                    Description = sprintf "Generated idea using %A process" processType
                    Timestamp = DateTime.UtcNow
                    IdeaId = idea.Id
                    Effectiveness = idea.Originality * idea.Value
                }
                
                this.AddCreativeProcess(process)
                
                lastIdeaGenerationTime <- DateTime.UtcNow
                
                logger.LogInformation("Generated creative idea: {Description} (Originality: {Originality:F2}, Value: {Value:F2})",
                                     idea.Description, idea.Originality, idea.Value)
                
                return Some idea
            with
            | ex ->
                logger.LogError(ex, "Error generating creative idea")
                return None
        }
    
    /// <summary>
    /// Generates a creative solution to a problem.
    /// </summary>
    /// <param name="problem">The problem description.</param>
    /// <param name="constraints">The constraints.</param>
    /// <returns>The creative solution.</returns>
    member this.GenerateCreativeSolutionAsync(problem: string, constraints: string list option) =
        task {
            if not this.IsInitialized || not this.IsActive then
                logger.LogWarning("Cannot generate creative solution: creative thinking not initialized or active")
                return None
            
            try
                logger.LogInformation("Generating creative solution for problem: {Problem}", problem)
                
                // Analyze the problem to determine the best approach
                let processType = CreativeSolutionGeneration.analyzeProblem 
                                    problem 
                                    constraints 
                                    this.DivergentThinkingLevel 
                                    this.ConvergentThinkingLevel 
                                    this.CombinatorialCreativityLevel 
                                    random
                
                // Generate solution based on process type
                let solution = CreativeSolutionGeneration.generateSolutionByProcess 
                                problem 
                                constraints 
                                processType 
                                this.DivergentThinkingLevel 
                                this.ConvergentThinkingLevel 
                                this.CombinatorialCreativityLevel 
                                random
                
                // Add to ideas list
                this.AddCreativeIdea(solution)
                
                // Add creative process
                let process = {
                    Id = Guid.NewGuid().ToString()
                    Type = processType
                    Description = sprintf "Generated solution for problem: %s" problem
                    Timestamp = DateTime.UtcNow
                    IdeaId = solution.Id
                    Effectiveness = solution.Originality * solution.Value
                }
                
                this.AddCreativeProcess(process)
                
                logger.LogInformation("Generated creative solution: {Description} (Originality: {Originality:F2}, Value: {Value:F2})",
                                     solution.Description, solution.Originality, solution.Value)
                
                return Some solution
            with
            | ex ->
                logger.LogError(ex, "Error generating creative solution")
                return None
        }
    
    interface ICreativeThinking with
        member this.CreativityLevel = this.CreativityLevel
        member this.DivergentThinkingLevel = this.DivergentThinkingLevel
        member this.ConvergentThinkingLevel = this.ConvergentThinkingLevel
        member this.CombinatorialCreativityLevel = this.CombinatorialCreativityLevel
        member this.CreativeIdeas = this.CreativeIdeas
        member this.CreativeProcesses = this.CreativeProcesses
        
        member this.InitializeAsync() = this.InitializeAsync()
        member this.ActivateAsync() = this.ActivateAsync()
        member this.DeactivateAsync() = this.DeactivateAsync()
        member this.UpdateAsync() = this.UpdateAsync()
        
        member this.GenerateCreativeIdeaAsync() = this.GenerateCreativeIdeaAsync()
        member this.GenerateCreativeSolutionAsync(problem, constraints) = this.GenerateCreativeSolutionAsync(problem, constraints)
        
        member this.GetRecentIdeas(count) = this.GetRecentIdeas(count)
        member this.GetIdeasByDomain(domain, count) = this.GetIdeasByDomain(domain, count)
        member this.GetIdeasByTag(tag, count) = this.GetIdeasByTag(tag, count)
        member this.GetMostOriginalIdeas(count) = this.GetMostOriginalIdeas(count)
        member this.GetMostValuableIdeas(count) = this.GetMostValuableIdeas(count)
        member this.GetMostEffectiveProcesses(count) = this.GetMostEffectiveProcesses(count)
