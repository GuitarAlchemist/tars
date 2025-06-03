namespace TarsEngine.FSharp.Reasoning

open System
open System.Threading.Tasks
open System.Collections.Generic
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Reasoning

/// Complex reasoning problem that may require multiple agents
type ComplexReasoningProblem = {
    Id: string
    Description: string
    Context: string option
    RequiredCapabilities: string list
    Priority: int
    Deadline: DateTime option
    ExpectedComplexity: int  // 1-10 scale
}

/// Reasoning task assignment
type ReasoningTaskAssignment = {
    TaskId: string
    AgentId: string
    Problem: ReasoningRequest
    AssignedAt: DateTime
    EstimatedDuration: TimeSpan
}

/// Collaborative reasoning session
type CollaborativeReasoningSession = {
    SessionId: string
    Problem: ComplexReasoningProblem
    ParticipatingAgents: ReasoningAgent list
    Coordinator: ReasoningAgent
    StartTime: DateTime
    Status: string
    IntermediateResults: ReasoningResponse list
    FinalSynthesis: ReasoningResponse option
}

/// Reasoning team coordinator interface
type IReasoningTeamCoordinator =
    abstract member SolveComplexProblemAsync: ComplexReasoningProblem -> Task<ReasoningResponse>
    abstract member GetAvailableAgents: unit -> ReasoningAgent list
    abstract member CreateCollaborativeSession: ComplexReasoningProblem -> Task<CollaborativeReasoningSession>
    abstract member SynthesizeResults: ReasoningResponse list -> Task<ReasoningResponse>

/// Reasoning team coordinator implementation
type ReasoningTeamCoordinator(logger: ILogger<ReasoningTeamCoordinator>) =
    
    let agents = new Dictionary<string, ReasoningAgent>()
    let activeSessions = new Dictionary<string, CollaborativeReasoningSession>()
    
    /// Initialize reasoning team with specialized agents
    member private this.InitializeTeam() =
        logger.LogInformation("Initializing Qwen3 Reasoning Team...")
        
        let specializations = [
            ReasoningSpecialization.MathematicalReasoning
            ReasoningSpecialization.LogicalReasoning
            ReasoningSpecialization.CausalReasoning
            ReasoningSpecialization.StrategicReasoning
            ReasoningSpecialization.MetaReasoning
            ReasoningSpecialization.CollaborativeReasoning
        ]
        
        for specialization in specializations do
            let agent = QwenReasoningEngineFactory.createSpecializedAgent specialization logger
            agents.[agent.Id] <- agent
            logger.LogInformation($"Created reasoning agent: {agent.Name} ({agent.Specialization})")
        
        logger.LogInformation($"Reasoning team initialized with {agents.Count} agents")
    
    /// Analyze problem complexity and required capabilities
    member private this.AnalyzeProblem(problem: ComplexReasoningProblem) =
        let complexityFactors = [
            problem.Description.Length / 100
            problem.RequiredCapabilities.Length * 2
            if problem.Context.IsSome then problem.Context.Value.Length / 200 else 0
            problem.ExpectedComplexity * 10
        ]
        
        let totalComplexity = complexityFactors |> List.sum
        
        let requiredSpecializations = 
            problem.RequiredCapabilities
            |> List.choose (fun capability ->
                match capability.ToLower() with
                | cap when cap.Contains("math") || cap.Contains("calculation") -> Some MathematicalReasoning
                | cap when cap.Contains("logic") || cap.Contains("deduction") -> Some LogicalReasoning
                | cap when cap.Contains("cause") || cap.Contains("causal") -> Some CausalReasoning
                | cap when cap.Contains("strategy") || cap.Contains("decision") -> Some StrategicReasoning
                | cap when cap.Contains("meta") || cap.Contains("reasoning") -> Some MetaReasoning
                | cap when cap.Contains("collaborative") || cap.Contains("consensus") -> Some CollaborativeReasoning
                | _ -> None)
            |> List.distinct
        
        (totalComplexity, requiredSpecializations)
    
    /// Select optimal agents for a problem
    member private this.SelectAgentsForProblem(problem: ComplexReasoningProblem) =
        let (complexity, requiredSpecializations) = this.AnalyzeProblem(problem)
        
        let selectedAgents = 
            agents.Values
            |> Seq.filter (fun agent -> 
                requiredSpecializations |> List.contains agent.Specialization ||
                agent.Capabilities |> List.exists (fun cap -> 
                    problem.RequiredCapabilities |> List.contains cap))
            |> Seq.toList
        
        // Always include meta-reasoner for complex problems
        let metaReasoner = 
            agents.Values 
            |> Seq.tryFind (fun a -> a.Specialization = MetaReasoning)
        
        let finalAgents = 
            match metaReasoner with
            | Some meta when complexity > 50 && not (selectedAgents |> List.exists (fun a -> a.Id = meta.Id)) ->
                meta :: selectedAgents
            | _ -> selectedAgents
        
        // Add collaborative reasoner for multi-agent sessions
        if finalAgents.Length > 1 then
            let collaborativeReasoner = 
                agents.Values 
                |> Seq.tryFind (fun a -> a.Specialization = CollaborativeReasoning)
            
            match collaborativeReasoner with
            | Some collab when not (finalAgents |> List.exists (fun a -> a.Id = collab.Id)) ->
                collab :: finalAgents
            | _ -> finalAgents
        else
            finalAgents
    
    /// Decompose complex problem into sub-problems
    member private this.DecomposeProblem(problem: ComplexReasoningProblem) (selectedAgents: ReasoningAgent list) =
        // Create reasoning requests for each selected agent
        selectedAgents
        |> List.map (fun agent ->
            let specializedPrompt = 
                match agent.Specialization with
                | MathematicalReasoning -> 
                    $"Analyze the mathematical aspects of this problem: {problem.Description}"
                | LogicalReasoning -> 
                    $"Apply logical reasoning to this problem: {problem.Description}"
                | CausalReasoning -> 
                    $"Identify causal relationships in this problem: {problem.Description}"
                | StrategicReasoning -> 
                    $"Develop strategic approaches for this problem: {problem.Description}"
                | MetaReasoning -> 
                    $"Analyze the reasoning strategies needed for this problem: {problem.Description}"
                | CollaborativeReasoning -> 
                    $"Coordinate the collaborative solution of this problem: {problem.Description}"
            
            {
                Problem = specializedPrompt
                Context = problem.Context
                Mode = ReasoningMode.Thinking  // Use deep thinking for complex problems
                ThinkingBudget = Some (problem.ExpectedComplexity * 100)
                RequiredCapabilities = agent.Capabilities
                Priority = problem.Priority
            })
    
    /// Synthesize multiple reasoning results into final answer
    member private this.SynthesizeResults(results: ReasoningResponse list) = async {
        if results.IsEmpty then
            return {
                Problem = "No results to synthesize"
                ThinkingContent = None
                FinalAnswer = "No reasoning results available"
                Confidence = 0.0
                ReasoningSteps = 0
                ProcessingTime = TimeSpan.Zero
                Model = Qwen3_8B
                Mode = ReasoningMode.NonThinking
            }
        
        // Find the collaborative reasoner for synthesis
        let collaborativeAgent = 
            agents.Values 
            |> Seq.tryFind (fun a -> a.Specialization = CollaborativeReasoning)
        
        match collaborativeAgent with
        | Some agent ->
            let synthesisPrompt = 
                let resultsSummary = 
                    results
                    |> List.mapi (fun i result -> 
                        $"Result {i+1}: {result.FinalAnswer}")
                    |> String.concat "\n"
                
                $"Synthesize these reasoning results into a comprehensive final answer:\n{resultsSummary}"
            
            let synthesisRequest = {
                Problem = synthesisPrompt
                Context = None
                Mode = ReasoningMode.Thinking
                ThinkingBudget = Some 500
                RequiredCapabilities = ["synthesis"; "consensus_building"]
                Priority = 1
            }
            
            let! synthesisResult = agent.Engine.ReasonAsync(synthesisRequest) |> Async.AwaitTask
            return synthesisResult
            
        | None ->
            // Fallback: simple aggregation
            let combinedAnswer = 
                results
                |> List.map (fun r -> r.FinalAnswer)
                |> String.concat "\n\n"
            
            let avgConfidence = 
                results 
                |> List.map (fun r -> r.Confidence)
                |> List.average
            
            return {
                Problem = "Synthesized results"
                ThinkingContent = Some "Combined multiple reasoning perspectives"
                FinalAnswer = combinedAnswer
                Confidence = avgConfidence
                ReasoningSteps = results |> List.sumBy (fun r -> r.ReasoningSteps)
                ProcessingTime = results |> List.map (fun r -> r.ProcessingTime) |> List.maxBy (fun t -> t.TotalMilliseconds)
                Model = Qwen3_235B_A22B
                Mode = ReasoningMode.Hybrid
            }
    }
    
    interface IReasoningTeamCoordinator with
        
        member this.SolveComplexProblemAsync(problem: ComplexReasoningProblem) = task {
            logger.LogInformation($"Solving complex problem: {problem.Id}")
            
            try
                // Select optimal agents for the problem
                let selectedAgents = this.SelectAgentsForProblem(problem)
                logger.LogInformation($"Selected {selectedAgents.Length} agents for problem {problem.Id}")
                
                if selectedAgents.IsEmpty then
                    logger.LogWarning($"No suitable agents found for problem {problem.Id}")
                    return {
                        Problem = problem.Description
                        ThinkingContent = Some "No suitable reasoning agents available"
                        FinalAnswer = "Unable to solve: No appropriate reasoning capabilities available"
                        Confidence = 0.0
                        ReasoningSteps = 0
                        ProcessingTime = TimeSpan.Zero
                        Model = Qwen3_8B
                        Mode = ReasoningMode.NonThinking
                    }
                
                // Decompose problem into specialized sub-problems
                let subProblems = this.DecomposeProblem(problem) selectedAgents
                
                // Execute reasoning tasks in parallel
                let reasoningTasks = 
                    List.zip selectedAgents subProblems
                    |> List.map (fun (agent, subProblem) ->
                        agent.Engine.ReasonAsync(subProblem))
                
                let! results = Task.WhenAll(reasoningTasks)
                let resultsList = results |> Array.toList
                
                logger.LogInformation($"Completed {resultsList.Length} reasoning tasks for problem {problem.Id}")
                
                // Synthesize results into final answer
                let! finalResult = this.SynthesizeResults(resultsList) |> Async.StartAsTask
                
                logger.LogInformation($"Synthesized final result for problem {problem.Id} with confidence {finalResult.Confidence}")
                
                return finalResult
                
            with
            | ex ->
                logger.LogError(ex, $"Error solving complex problem {problem.Id}")
                return {
                    Problem = problem.Description
                    ThinkingContent = Some $"Error occurred: {ex.Message}"
                    FinalAnswer = $"Unable to solve due to error: {ex.Message}"
                    Confidence = 0.0
                    ReasoningSteps = 0
                    ProcessingTime = TimeSpan.Zero
                    Model = Qwen3_8B
                    Mode = ReasoningMode.NonThinking
                }
        }
        
        member this.GetAvailableAgents() =
            agents.Values |> Seq.toList
        
        member this.CreateCollaborativeSession(problem: ComplexReasoningProblem) = task {
            let selectedAgents = this.SelectAgentsForProblem(problem)
            let coordinator = 
                selectedAgents 
                |> List.tryFind (fun a -> a.Specialization = CollaborativeReasoning)
                |> Option.defaultWith (fun () -> selectedAgents.Head)
            
            let session = {
                SessionId = Guid.NewGuid().ToString()
                Problem = problem
                ParticipatingAgents = selectedAgents
                Coordinator = coordinator
                StartTime = DateTime.UtcNow
                Status = "Active"
                IntermediateResults = []
                FinalSynthesis = None
            }
            
            activeSessions.[session.SessionId] <- session
            logger.LogInformation($"Created collaborative reasoning session {session.SessionId}")
            
            return session
        }
        
        member this.SynthesizeResults(results: ReasoningResponse list) = task {
            let! synthesized = this.SynthesizeResults(results) |> Async.StartAsTask
            return synthesized
        }
    
    /// Initialize the coordinator
    member this.InitializeAsync() = async {
        this.InitializeTeam()
        logger.LogInformation("Reasoning Team Coordinator initialized successfully")
    }

/// Factory for creating reasoning team coordinators
module ReasoningTeamCoordinatorFactory =
    
    let create (logger: ILogger<ReasoningTeamCoordinator>) =
        let coordinator = new ReasoningTeamCoordinator(logger)
        coordinator.InitializeAsync() |> Async.RunSynchronously
        coordinator :> IReasoningTeamCoordinator
