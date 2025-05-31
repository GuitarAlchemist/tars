namespace TarsEngine.FSharp.Agents

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open FSharp.Control
open AgentTypes
open AgentCommunication

/// Agent team management and coordination
module AgentTeams =
    
    /// Team coordination strategies
    type CoordinationStrategy =
        | Hierarchical of AgentId // Leader-based coordination
        | Democratic // Consensus-based decisions
        | Specialized // Task-based specialization
        | Swarm // Emergent coordination
    
    /// Team decision result
    type TeamDecision = {
        DecisionId: Guid
        Question: string
        Options: string list
        Votes: Map<AgentId, string>
        Result: string option
        Consensus: bool
        Timestamp: DateTime
    }
    
    /// Team performance metrics
    type TeamMetrics = {
        TasksCompleted: int
        SuccessRate: float
        AverageResponseTime: TimeSpan
        CollaborationScore: float
        ConflictCount: int
        DecisionEfficiency: float
        LastUpdated: DateTime
    }
    
    /// Agent team implementation
    type AgentTeam(
        configuration: TeamConfiguration,
        messageBus: MessageBus,
        logger: ILogger<AgentTeam>) =
        
        let mutable members = configuration.Members
        let mutable metrics = {
            TasksCompleted = 0
            SuccessRate = 0.0
            AverageResponseTime = TimeSpan.Zero
            CollaborationScore = 0.0
            ConflictCount = 0
            DecisionEfficiency = 0.0
            LastUpdated = DateTime.UtcNow
        }
        let activeDecisions = System.Collections.Concurrent.ConcurrentDictionary<Guid, TeamDecision>()
        let taskHistory = System.Collections.Concurrent.ConcurrentQueue<AgentTaskResult>()
        
        /// Add agent to team
        member this.AddMember(agentId: AgentId) =
            if not (members |> List.contains agentId) then
                members <- agentId :: members
                logger.LogInformation("Agent {AgentId} added to team {TeamName}", agentId, configuration.Name)
                
                // Notify team of new member
                let message = createMessage 
                    (AgentId(Guid.Empty)) // System message
                    None // Broadcast
                    "TeamMemberAdded"
                    {| TeamName = configuration.Name; NewMember = agentId |}
                    MessagePriority.Normal
                
                messageBus.SendMessageAsync(message) |> ignore
        
        /// Remove agent from team
        member this.RemoveMember(agentId: AgentId) =
            members <- members |> List.filter ((<>) agentId)
            logger.LogInformation("Agent {AgentId} removed from team {TeamName}", agentId, configuration.Name)
        
        /// Get team members
        member this.GetMembers() = members
        
        /// Assign task to team
        member this.AssignTaskAsync(taskName: string, taskDescription: string, requirements: AgentCapability list) =
            task {
                logger.LogInformation("Assigning task {TaskName} to team {TeamName}", taskName, configuration.Name)
                
                // Find suitable agents based on capabilities
                let suitableAgents = 
                    members
                    |> List.filter (fun agentId ->
                        // In a real implementation, we would check agent capabilities
                        // For now, assume all agents can handle any task
                        true)
                
                match suitableAgents with
                | [] ->
                    logger.LogWarning("No suitable agents found for task {TaskName}", taskName)
                    return None
                | agents ->
                    // Distribute task based on coordination strategy
                    let assignedAgent = 
                        match configuration.LeaderAgent with
                        | Some leader when agents |> List.contains leader -> leader
                        | _ -> agents |> List.head // Simple assignment
                    
                    let taskMessage = createMessage
                        (AgentId(Guid.Empty)) // System message
                        (Some assignedAgent)
                        "TaskAssignment"
                        {| TaskName = taskName; Description = taskDescription; Requirements = requirements |}
                        MessagePriority.High
                    
                    do! messageBus.SendMessageAsync(taskMessage)
                    
                    logger.LogInformation("Task {TaskName} assigned to agent {AgentId}", taskName, assignedAgent)
                    return Some assignedAgent
            }
        
        /// Initiate team decision
        member this.InitiateDecisionAsync(question: string, options: string list, timeoutMinutes: int) =
            task {
                let decisionId = Guid.NewGuid()
                let decision = {
                    DecisionId = decisionId
                    Question = question
                    Options = options
                    Votes = Map.empty
                    Result = None
                    Consensus = false
                    Timestamp = DateTime.UtcNow
                }
                
                activeDecisions.TryAdd(decisionId, decision) |> ignore
                
                // Send voting request to all team members
                let votingMessage = createMessage
                    (AgentId(Guid.Empty)) // System message
                    None // Broadcast to team
                    "VotingRequest"
                    {| DecisionId = decisionId; Question = question; Options = options; TimeoutMinutes = timeoutMinutes |}
                    MessagePriority.High
                
                do! messageBus.SendMessageAsync(votingMessage)
                
                logger.LogInformation("Decision {DecisionId} initiated for team {TeamName}: {Question}", 
                                     decisionId, configuration.Name, question)
                
                return decisionId
            }
        
        /// Cast vote for a decision
        member this.CastVoteAsync(decisionId: Guid, agentId: AgentId, vote: string) =
            task {
                match activeDecisions.TryGetValue(decisionId) with
                | true, decision ->
                    if members |> List.contains agentId then
                        let updatedVotes = decision.Votes |> Map.add agentId vote
                        let updatedDecision = { decision with Votes = updatedVotes }
                        activeDecisions.TryUpdate(decisionId, updatedDecision, decision) |> ignore
                        
                        logger.LogInformation("Vote cast by {AgentId} for decision {DecisionId}: {Vote}", 
                                             agentId, decisionId, vote)
                        
                        // Check if all members have voted
                        if updatedVotes.Count = members.Length then
                            do! this.FinalizeDecisionAsync(decisionId)
                    else
                        logger.LogWarning("Agent {AgentId} is not a member of team {TeamName}", 
                                         agentId, configuration.Name)
                | false, _ ->
                    logger.LogWarning("Decision {DecisionId} not found", decisionId)
            }
        
        /// Finalize a team decision
        member private this.FinalizeDecisionAsync(decisionId: Guid) =
            task {
                match activeDecisions.TryGetValue(decisionId) with
                | true, decision ->
                    // Count votes
                    let voteCounts = 
                        decision.Votes.Values
                        |> Seq.groupBy id
                        |> Seq.map (fun (option, votes) -> option, Seq.length votes)
                        |> Map.ofSeq
                    
                    // Determine result
                    let maxVotes = voteCounts.Values |> Seq.max
                    let winners = 
                        voteCounts 
                        |> Map.filter (fun _ count -> count = maxVotes)
                        |> Map.keys
                        |> Seq.toList
                    
                    let result = 
                        match winners with
                        | [single] -> Some single
                        | multiple -> 
                            // Tie - use leader's vote or random selection
                            match configuration.LeaderAgent with
                            | Some leader when decision.Votes.ContainsKey(leader) ->
                                Some decision.Votes.[leader]
                            | _ ->
                                Some (multiple |> List.head) // Random selection
                    
                    let consensus = voteCounts.Count = 1 // All votes the same
                    
                    let finalDecision = { 
                        decision with 
                            Result = result
                            Consensus = consensus 
                    }
                    
                    activeDecisions.TryUpdate(decisionId, finalDecision, decision) |> ignore
                    
                    // Notify team of decision result
                    let resultMessage = createMessage
                        (AgentId(Guid.Empty)) // System message
                        None // Broadcast
                        "DecisionResult"
                        {| DecisionId = decisionId; Result = result; Consensus = consensus |}
                        MessagePriority.High
                    
                    do! messageBus.SendMessageAsync(resultMessage)
                    
                    logger.LogInformation("Decision {DecisionId} finalized for team {TeamName}: {Result} (Consensus: {Consensus})", 
                                         decisionId, configuration.Name, result, consensus)
                | false, _ ->
                    logger.LogWarning("Decision {DecisionId} not found for finalization", decisionId)
            }
        
        /// Get team metrics
        member this.GetMetrics() = metrics
        
        /// Update team metrics
        member this.UpdateMetrics(taskResult: AgentTaskResult) =
            taskHistory.Enqueue(taskResult)
            
            let allResults = taskHistory.ToArray()
            let successCount = allResults |> Array.filter (fun r -> r.Success) |> Array.length
            
            metrics <- {
                TasksCompleted = allResults.Length
                SuccessRate = if allResults.Length > 0 then float successCount / float allResults.Length else 0.0
                AverageResponseTime = 
                    if allResults.Length > 0 then
                        let totalTime = allResults |> Array.sumBy (fun r -> r.ExecutionTime.TotalMilliseconds)
                        TimeSpan.FromMilliseconds(totalTime / float allResults.Length)
                    else TimeSpan.Zero
                CollaborationScore = 0.8 // Placeholder calculation
                ConflictCount = metrics.ConflictCount
                DecisionEfficiency = 0.7 // Placeholder calculation
                LastUpdated = DateTime.UtcNow
            }
        
        /// Get active decisions
        member this.GetActiveDecisions() =
            activeDecisions.Values |> Seq.toList
        
        /// Get team configuration
        member this.GetConfiguration() = configuration
    
    /// Team coordination patterns
    module CoordinationPatterns =
        
        /// Development team pattern
        let developmentTeam = {
            Name = "Development Team"
            Description = "Collaborative development team with architect leadership"
            LeaderAgent = None // Will be set when agents are created
            Members = []
            SharedObjectives = [
                "Deliver high-quality code"
                "Maintain system architecture"
                "Ensure comprehensive testing"
                "Optimize performance"
            ]
            CommunicationProtocol = "Agile/Scrum-based communication"
            DecisionMakingProcess = "Consensus with architect guidance"
            ConflictResolution = "Technical discussion and voting"
        }
        
        /// Research team pattern
        let researchTeam = {
            Name = "Research Team"
            Description = "Exploratory research team with collaborative decision making"
            LeaderAgent = None
            Members = []
            SharedObjectives = [
                "Explore new technologies"
                "Gather knowledge and insights"
                "Validate hypotheses"
                "Share discoveries"
            ]
            CommunicationProtocol = "Open discussion and knowledge sharing"
            DecisionMakingProcess = "Democratic consensus"
            ConflictResolution = "Evidence-based discussion"
        }
        
        /// Quality assurance team pattern
        let qualityAssuranceTeam = {
            Name = "Quality Assurance Team"
            Description = "Quality-focused team with guardian leadership"
            LeaderAgent = None
            Members = []
            SharedObjectives = [
                "Ensure code quality"
                "Maintain security standards"
                "Validate system reliability"
                "Monitor performance"
            ]
            CommunicationProtocol = "Structured reporting and reviews"
            DecisionMakingProcess = "Risk-based decision making"
            ConflictResolution = "Standards-based resolution"
        }
