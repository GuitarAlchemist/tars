namespace TarsEngine.FSharp.Agents

open System
open System.IO
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open FSharp.Control
open AgentTypes
open AgentPersonas
open AgentCommunication

/// Long-running metascript-based agents with TaskSeq support
module MetascriptAgent =
    
    /// Metascript agent state
    type MetascriptAgentState = {
        Agent: Agent
        Communication: IAgentCommunication
        CurrentExecution: TaskSeq<AgentTaskResult> option
        ExecutionCancellation: CancellationTokenSource option
        LearningData: LearningData list
        PerformanceMetrics: AgentMetrics
    }
    
    /// Metascript agent implementation
    type MetascriptAgent(
        persona: AgentPersona,
        metascriptPath: string,
        messageBus: MessageBus,
        logger: ILogger<MetascriptAgent>) =
        
        let agentId = AgentId(Guid.NewGuid())
        let messageChannel = messageBus.RegisterAgent(agentId)
        let communication = AgentCommunication(agentId, messageBus, logger) :> IAgentCommunication
        
        let mutable state = {
            Agent = {
                Id = agentId
                Persona = persona
                Status = AgentStatus.Initializing
                Context = {
                    AgentId = agentId
                    WorkingDirectory = Path.GetDirectoryName(metascriptPath)
                    Variables = Map.empty
                    SharedMemory = Map.empty
                    CancellationToken = CancellationToken.None
                    Logger = logger
                }
                CurrentTasks = []
                MessageQueue = messageChannel
                MetascriptPath = Some metascriptPath
                StartTime = DateTime.UtcNow
                LastActivity = DateTime.UtcNow
                Statistics = Map.empty
            }
            Communication = communication
            CurrentExecution = None
            ExecutionCancellation = None
            LearningData = []
            PerformanceMetrics = {
                TasksCompleted = 0
                TasksSuccessful = 0
                AverageExecutionTime = TimeSpan.Zero
                MessagesProcessed = 0
                CollaborationScore = persona.CollaborationPreference
                LearningProgress = 0.0
                EfficiencyRating = 0.0
                LastUpdated = DateTime.UtcNow
            }
        }
        
        /// Start the agent
        member this.StartAsync() =
            task {
                try
                    logger.LogInformation("Starting metascript agent {AgentId} with persona {PersonaName}", 
                                         agentId, persona.Name)
                    
                    state <- { state with Agent = { state.Agent with Status = AgentStatus.Running } }
                    
                    // Start message processing
                    let messageProcessingTask = this.ProcessMessagesAsync()
                    
                    // Start metascript execution if available
                    let executionTask = 
                        match metascriptPath with
                        | path when File.Exists(path) -> this.ExecuteMetascriptAsync(path)
                        | _ -> Task.CompletedTask
                    
                    // Run both tasks concurrently
                    let! _ = Task.WhenAll([| messageProcessingTask; executionTask |])
                    
                    logger.LogInformation("Metascript agent {AgentId} started successfully", agentId)
                    
                with
                | ex ->
                    logger.LogError(ex, "Failed to start metascript agent {AgentId}", agentId)
                    state <- { state with Agent = { state.Agent with Status = AgentStatus.Failed(ex.Message) } }
            }
        
        /// Stop the agent
        member this.StopAsync() =
            task {
                try
                    logger.LogInformation("Stopping metascript agent {AgentId}", agentId)
                    
                    state <- { state with Agent = { state.Agent with Status = AgentStatus.Stopping } }
                    
                    // Cancel current execution
                    match state.ExecutionCancellation with
                    | Some cts -> cts.Cancel()
                    | None -> ()
                    
                    // Unregister from message bus
                    messageBus.UnregisterAgent(agentId)
                    
                    state <- { state with Agent = { state.Agent with Status = AgentStatus.Stopped } }
                    
                    logger.LogInformation("Metascript agent {AgentId} stopped", agentId)
                    
                with
                | ex ->
                    logger.LogError(ex, "Error stopping metascript agent {AgentId}", agentId)
            }
        
        /// Execute metascript with TaskSeq for long-running operations
        member private this.ExecuteMetascriptAsync(scriptPath: string) =
            task {
                try
                    let cts = new CancellationTokenSource()
                    state <- { state with ExecutionCancellation = Some cts }
                    
                    logger.LogInformation("Executing metascript {ScriptPath} for agent {AgentId}", 
                                         scriptPath, agentId)
                    
                    // Create long-running task sequence
                    let executionResults = taskSeq {
                        let startTime = DateTime.UtcNow
                        
                        // Simulate metascript execution with streaming results
                        for i in 1..10 do
                            if not cts.Token.IsCancellationRequested then
                                // Simulate work
                                do! Task.Delay(1000, cts.Token)
                                
                                let result = {
                                    Success = true
                                    Output = Some $"Step {i} completed by {persona.Name}"
                                    Error = None
                                    ExecutionTime = DateTime.UtcNow - startTime
                                    Metadata = Map.ofList [
                                        ("step", i :> obj)
                                        ("agent_id", agentId :> obj)
                                        ("persona", persona.Name :> obj)
                                    ]
                                }
                                
                                // Update metrics
                                this.UpdateMetrics(result)
                                
                                // Send progress update
                                let progressMessage = createMessage
                                    agentId
                                    None // Broadcast
                                    "ExecutionProgress"
                                    {| Step = i; Total = 10; Result = result |}
                                    MessagePriority.Low
                                
                                do! communication.SendMessageAsync(progressMessage)
                                
                                yield result
                    }
                    
                    state <- { state with CurrentExecution = Some executionResults }
                    
                    // Process all results
                    for result in executionResults do
                        logger.LogDebug("Metascript step completed: {Output}", result.Output)
                    
                    logger.LogInformation("Metascript execution completed for agent {AgentId}", agentId)
                    
                with
                | :? OperationCanceledException ->
                    logger.LogInformation("Metascript execution cancelled for agent {AgentId}", agentId)
                | ex ->
                    logger.LogError(ex, "Error executing metascript for agent {AgentId}", agentId)
                    
                    let errorResult = {
                        Success = false
                        Output = None
                        Error = Some ex.Message
                        ExecutionTime = TimeSpan.Zero
                        Metadata = Map.empty
                    }
                    
                    this.UpdateMetrics(errorResult)
            }
        
        /// Process incoming messages
        member private this.ProcessMessagesAsync() =
            task {
                try
                    logger.LogInformation("Starting message processing for agent {AgentId}", agentId)
                    
                    let messageStream = communication.GetMessageStream()
                    
                    for message in messageStream do
                        try
                            state <- { state with Agent = { state.Agent with LastActivity = DateTime.UtcNow } }
                            
                            // Update message count
                            let updatedMetrics = { 
                                state.PerformanceMetrics with 
                                    MessagesProcessed = state.PerformanceMetrics.MessagesProcessed + 1
                                    LastUpdated = DateTime.UtcNow
                            }
                            state <- { state with PerformanceMetrics = updatedMetrics }
                            
                            // Process message based on type
                            match message.MessageType with
                            | "TaskAssignment" ->
                                do! this.HandleTaskAssignment(message)
                            | "VotingRequest" ->
                                do! this.HandleVotingRequest(message)
                            | "CollaborationRequest" ->
                                do! this.HandleCollaborationRequest(message)
                            | "LearningData" ->
                                this.HandleLearningData(message)
                            | _ ->
                                logger.LogDebug("Agent {AgentId} received message: {MessageType}", 
                                               agentId, message.MessageType)
                        with
                        | ex ->
                            logger.LogError(ex, "Error processing message {MessageId} for agent {AgentId}", 
                                           message.Id, agentId)
                    
                with
                | ex ->
                    logger.LogError(ex, "Error in message processing for agent {AgentId}", agentId)
            }
        
        /// Handle task assignment
        member private this.HandleTaskAssignment(message: AgentMessage) =
            task {
                logger.LogInformation("Agent {AgentId} received task assignment", agentId)
                
                // Check if agent has required capabilities
                let taskData = message.Content :?> {| TaskName: string; Description: string; Requirements: AgentCapability list |}
                
                let hasCapabilities = 
                    taskData.Requirements 
                    |> List.forall (fun req -> persona.Capabilities |> List.contains req)
                
                if hasCapabilities then
                    // Accept task
                    let response = {| Accepted = true; EstimatedTime = TimeSpan.FromMinutes(30) |}
                    do! communication.ReplyToAsync(message, response)
                    
                    logger.LogInformation("Agent {AgentId} accepted task: {TaskName}", agentId, taskData.TaskName)
                else
                    // Decline task
                    let response = {| Accepted = false; Reason = "Missing required capabilities" |}
                    do! communication.ReplyToAsync(message, response)
                    
                    logger.LogInformation("Agent {AgentId} declined task: {TaskName}", agentId, taskData.TaskName)
            }
        
        /// Handle voting request
        member private this.HandleVotingRequest(message: AgentMessage) =
            task {
                let votingData = message.Content :?> {| DecisionId: Guid; Question: string; Options: string list |}
                
                // Make decision based on persona
                let vote = 
                    match persona.DecisionMakingStyle with
                    | style when style.Contains("consensus") -> votingData.Options |> List.head
                    | style when style.Contains("risk-averse") -> votingData.Options |> List.last
                    | _ -> votingData.Options |> List.head // Default choice
                
                let voteMessage = createMessage
                    agentId
                    message.ReplyTo
                    "Vote"
                    {| DecisionId = votingData.DecisionId; Vote = vote |}
                    MessagePriority.Normal
                
                do! communication.SendMessageAsync(voteMessage)
                
                logger.LogInformation("Agent {AgentId} voted: {Vote} for decision {DecisionId}", 
                                     agentId, vote, votingData.DecisionId)
            }
        
        /// Handle collaboration request
        member private this.HandleCollaborationRequest(message: AgentMessage) =
            task {
                // Respond based on collaboration preference
                let willCollaborate = 
                    let random = Random()
                    random.NextDouble() < persona.CollaborationPreference
                
                let response = {| WillCollaborate = willCollaborate; Availability = "Available" |}
                do! communication.ReplyToAsync(message, response)
                
                logger.LogInformation("Agent {AgentId} collaboration response: {WillCollaborate}", 
                                     agentId, willCollaborate)
            }
        
        /// Handle learning data
        member private this.HandleLearningData(message: AgentMessage) =
            let learningData = message.Content :?> LearningData
            state <- { state with LearningData = learningData :: state.LearningData }
            
            // Update learning progress
            let updatedMetrics = { 
                state.PerformanceMetrics with 
                    LearningProgress = min 1.0 (state.PerformanceMetrics.LearningProgress + persona.LearningRate * 0.1)
                    LastUpdated = DateTime.UtcNow
            }
            state <- { state with PerformanceMetrics = updatedMetrics }
            
            logger.LogDebug("Agent {AgentId} processed learning data: {ExperienceType}", 
                           agentId, learningData.ExperienceType)
        
        /// Update performance metrics
        member private this.UpdateMetrics(result: AgentTaskResult) =
            let currentMetrics = state.PerformanceMetrics
            let newTaskCount = currentMetrics.TasksCompleted + 1
            let newSuccessCount = 
                if result.Success then currentMetrics.TasksSuccessful + 1 
                else currentMetrics.TasksSuccessful
            
            let newAverageTime = 
                let totalTime = currentMetrics.AverageExecutionTime.TotalMilliseconds * float currentMetrics.TasksCompleted
                let newTotalTime = totalTime + result.ExecutionTime.TotalMilliseconds
                TimeSpan.FromMilliseconds(newTotalTime / float newTaskCount)
            
            let updatedMetrics = {
                TasksCompleted = newTaskCount
                TasksSuccessful = newSuccessCount
                AverageExecutionTime = newAverageTime
                MessagesProcessed = currentMetrics.MessagesProcessed
                CollaborationScore = currentMetrics.CollaborationScore
                LearningProgress = currentMetrics.LearningProgress
                EfficiencyRating = float newSuccessCount / float newTaskCount
                LastUpdated = DateTime.UtcNow
            }
            
            state <- { state with PerformanceMetrics = updatedMetrics }
        
        /// Get agent state
        member this.GetState() = state
        
        /// Get agent ID
        member this.GetId() = agentId
        
        /// Get agent persona
        member this.GetPersona() = persona
        
        /// Get performance metrics
        member this.GetMetrics() = state.PerformanceMetrics
        
        /// Send message to another agent
        member this.SendMessageAsync(targetAgent: AgentId, messageType: string, content: obj) =
            let message = createMessage agentId (Some targetAgent) messageType content MessagePriority.Normal
            communication.SendMessageAsync(message)
        
        /// Broadcast message to all agents
        member this.BroadcastAsync(messageType: string, content: obj) =
            communication.BroadcastAsync(messageType, content)
