namespace TarsEngine.FSharp.Reasoning

open System
open System.Collections.Concurrent
open System.Threading
open System.Threading.Tasks
open System.Threading.Channels
open Microsoft.Extensions.Logging

/// Real-time reasoning request
type RealTimeReasoningRequest = {
    RequestId: string
    Problem: string
    Context: string option
    Priority: int
    Deadline: DateTime option
    StreamingEnabled: bool
    CallbackUrl: string option
}

/// Real-time reasoning update
type RealTimeReasoningUpdate = {
    RequestId: string
    UpdateType: string
    Content: string
    Confidence: float
    Progress: float
    Timestamp: DateTime
}

/// Streaming reasoning result
type StreamingReasoningResult = {
    RequestId: string
    IsComplete: bool
    CurrentStep: ThoughtStep option
    IntermediateResult: string option
    FinalResult: ReasoningResponse option
    Updates: RealTimeReasoningUpdate list
    ProcessingTime: TimeSpan
}

/// Real-time reasoning context
type ReasoningContext = {
    ContextId: string
    ActiveRequests: string list
    SharedKnowledge: Map<string, obj>
    ContextHistory: ReasoningResponse list
    LastUpdated: DateTime
}

/// Interface for real-time reasoning
type IRealTimeReasoningEngine =
    abstract member ProcessStreamingAsync: RealTimeReasoningRequest -> IAsyncEnumerable<StreamingReasoningResult>
    abstract member HandleInterruptAsync: string -> RealTimeReasoningRequest -> Task<unit>
    abstract member MaintainContext: string -> ReasoningContext -> Task<unit>
    abstract member GetActiveRequests: unit -> RealTimeReasoningRequest list

/// Real-time reasoning engine implementation
type RealTimeReasoningEngine(
    chainEngine: IChainOfThoughtEngine,
    budgetController: IDynamicBudgetController,
    qualityMetrics: IReasoningQualityMetrics,
    logger: ILogger<RealTimeReasoningEngine>) =
    
    let activeRequests = new ConcurrentDictionary<string, RealTimeReasoningRequest>()
    let reasoningContexts = new ConcurrentDictionary<string, ReasoningContext>()
    let updateChannels = new ConcurrentDictionary<string, Channel<RealTimeReasoningUpdate>>()
    let cancellationTokens = new ConcurrentDictionary<string, CancellationTokenSource>()
    
    /// Create update channel for streaming
    let createUpdateChannel() =
        let options = new BoundedChannelOptions(1000)
        options.FullMode <- BoundedChannelFullMode.Wait
        options.SingleReader <- false
        options.SingleWriter <- true
        Channel.CreateBounded<RealTimeReasoningUpdate>(options)
    
    /// Send real-time update
    let sendUpdate (requestId: string) (updateType: string) (content: string) (confidence: float) (progress: float) = async {
        try
            match updateChannels.TryGetValue(requestId) with
            | (true, channel) ->
                let update = {
                    RequestId = requestId
                    UpdateType = updateType
                    Content = content
                    Confidence = confidence
                    Progress = progress
                    Timestamp = DateTime.UtcNow
                }
                
                let! success = channel.Writer.WriteAsync(update).AsTask() |> Async.AwaitTask
                if not success then
                    logger.LogWarning($"Failed to send update for request {requestId}")
            | (false, _) ->
                logger.LogWarning($"No update channel found for request {requestId}")
        with
        | ex ->
            logger.LogError(ex, $"Error sending update for request {requestId}")
    }
    
    /// Process reasoning with real-time updates
    let processWithUpdates (request: RealTimeReasoningRequest) (cancellationToken: CancellationToken) = async {
        try
            let startTime = DateTime.UtcNow
            
            // Send initial update
            do! sendUpdate request.RequestId "started" "Reasoning process initiated" 0.0 0.0
            
            // Allocate budget
            let! budget = budgetController.AllocateBudgetAsync request.Problem request.Priority |> Async.AwaitTask
            do! sendUpdate request.RequestId "budget_allocated" $"Budget allocated: {budget.ComputationalUnits} units" 0.1 0.1
            
            // Check for cancellation
            cancellationToken.ThrowIfCancellationRequested()
            
            // Generate chain of thought with progress updates
            do! sendUpdate request.RequestId "thinking_started" "Generating chain of thought" 0.2 0.2
            
            let! chain = chainEngine.GenerateChainAsync request.Problem request.Context |> Async.AwaitTask
            do! sendUpdate request.RequestId "chain_generated" $"Generated {chain.Steps.Length} reasoning steps" 0.6 0.6
            
            // Check for cancellation
            cancellationToken.ThrowIfCancellationRequested()
            
            // Validate chain
            do! sendUpdate request.RequestId "validating" "Validating reasoning chain" 0.7 0.7
            let! validation = chainEngine.ValidateChainAsync chain |> Async.AwaitTask
            
            // Assess quality
            do! sendUpdate request.RequestId "quality_assessment" "Assessing reasoning quality" 0.8 0.8
            let! qualityAssessment = qualityMetrics.AssessQualityAsync chain |> Async.AwaitTask
            
            // Create final result
            let processingTime = DateTime.UtcNow - startTime
            let finalResult = {
                Problem = request.Problem
                ThinkingContent = Some (chainEngine.VisualizeChain chain)
                FinalAnswer = chain.FinalConclusion
                Confidence = chain.OverallConfidence
                ReasoningSteps = chain.Steps.Length
                ProcessingTime = processingTime
                Model = Qwen3_14B  // Default model for real-time
                Mode = ReasoningMode.Hybrid
            }
            
            do! sendUpdate request.RequestId "completed" "Reasoning process completed" finalResult.Confidence 1.0
            
            return Some finalResult
            
        with
        | :? OperationCanceledException ->
            do! sendUpdate request.RequestId "cancelled" "Reasoning process cancelled" 0.0 0.0
            return None
        | ex ->
            logger.LogError(ex, $"Error processing real-time reasoning for request {request.RequestId}")
            do! sendUpdate request.RequestId "error" $"Error: {ex.Message}" 0.0 0.0
            return None
    }
    
    /// Monitor reasoning context
    let monitorContext (contextId: string) = async {
        try
            while reasoningContexts.ContainsKey(contextId) do
                match reasoningContexts.TryGetValue(contextId) with
                | (true, context) ->
                    // Update context with latest information
                    let updatedContext = { context with LastUpdated = DateTime.UtcNow }
                    reasoningContexts.TryUpdate(contextId, updatedContext, context) |> ignore
                    
                    // Clean up old requests
                    let activeRequestIds = 
                        context.ActiveRequests 
                        |> List.filter (fun reqId -> activeRequests.ContainsKey(reqId))
                    
                    if activeRequestIds.Length <> context.ActiveRequests.Length then
                        let cleanedContext = { updatedContext with ActiveRequests = activeRequestIds }
                        reasoningContexts.TryUpdate(contextId, cleanedContext, updatedContext) |> ignore
                
                | (false, _) -> ()
                
                // Wait before next monitoring cycle
                do! Async.Sleep(5000)  // Monitor every 5 seconds
        with
        | ex ->
            logger.LogError(ex, $"Error monitoring context {contextId}")
    }
    
    interface IRealTimeReasoningEngine with
        
        member this.ProcessStreamingAsync(request: RealTimeReasoningRequest) =
            async {
                try
                    logger.LogInformation($"Starting streaming reasoning for request: {request.RequestId}")
                    
                    // Register request
                    activeRequests.[request.RequestId] <- request
                    
                    // Create update channel
                    let channel = createUpdateChannel()
                    updateChannels.[request.RequestId] <- channel
                    
                    // Create cancellation token
                    let cts = new CancellationTokenSource()
                    if request.Deadline.IsSome then
                        let timeout = request.Deadline.Value - DateTime.UtcNow
                        if timeout > TimeSpan.Zero then
                            cts.CancelAfter(timeout)
                    
                    cancellationTokens.[request.RequestId] <- cts
                    
                    // Start reasoning process
                    let reasoningTask = processWithUpdates request cts.Token
                    
                    // Return async enumerable of updates
                    return seq {
                        let mutable isComplete = false
                        let mutable finalResult = None
                        let updates = ResizeArray<RealTimeReasoningUpdate>()
                        
                        while not isComplete do
                            try
                                // Try to read update from channel
                                let updateTask = channel.Reader.ReadAsync(cts.Token).AsTask()
                                let completed = updateTask.Wait(1000)  // 1 second timeout
                                
                                if completed && not updateTask.IsFaulted then
                                    let update = updateTask.Result
                                    updates.Add(update)
                                    
                                    // Check if reasoning is complete
                                    if update.UpdateType = "completed" || update.UpdateType = "error" || update.UpdateType = "cancelled" then
                                        isComplete <- true
                                        
                                        // Get final result if available
                                        if reasoningTask.IsCompleted && not reasoningTask.IsFaulted then
                                            finalResult <- reasoningTask.Result
                                
                                // Yield current state
                                yield {
                                    RequestId = request.RequestId
                                    IsComplete = isComplete
                                    CurrentStep = None  // TODO: Track current step
                                    IntermediateResult = if updates.Count > 0 then Some updates.[updates.Count - 1].Content else None
                                    FinalResult = finalResult
                                    Updates = updates |> Seq.toList
                                    ProcessingTime = DateTime.UtcNow - DateTime.UtcNow  // TODO: Track actual time
                                }
                                
                            with
                            | :? TimeoutException ->
                                // Continue if no update available
                                ()
                            | ex ->
                                logger.LogError(ex, $"Error in streaming reasoning for request {request.RequestId}")
                                isComplete <- true
                    } |> AsyncSeq.ofSeq
                    
                with
                | ex ->
                    logger.LogError(ex, $"Error setting up streaming reasoning for request {request.RequestId}")
                    return AsyncSeq.empty
            } |> Async.StartAsTask |> Async.AwaitTask |> AsyncSeq.ofAsyncEnum
        
        member this.HandleInterruptAsync(requestId: string) (newRequest: RealTimeReasoningRequest) = task {
            try
                logger.LogInformation($"Handling interrupt for request: {requestId}")
                
                // Cancel existing request
                match cancellationTokens.TryGetValue(requestId) with
                | (true, cts) -> 
                    cts.Cancel()
                    cancellationTokens.TryRemove(requestId) |> ignore
                | (false, _) -> ()
                
                // Clean up existing request
                activeRequests.TryRemove(requestId) |> ignore
                updateChannels.TryRemove(requestId) |> ignore
                
                // Start new request
                let! _ = this.ProcessStreamingAsync(newRequest) |> AsyncSeq.toListAsync
                ()
                
            with
            | ex ->
                logger.LogError(ex, $"Error handling interrupt for request {requestId}")
        }
        
        member this.MaintainContext(contextId: string) (context: ReasoningContext) = task {
            try
                logger.LogInformation($"Maintaining reasoning context: {contextId}")
                
                // Store or update context
                reasoningContexts.[contextId] <- context
                
                // Start context monitoring if not already running
                if not (reasoningContexts.ContainsKey($"{contextId}_monitor")) then
                    reasoningContexts.[$"{contextId}_monitor"] <- context
                    Async.Start(monitorContext contextId)
                
            with
            | ex ->
                logger.LogError(ex, $"Error maintaining context {contextId}")
        }
        
        member this.GetActiveRequests() =
            activeRequests.Values |> Seq.toList

/// Async sequence helper module
module AsyncSeq =
    let ofSeq (seq: seq<'T>) = 
        seq |> Seq.map async.Return |> AsyncSeq.ofSeq
    
    let ofAsyncEnum (asyncEnum: IAsyncEnumerable<'T>) =
        asyncEnum |> AsyncSeq.ofAsyncEnum

/// Factory for creating real-time reasoning engines
module RealTimeReasoningEngineFactory =
    
    let create 
        (chainEngine: IChainOfThoughtEngine)
        (budgetController: IDynamicBudgetController)
        (qualityMetrics: IReasoningQualityMetrics)
        (logger: ILogger<RealTimeReasoningEngine>) =
        new RealTimeReasoningEngine(chainEngine, budgetController, qualityMetrics, logger) :> IRealTimeReasoningEngine
