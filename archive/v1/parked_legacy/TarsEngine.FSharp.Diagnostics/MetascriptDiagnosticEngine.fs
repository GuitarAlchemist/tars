namespace TarsEngine.FSharp.Diagnostics

open System
open System.IO
open System.Text.Json
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Metascript.Types
open TarsEngine.FSharp.Metascript.Types

/// Comprehensive diagnostic trace for metascript execution
type MetascriptDiagnosticTrace = {
    TraceId: string
    MetascriptPath: string
    ProjectPath: string option
    StartTime: DateTime
    EndTime: DateTime option
    
    // Execution Flow
    ExecutionPhases: ExecutionPhase list
    BlockExecutions: BlockExecution list
    FunctionCalls: FunctionCall list
    
    // Component Analysis
    ComponentGeneration: ComponentGeneration list
    UIInteractions: UIInteraction list
    EventHandling: EventHandling list
    
    // System State
    MemoryUsage: MemorySnapshot list
    PerformanceMetrics: PerformanceMetric list
    ErrorEvents: ErrorEvent list
    
    // TARS-Specific
    VectorStoreOperations: VectorStoreOperation list
    AgentActivations: AgentActivation list
    MetascriptDependencies: MetascriptDependency list
    
    // Root Cause Analysis
    IssueAnalysis: IssueAnalysis option
    RecommendedFixes: RecommendedFix list
    
    // Metadata
    Environment: Map<string, string>
    Configuration: Map<string, obj>
    Metadata: Map<string, obj>
}

and ExecutionPhase = {
    PhaseId: string
    Name: string
    StartTime: DateTime
    EndTime: DateTime option
    Status: ExecutionStatus
    SubPhases: ExecutionPhase list
    Outputs: string list
    Errors: string list
}

and BlockExecution = {
    BlockId: string
    BlockType: string
    Content: string
    StartTime: DateTime
    EndTime: DateTime option
    Status: ExecutionStatus
    Output: string option
    Error: string option
    Variables: Map<string, obj>
    Dependencies: string list
}

and FunctionCall = {
    FunctionName: string
    Module: string
    Parameters: Map<string, obj>
    ReturnValue: obj option
    StartTime: DateTime
    EndTime: DateTime option
    StackTrace: string list
    Success: bool
}

and ComponentGeneration = {
    ComponentId: string
    ComponentType: string
    GenerationTime: DateTime
    Success: bool
    Properties: Map<string, obj>
    Dependencies: string list
    RenderAttempts: int
    LastError: string option
}

and UIInteraction = {
    InteractionId: string
    EventType: string
    TargetElement: string
    Timestamp: DateTime
    Success: bool
    Error: string option
    UserAction: string option
}

and EventHandling = {
    EventId: string
    EventType: string
    Handler: string
    Timestamp: DateTime
    Processed: bool
    Error: string option
    PropagationStopped: bool
}

and MemorySnapshot = {
    Timestamp: DateTime
    TotalMemoryMB: float
    UsedMemoryMB: float
    GCCollections: int
    LargeObjectHeapMB: float
}

and PerformanceMetric = {
    MetricName: string
    Value: float
    Unit: string
    Timestamp: DateTime
    Category: string
}

and ErrorEvent = {
    ErrorId: string
    Timestamp: DateTime
    Severity: ErrorSeverity
    Message: string
    StackTrace: string option
    Context: Map<string, obj>
    Component: string option
}

and VectorStoreOperation = {
    OperationId: string
    OperationType: string
    Timestamp: DateTime
    Duration: TimeSpan
    Success: bool
    DocumentCount: int option
    QueryText: string option
    Results: int option
}

and AgentActivation = {
    AgentId: string
    AgentType: string
    ActivationTime: DateTime
    DeactivationTime: DateTime option
    TasksExecuted: int
    Success: bool
    Outputs: string list
}

and MetascriptDependency = {
    DependencyName: string
    DependencyType: string
    Version: string option
    LoadTime: DateTime
    LoadSuccess: bool
    Error: string option
}

and IssueAnalysis = {
    IssueId: string
    IssueType: string
    Severity: IssueSeverity
    Description: string
    RootCause: string
    AffectedComponents: string list
    Timeline: DateTime list
    Evidence: Evidence list
}

and RecommendedFix = {
    FixId: string
    Priority: FixPriority
    Description: string
    Implementation: string
    EstimatedEffort: string
    Dependencies: string list
    RiskLevel: RiskLevel
}

and Evidence = {
    EvidenceType: string
    Description: string
    Data: obj
    Timestamp: DateTime
    Confidence: float
}

and ExecutionStatus = Running | Completed | Failed | Cancelled
and ErrorSeverity = Info | Warning | Error | Critical
and IssueSeverity = Low | Medium | High | Critical
and FixPriority = Low | Medium | High | Critical
and RiskLevel = Low | Medium | High

/// Metascript Diagnostic Engine
type MetascriptDiagnosticEngine(logger: ILogger<MetascriptDiagnosticEngine>) =
    
    let mutable currentTrace: MetascriptDiagnosticTrace option = None
    let tracesDirectory = ".tars/traces"
    
    /// Ensure traces directory exists
    member private _.EnsureTracesDirectory() =
        if not (Directory.Exists(tracesDirectory)) then
            Directory.CreateDirectory(tracesDirectory) |> ignore
    
    /// Start diagnostic trace for metascript execution
    member this.StartTrace(metascriptPath: string, ?projectPath: string) =
        this.EnsureTracesDirectory()
        
        let traceId = Guid.NewGuid().ToString("N")[..7]
        let trace = {
            TraceId = traceId
            MetascriptPath = metascriptPath
            ProjectPath = projectPath
            StartTime = DateTime.UtcNow
            EndTime = None
            
            ExecutionPhases = []
            BlockExecutions = []
            FunctionCalls = []
            
            ComponentGeneration = []
            UIInteractions = []
            EventHandling = []
            
            MemoryUsage = []
            PerformanceMetrics = []
            ErrorEvents = []
            
            VectorStoreOperations = []
            AgentActivations = []
            MetascriptDependencies = []
            
            IssueAnalysis = None
            RecommendedFixes = []
            
            Environment = Environment.GetEnvironmentVariables() 
                         |> Seq.cast<System.Collections.DictionaryEntry>
                         |> Seq.map (fun kv -> kv.Key.ToString(), kv.Value.ToString())
                         |> Map.ofSeq
            Configuration = Map.empty
            Metadata = Map.empty
        }
        
        currentTrace <- Some trace
        logger.LogInformation("🔍 Started diagnostic trace {TraceId} for {MetascriptPath}", traceId, metascriptPath)
        traceId
    
    /// Record execution phase
    member this.RecordPhase(name: string, status: ExecutionStatus, ?outputs: string list, ?errors: string list) =
        match currentTrace with
        | Some trace ->
            let phase = {
                PhaseId = Guid.NewGuid().ToString("N")[..7]
                Name = name
                StartTime = DateTime.UtcNow
                EndTime = if status = Completed || status = Failed then Some DateTime.UtcNow else None
                Status = status
                SubPhases = []
                Outputs = outputs |> Option.defaultValue []
                Errors = errors |> Option.defaultValue []
            }
            
            let updatedTrace = { trace with ExecutionPhases = phase :: trace.ExecutionPhases }
            currentTrace <- Some updatedTrace
            
        | None -> logger.LogWarning("No active trace to record phase: {PhaseName}", name)
    
    /// Record block execution
    member this.RecordBlockExecution(blockType: string, content: string, status: ExecutionStatus, ?output: string, ?error: string, ?variables: Map<string, obj>) =
        match currentTrace with
        | Some trace ->
            let blockExecution = {
                BlockId = Guid.NewGuid().ToString("N")[..7]
                BlockType = blockType
                Content = content
                StartTime = DateTime.UtcNow
                EndTime = if status = Completed || status = Failed then Some DateTime.UtcNow else None
                Status = status
                Output = output
                Error = error
                Variables = variables |> Option.defaultValue Map.empty
                Dependencies = []
            }
            
            let updatedTrace = { trace with BlockExecutions = blockExecution :: trace.BlockExecutions }
            currentTrace <- Some updatedTrace
            
        | None -> logger.LogWarning("No active trace to record block execution")
    
    /// Record component generation
    member this.RecordComponentGeneration(componentId: string, componentType: string, success: bool, ?properties: Map<string, obj>, ?error: string) =
        match currentTrace with
        | Some trace ->
            let componentGen = {
                ComponentId = componentId
                ComponentType = componentType
                GenerationTime = DateTime.UtcNow
                Success = success
                Properties = properties |> Option.defaultValue Map.empty
                Dependencies = []
                RenderAttempts = 1
                LastError = error
            }
            
            let updatedTrace = { trace with ComponentGeneration = componentGen :: trace.ComponentGeneration }
            currentTrace <- Some updatedTrace
            
        | None -> logger.LogWarning("No active trace to record component generation")
    
    /// Record UI interaction
    member this.RecordUIInteraction(eventType: string, targetElement: string, success: bool, ?error: string, ?userAction: string) =
        match currentTrace with
        | Some trace ->
            let interaction = {
                InteractionId = Guid.NewGuid().ToString("N")[..7]
                EventType = eventType
                TargetElement = targetElement
                Timestamp = DateTime.UtcNow
                Success = success
                Error = error
                UserAction = userAction
            }
            
            let updatedTrace = { trace with UIInteractions = interaction :: trace.UIInteractions }
            currentTrace <- Some updatedTrace
            
        | None -> logger.LogWarning("No active trace to record UI interaction")
    
    /// Record error event
    member this.RecordError(severity: ErrorSeverity, message: string, ?stackTrace: string, ?context: Map<string, obj>, ?component: string) =
        match currentTrace with
        | Some trace ->
            let errorEvent = {
                ErrorId = Guid.NewGuid().ToString("N")[..7]
                Timestamp = DateTime.UtcNow
                Severity = severity
                Message = message
                StackTrace = stackTrace
                Context = context |> Option.defaultValue Map.empty
                Component = component
            }
            
            let updatedTrace = { trace with ErrorEvents = errorEvent :: trace.ErrorEvents }
            currentTrace <- Some updatedTrace
            
        | None -> logger.LogWarning("No active trace to record error")
    
    /// Record performance metric
    member this.RecordPerformanceMetric(metricName: string, value: float, unit: string, category: string) =
        match currentTrace with
        | Some trace ->
            let metric = {
                MetricName = metricName
                Value = value
                Unit = unit
                Timestamp = DateTime.UtcNow
                Category = category
            }
            
            let updatedTrace = { trace with PerformanceMetrics = metric :: trace.PerformanceMetrics }
            currentTrace <- Some updatedTrace
            
        | None -> logger.LogWarning("No active trace to record performance metric")

    /// End trace and perform comprehensive analysis
    member this.EndTrace() =
        task {
            match currentTrace with
            | Some trace ->
                let endTime = DateTime.UtcNow
                let completedTrace = { trace with EndTime = Some endTime }

                // Perform comprehensive issue analysis
                let! issueAnalysis = this.AnalyzeIssues(completedTrace)
                let! recommendedFixes = this.GenerateRecommendedFixes(issueAnalysis)

                let finalTrace = {
                    completedTrace with
                        IssueAnalysis = Some issueAnalysis
                        RecommendedFixes = recommendedFixes
                }

                // Save trace to file
                let! traceFile = this.SaveTrace(finalTrace)

                // Generate diagnostic report
                let! reportFile = this.GenerateDiagnosticReport(finalTrace)

                currentTrace <- None
                logger.LogInformation("🔍 Completed diagnostic trace {TraceId}. Files: {TraceFile}, {ReportFile}",
                                     finalTrace.TraceId, traceFile, reportFile)

                return (finalTrace, traceFile, reportFile)

            | None ->
                logger.LogWarning("No active trace to end")
                return failwith "No active trace"
        }

    /// Analyze issues from trace data - ENHANCED WITH TARS INTELLIGENCE DETECTION
    member private this.AnalyzeIssues(trace: MetascriptDiagnosticTrace) =
        task {
            logger.LogInformation("🔍 Analyzing issues from trace data with TARS intelligence...")

            let errors = trace.ErrorEvents
            let failedComponents = trace.ComponentGeneration |> List.filter (fun c -> not c.Success)
            let successfulComponents = trace.ComponentGeneration |> List.filter (fun c -> c.Success)
            let failedInteractions = trace.UIInteractions |> List.filter (fun i -> not i.Success)
            let failedBlocks = trace.BlockExecutions |> List.filter (fun b -> b.Status = Failed)

            // SMART TARS: Analyze component generation intelligence
            let componentIssues =
                if successfulComponents.Length <= 5 then
                    // TARS detected limited component generation - analyze the intelligence layer
                    let rootCause =
                        if successfulComponents.Length = 5 then
                            "TARS Intelligence Issue: Component generation algorithm has hardcoded limit or insufficient scaling logic in evolveUIBasedOnContent function. The TARS introspection engines (analyzeTarsStructure, introspectVectorStore) are not providing sufficient complexity data to drive UI evolution."
                        elif successfulComponents.Length < 3 then
                            "TARS Intelligence Issue: Severe limitation in TARS autonomous analysis - vector store introspection and metascript analysis engines are not discovering sufficient TARS capabilities to drive component generation."
                        else
                            "TARS Intelligence Issue: Suboptimal component scaling - TARS feedback loop between introspection engines and UI evolution needs enhancement to reflect actual system complexity."

                    Some {
                        IssueId = "TARS_INTELLIGENCE_LIMITATION"
                        IssueType = "TarsIntelligenceScaling"
                        Severity = High
                        Description = sprintf "TARS generated only %d components. Intelligent system should scale based on actual codebase complexity, metascript count, and vector store size." successfulComponents.Length
                        RootCause = rootCause
                        AffectedComponents = successfulComponents |> List.map (fun c -> c.ComponentId)
                        Timeline = successfulComponents |> List.map (fun c -> c.GenerationTime)
                        Evidence = [
                            {
                                EvidenceType = "TarsIntelligenceAnalysis"
                                Description = sprintf "TARS autonomous analysis engines need enhancement - only %d components generated despite rich codebase" successfulComponents.Length
                                Data = box {|
                                    ComponentCount = successfulComponents.Length
                                    ExpectedMinimum = 8
                                    IntelligenceGap = "TARS introspection engines not reflecting actual system complexity"
                                |}
                                Timestamp = DateTime.UtcNow
                                Confidence = 0.95
                            }
                        ]
                    }
                elif failedComponents.Length > 0 then
                    let affectedComponents = failedComponents |> List.map (fun c -> c.ComponentId)
                    let rootCause =
                        if failedComponents |> List.exists (fun c -> c.LastError.IsSome && c.LastError.Value.Contains("render")) then
                            "Rendering pipeline failure - components generated but failed to render in DOM"
                        elif failedComponents |> List.exists (fun c -> c.LastError.IsSome && c.LastError.Value.Contains("event")) then
                            "Event binding failure - components created but event handlers not properly attached"
                        else
                            "Component generation pipeline failure - unknown cause"

                    Some {
                        IssueId = "COMP_GEN_FAILURE"
                        IssueType = "ComponentGeneration"
                        Severity = High
                        Description = sprintf "Component generation failed after %d successful components." successfulComponents.Length
                        RootCause = rootCause
                        AffectedComponents = affectedComponents
                        Timeline = failedComponents |> List.map (fun c -> c.GenerationTime)
                        Evidence = [
                            {
                                EvidenceType = "FailedComponents"
                                Description = sprintf "%d components failed to generate properly" failedComponents.Length
                                Data = box failedComponents
                                Timestamp = DateTime.UtcNow
                                Confidence = 0.9
                            }
                        ]
                    }
                else None

            // Analyze UI interaction issues
            let interactionIssues =
                if failedInteractions.Length > 0 then
                    Some {
                        IssueId = "UI_INTERACTION_FAILURE"
                        IssueType = "UIInteraction"
                        Severity = High
                        Description = "Button clicks and UI interactions not working properly"
                        RootCause = "Event handlers not properly bound to DOM elements or Elmish message dispatch not working"
                        AffectedComponents = failedInteractions |> List.map (fun i -> i.TargetElement)
                        Timeline = failedInteractions |> List.map (fun i -> i.Timestamp)
                        Evidence = [
                            {
                                EvidenceType = "FailedInteractions"
                                Description = sprintf "%d UI interactions failed" failedInteractions.Length
                                Data = box failedInteractions
                                Timestamp = DateTime.UtcNow
                                Confidence = 0.85
                            }
                        ]
                    }
                else None

            // Return the most critical issue
            return componentIssues |> Option.orElse interactionIssues |> Option.defaultValue {
                IssueId = "EXECUTION_INCOMPLETE"
                IssueType = "ExecutionFlow"
                Severity = Medium
                Description = "Metascript execution appears to have stopped prematurely"
                RootCause = "Unknown - requires deeper analysis of execution flow"
                AffectedComponents = []
                Timeline = []
                Evidence = []
            }
        }

    /// Generate recommended fixes - ENHANCED FOR TARS INTELLIGENCE ISSUES
    member private this.GenerateRecommendedFixes(issueAnalysis: IssueAnalysis) =
        task {
            logger.LogInformation("🔧 Generating TARS-intelligent recommended fixes...")

            let fixes =
                match issueAnalysis.IssueType with
                | "TarsIntelligenceScaling" -> [
                    {
                        FixId = "ENHANCE_TARS_INTROSPECTION_ENGINES"
                        Priority = Critical
                        Description = "Enhance TARS autonomous introspection engines to provide richer data"
                        Implementation = """
1. Modify analyzeTarsStructure() to scan actual codebase files for realistic complexity
2. Update introspectVectorStore() to calculate document count from actual repository
3. Enhance parseAllMetascripts() to discover real metascript patterns
4. Implement smart scaling in evolveUIBasedOnContent() based on multiple TARS factors
5. Add feedback loop between introspection results and UI evolution parameters"""
                        EstimatedEffort = "2-3 hours"
                        Dependencies = ["System.IO"; "Directory scanning"]
                        RiskLevel = Low
                    }
                    {
                        FixId = "IMPROVE_COMPONENT_SCALING_ALGORITHM"
                        Priority = High
                        Description = "Replace hardcoded component limits with intelligent scaling"
                        Implementation = """
1. Replace 'max 3 (tarsStructure.MetascriptCount / 3)' with dynamic calculation
2. Add baseComponents = max 6 tarsStructure.MetascriptCount
3. Add vectorComponents = max 2 (vectorStore.DocumentCount / 800)
4. Add agentComponents based on discovered agent definitions
5. Implement reasonable upper limit (e.g., min 25 totalComponents)"""
                        EstimatedEffort = "1-2 hours"
                        Dependencies = []
                        RiskLevel = Low
                    }
                    {
                        FixId = "ADD_TARS_FEEDBACK_LOOP"
                        Priority = High
                        Description = "Implement TARS self-improvement feedback loop"
                        Implementation = """
1. Add performance metrics collection in UI evolution
2. Implement success rate tracking for component generation
3. Add adaptive scaling based on previous generation results
4. Create TARS learning mechanism to optimize future generations
5. Implement diagnostic trace analysis for continuous improvement"""
                        EstimatedEffort = "3-4 hours"
                        Dependencies = ["Performance monitoring"]
                        RiskLevel = Medium
                    }
                ]
                | "ComponentGeneration" -> [
                    {
                        FixId = "FIX_COMPONENT_RENDERING"
                        Priority = High
                        Description = "Fix component rendering pipeline"
                        Implementation = """
1. Verify Elmish view function is properly structured
2. Check that all components are properly added to the DOM
3. Ensure React/Fable.React bindings are working
4. Add error boundaries around component rendering
5. Implement proper component lifecycle management"""
                        EstimatedEffort = "2-4 hours"
                        Dependencies = ["Fable.React"; "Elmish"]
                        RiskLevel = Medium
                    }
                ]
                | "UIInteraction" -> [
                    {
                        FixId = "FIX_EVENT_BINDING"
                        Priority = Critical
                        Description = "Fix button click event handlers"
                        Implementation = """
1. Verify Elmish message dispatch is properly configured
2. Check that OnClick handlers are correctly bound
3. Ensure event propagation is not being stopped
4. Add debugging to track message flow
5. Verify update function handles all message types"""
                        EstimatedEffort = "1-3 hours"
                        Dependencies = ["Elmish"]
                        RiskLevel = Low
                    }
                    {
                        FixId = "FIX_DOM_BINDING"
                        Priority = High
                        Description = "Fix DOM element binding"
                        Implementation = """
1. Ensure components are properly mounted to DOM
2. Check for timing issues in DOM manipulation
3. Verify CSS selectors and element IDs
4. Add proper error handling for DOM operations
5. Implement proper component cleanup"""
                        EstimatedEffort = "2-3 hours"
                        Dependencies = ["Browser.Dom"]
                        RiskLevel = Medium
                    }
                ]
                | _ -> [
                    {
                        FixId = "FIX_GENERAL_EXECUTION"
                        Priority = Medium
                        Description = "General execution flow improvements"
                        Implementation = """
1. Add comprehensive logging throughout execution
2. Implement proper error handling and recovery
3. Add performance monitoring and optimization
4. Ensure proper resource cleanup
5. Implement graceful degradation for failures"""
                        EstimatedEffort = "3-5 hours"
                        Dependencies = []
                        RiskLevel = Low
                    }
                ]

            return fixes
        }

    /// Save trace to JSON file
    member private this.SaveTrace(trace: MetascriptDiagnosticTrace) =
        task {
            let timestamp = DateTime.UtcNow.ToString("yyyy-MM-dd_HH-mm-ss")
            let fileName = sprintf "trace_%s_%s.json" trace.TraceId timestamp
            let filePath = Path.Combine(tracesDirectory, fileName)

            let options = JsonSerializerOptions(WriteIndented = true)
            let json = JsonSerializer.Serialize(trace, options)

            do! File.WriteAllTextAsync(filePath, json)
            logger.LogInformation("💾 Saved diagnostic trace to {FilePath}", filePath)

            return filePath
        }
