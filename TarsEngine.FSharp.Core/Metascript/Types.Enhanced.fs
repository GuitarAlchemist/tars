namespace TarsEngine.FSharp.Core.Metascript

open System
open System.Collections.Generic

/// <summary>
/// Enhanced types for the TARS metascript system
/// Provides comprehensive support for autonomous coding and exploration
/// </summary>
module Types =
    
    /// <summary>
    /// Execution context for metascript blocks
    /// </summary>
    type MetascriptExecutionContext = {
        Variables: Map<string, string>
        ProjectPath: string
        OutputPath: string
    }
    
    /// <summary>
    /// Result of executing a metascript block
    /// </summary>
    type MetascriptBlockResult = {
        Success: bool
        Output: string
        Variables: Map<string, string>
        Logs: string list
        ExecutionTime: TimeSpan
    }
    
    /// <summary>
    /// Types of metascript blocks supported
    /// </summary>
    type BlockType =
        | FSharp
        | Tars
        | Yaml
        | Action
        | Variable
        | Function
        | Describe
        | Config
        | LLM
        | Unknown of string
    
    /// <summary>
    /// A metascript block with its content and metadata
    /// </summary>
    type MetascriptBlock = {
        BlockType: BlockType
        Content: string
        LineNumber: int
        Parameters: Map<string, string>
    }
    
    /// <summary>
    /// Status of metascript execution
    /// </summary>
    type ExecutionStatus =
        | NotStarted
        | Running
        | Completed
        | Failed of string
        | Exploring
        | Recovering
    
    /// <summary>
    /// Exploration strategy for autonomous recovery
    /// </summary>
    type ExplorationStrategy =
        | DeepDive
        | AlternativeApproach
        | WebResearch
        | PatternMatching
        | UserConsultation
    
    /// <summary>
    /// Recovery action for stuck scenarios
    /// </summary>
    type RecoveryAction = {
        Strategy: ExplorationStrategy
        Description: string
        Implementation: string
        ExpectedOutcome: string
    }
    
    /// <summary>
    /// Autonomous coding task
    /// </summary>
    type AutonomousCodingTask = {
        TaskId: string
        Description: string
        Requirements: string list
        TechnologyStack: string option
        OutputPath: string
        Status: ExecutionStatus
        CreatedAt: DateTime
        CompletedAt: DateTime option
    }
    
    /// <summary>
    /// Project generation request
    /// </summary>
    type ProjectGenerationRequest = {
        ProjectType: string
        ProjectName: string
        Description: string
        TechnologyStack: string option
        Features: string list
        OutputDirectory: string
    }
    
    /// <summary>
    /// Code analysis result
    /// </summary>
    type CodeAnalysisResult = {
        FilePath: string
        Language: string
        LinesOfCode: int
        Complexity: int
        Issues: string list
        Suggestions: string list
        Quality: float
    }
    
    /// <summary>
    /// Code improvement suggestion
    /// </summary>
    type CodeImprovement = {
        FilePath: string
        LineNumber: int
        IssueType: string
        Description: string
        Suggestion: string
        Priority: int
        AutoFixable: bool
    }
    
    /// <summary>
    /// Memory session for project context
    /// </summary>
    type MemorySession = {
        SessionId: string
        ProjectPath: string
        CreatedAt: DateTime
        LastAccessed: DateTime
        Memories: Map<string, obj>
        VectorEmbeddings: float array option
    }
    
    /// <summary>
    /// Vector embedding for semantic search
    /// </summary>
    type VectorEmbedding = {
        Id: string
        Content: string
        Vector: float array
        Metadata: Map<string, string>
        CreatedAt: DateTime
    }
    
    /// <summary>
    /// YAML status structure
    /// </summary>
    type YamlStatus = {
        Phase: string
        Status: string
        Progress: float
        CurrentTask: string option
        ExplorationMode: bool
        RecoveryActions: RecoveryAction list
        LastUpdated: DateTime
        Comments: string list
    }
    
    /// <summary>
    /// Metascript execution session
    /// </summary>
    type MetascriptSession = {
        SessionId: string
        MetascriptPath: string
        Context: MetascriptExecutionContext
        Status: ExecutionStatus
        Blocks: MetascriptBlock list
        Results: MetascriptBlockResult list
        StartTime: DateTime
        EndTime: DateTime option
        MemorySession: MemorySession option
    }
    
    /// <summary>
    /// Technology detection result
    /// </summary>
    type TechnologyDetection = {
        PrimaryLanguage: string
        Framework: string option
        DatabaseType: string option
        FrontendTechnology: string option
        BackendTechnology: string option
        Confidence: float
        DetectedFeatures: string list
    }
    
    /// <summary>
    /// Autonomous generation result
    /// </summary>
    type AutonomousGenerationResult = {
        Success: bool
        ProjectPath: string
        GeneratedFiles: string list
        TechnologyStack: TechnologyDetection
        ExecutionTime: TimeSpan
        MemorySession: MemorySession
        Logs: string list
        Errors: string list
    }
    
    /// <summary>
    /// Exploration report
    /// </summary>
    type ExplorationReport = {
        SessionId: string
        TriggerCondition: string
        StrategiesAttempted: ExplorationStrategy list
        SuccessfulStrategy: ExplorationStrategy option
        Resolution: string option
        TimeSpent: TimeSpan
        LessonsLearned: string list
        CreatedAt: DateTime
    }
    
    /// <summary>
    /// Helper functions for working with enhanced types
    /// </summary>
    module Helpers =
        
        /// <summary>
        /// Creates a new execution context
        /// </summary>
        let createContext (projectPath: string) (outputPath: string) : MetascriptExecutionContext =
            {
                Variables = Map.empty
                ProjectPath = projectPath
                OutputPath = outputPath
            }
        
        /// <summary>
        /// Creates a successful block result
        /// </summary>
        let createSuccessResult (output: string) (executionTime: TimeSpan) : MetascriptBlockResult =
            {
                Success = true
                Output = output
                Variables = Map.empty
                Logs = []
                ExecutionTime = executionTime
            }
        
        /// <summary>
        /// Creates a failed block result
        /// </summary>
        let createFailureResult (error: string) : MetascriptBlockResult =
            {
                Success = false
                Output = error
                Variables = Map.empty
                Logs = [error]
                ExecutionTime = TimeSpan.Zero
            }
        
        /// <summary>
        /// Parses block type from string
        /// </summary>
        let parseBlockType (blockTypeStr: string) : BlockType =
            match blockTypeStr.ToUpper() with
            | "FSHARP" -> FSharp
            | "TARS" -> Tars
            | "YAML" -> Yaml
            | "ACTION" -> Action
            | "VARIABLE" -> Variable
            | "FUNCTION" -> Function
            | "DESCRIBE" -> Describe
            | "CONFIG" -> Config
            | "LLM" -> LLM
            | other -> Unknown other
