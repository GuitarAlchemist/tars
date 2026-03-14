module TarsFableUI.AgentTypes

open System
open TarsFableUI.Types

// Agent Communication Types
type AgentMessage = {
    Id: string
    FromAgent: string
    ToAgent: string option
    MessageType: string
    Content: obj
    Timestamp: DateTime
    Priority: MessagePriority
}

and MessagePriority =
    | Low
    | Normal
    | High
    | Critical

// UI Generation Agent Types
type UIArchitectRequest = {
    Requirements: string
    TargetComponents: string list
    LayoutPreferences: Map<string, obj>
    UserContext: Map<string, obj>
}

type UIArchitectResponse = {
    ComponentHierarchy: ComponentSpec
    LayoutStrategy: string
    StateStructure: Map<string, obj>
    Recommendations: string list
}

type ComponentGenerationRequest = {
    ComponentSpec: ComponentSpec
    ParentContext: Map<string, obj>
    StyleRequirements: string list
    InteractionPatterns: string list
}

type ComponentGenerationResponse = {
    FSharpCode: string
    ComponentName: string
    Dependencies: string list
    TestSuggestions: string list
}

type StyleGenerationRequest = {
    ComponentSpecs: ComponentSpec list
    ThemePreferences: Map<string, obj>
    ResponsiveRequirements: string list
    AccessibilityLevel: string
}

type StyleGenerationResponse = {
    TailwindClasses: Map<string, string list>
    CustomCSS: string
    ResponsiveBreakpoints: Map<string, string>
    AccessibilityFeatures: string list
}

type StateManagementRequest = {
    ComponentHierarchy: ComponentSpec
    DataFlow: Map<string, obj>
    UserInteractions: string list
    ExternalAPIs: string list
}

type StateManagementResponse = {
    ModelDefinition: string
    MessageTypes: string
    UpdateFunctions: string
    SubscriptionHandlers: string
}

// Agent Status and Coordination
type AgentTeamStatus = {
    UIArchitect: AgentWorkStatus
    ComponentGenerator: AgentWorkStatus
    StyleAgent: AgentWorkStatus
    StateManager: AgentWorkStatus
    IntegrationAgent: AgentWorkStatus
    QualityAgent: AgentWorkStatus
    LastCoordination: DateTime
    ActiveWorkflow: string option
}

and AgentWorkStatus = {
    Status: AgentStatus
    CurrentTask: string option
    Progress: float
    EstimatedCompletion: DateTime option
    LastOutput: DateTime option
}

// Real-time UI Generation Workflow
type UIGenerationWorkflow = {
    Id: string
    Request: UIGenerationRequest
    Status: WorkflowStatus
    Steps: WorkflowStep list
    StartTime: DateTime
    EstimatedCompletion: DateTime option
    Results: UIGenerationResults option
}

and UIGenerationRequest = {
    UserPrompt: string
    ComponentType: string
    TargetPage: string
    Constraints: Map<string, obj>
    Priority: MessagePriority
}

and WorkflowStatus =
    | Queued
    | InProgress
    | Completed
    | Failed of string
    | Cancelled

and WorkflowStep = {
    Agent: string
    Task: string
    Status: WorkflowStatus
    StartTime: DateTime option
    CompletionTime: DateTime option
    Output: obj option
}

and UIGenerationResults = {
    GeneratedComponents: string list
    CompiledCode: string
    StyleDefinitions: string
    TestResults: TestResult list
    PerformanceMetrics: Map<string, float>
}
