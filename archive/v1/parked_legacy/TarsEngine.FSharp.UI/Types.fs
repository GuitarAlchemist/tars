namespace TarsEngine.FSharp.UI

open System
open Fable.Core
open Thoth.Json

/// Core types for the TARS UI application
module Types =

    /// Application pages
    type Page =
        | Dashboard
        | Agents
        | Metascripts
        | Nodes
        | Chat
        | Editor of string

    /// Agent status enumeration
    type AgentStatus =
        | Idle
        | Active
        | Busy
        | Error of string
        | Offline

    /// Node status enumeration
    type NodeStatus =
        | Online
        | Offline
        | Degraded
        | Maintenance

    /// Chat message types
    type MessageType =
        | Text
        | Code
        | Error
        | System
        | AgentResponse

    /// Agent tree structure for hierarchical display
    type AgentTreeNode = {
        Id: string
        Name: string
        Type: string
        Status: AgentStatus
        Children: AgentTreeNode list
        Capabilities: string list
        LastActivity: DateTime option
        Metrics: Map<string, obj>
    }

    /// Agent information for UI
    type AgentInfo = {
        Id: string
        Name: string
        Type: string
        Status: AgentStatus
        Department: string
        Team: string option
        Capabilities: string list
        CurrentTask: string option
        Performance: float
        LastActivity: DateTime
    }

    /// Department information
    type DepartmentInfo = {
        Name: string
        Description: string
        Head: string
        Teams: string list
        AgentCount: int
        Status: string
    }

    /// Team information
    type TeamInfo = {
        Name: string
        Department: string
        Lead: string option
        Members: string list
        Objectives: string list
        Status: string
        Performance: float
    }

    /// Network node for visualization
    type NetworkNode = {
        Id: string
        Name: string
        Type: string // "core" | "department" | "team" | "agent"
        Position: float * float * float
        Connections: string list
        Status: AgentStatus
        Metadata: Map<string, obj>
    }

    /// Metascript information
    type MetascriptInfo = {
        Name: string
        Path: string
        Description: string option
        Author: string option
        Version: string option
        LastModified: DateTime
        IsRunning: bool
        ExecutionCount: int
        Tags: string list
    }

    /// Node information for monitoring
    type NodeInfo = {
        Id: string
        Name: string
        Address: string
        Status: NodeStatus
        RunningMetascripts: string list
        CpuUsage: float
        MemoryUsage: float
        LastHeartbeat: DateTime
        Capabilities: string list
    }

    /// Chat message structure
    type ChatMessage = {
        Id: string
        Content: string
        IsFromUser: bool
        Timestamp: DateTime
        MessageType: MessageType
        Metadata: Map<string, obj> option
        // Legacy fields for compatibility
        AgentId: string option
        AgentName: string option
    }

    /// UI generation request
    type UIGenerationRequest = {
        Prompt: string
        Requirements: string list
        TargetAgents: string list
        Priority: int
        Timestamp: DateTime
    }

    /// Generated UI component
    type GeneratedUIComponent = {
        Name: string
        Type: string
        Code: string
        Styling: string
        Dependencies: string list
        GeneratedBy: string list
        Timestamp: DateTime
        Status: string
    }

    /// Application state - comprehensive model combining both versions
    type Model = {
        // Page navigation
        CurrentPage: Page

        // Agent data (comprehensive)
        Agents: AgentInfo list
        AgentTree: AgentTreeNode list
        Departments: DepartmentInfo list
        Teams: TeamInfo list
        NetworkNodes: NetworkNode list

        // Metascripts and nodes
        Metascripts: MetascriptInfo list
        Nodes: NodeInfo list
        CurrentMetascript: string option
        MonacoEditorContent: string

        // UI state
        SelectedAgent: AgentInfo option
        SelectedAgentId: string option
        SelectedNode: NetworkNode option
        SelectedNodeId: string option
        ActiveChat: string option
        ChatMessages: ChatMessage list
        ChatHistory: ChatMessage list

        // UI generation
        GenerationPrompt: string
        GenerationInProgress: bool
        GeneratedComponents: GeneratedUIComponent list

        // Filters and search
        SearchQuery: string
        StatusFilter: AgentStatus option
        DepartmentFilter: string option

        // Real-time updates and connection
        ConnectionStatus: string
        WebSocketConnected: bool
        SemanticKernelReady: bool
        LastUpdate: DateTime

        // Loading and error handling
        IsLoading: bool
        Error: string option
        Errors: string list
    }

    /// Application messages - comprehensive combining both versions
    type Msg =
        // Navigation
        | NavigateTo of Page

        // Agent management
        | LoadAgents
        | AgentsLoaded of AgentInfo list
        | AgentTreeLoaded of AgentTreeNode list
        | AgentStatusChanged of string * AgentStatus
        | SelectAgent of AgentInfo
        | SelectAgentById of string
        | DeselectAgent

        // Department and team management
        | LoadDepartments
        | DepartmentsLoaded of DepartmentInfo list
        | LoadTeams
        | TeamsLoaded of TeamInfo list

        // Metascripts
        | LoadMetascripts
        | MetascriptsLoaded of MetascriptInfo list
        | SelectMetascript of string
        | RunMetascript of string
        | StopMetascript of string
        | SaveMetascript of string * string
        | UpdateMonacoContent of string

        // Nodes
        | LoadNodes
        | NodesLoaded of NodeInfo list
        | SelectNode of NetworkNode
        | SelectNodeById of string
        | DeselectNode
        | UpdateNodePosition of string * (float * float * float)

        // Chat and interaction
        | StartChat of string
        | EndChat
        | SendMessage of string
        | SendChatMessage of string
        | MessageReceived of ChatMessage
        | ChatMessageReceived of ChatMessage
        | ChatHistoryLoaded of ChatMessage list

        // UI generation
        | UpdateGenerationPrompt of string
        | StartUIGeneration
        | UIGenerationCompleted of GeneratedUIComponent
        | UIGenerationFailed of string

        // Search and filtering
        | UpdateSearchQuery of string
        | SetStatusFilter of AgentStatus option
        | SetDepartmentFilter of string option
        | ClearFilters

        // Real-time updates and connection
        | WebSocketConnected
        | WebSocketDisconnected
        | WebSocketMessage of string
        | UpdateReceived of string
        | SemanticKernelInitialized
        | Refresh

        // Error handling
        | AddError of string
        | ClearError of string
        | ClearAllErrors
        | Error of string
        | ClearError

        // System
        | Tick of DateTime
        | NoOp
    
    /// Command types for side effects
    type Cmd =
        | LoadAgentsCmd
        | LoadDepartmentsCmd
        | LoadTeamsCmd
        | LoadNetworkNodesCmd
        | ConnectWebSocketCmd
        | SendChatMessageCmd of string * string
        | GenerateUICmd of UIGenerationRequest
        | UpdateAgentStatusCmd of string * AgentStatus
    
    /// Initial model state - comprehensive combining both versions
    let init () : Model * Cmd list =
        {
            // Page navigation
            CurrentPage = Dashboard

            // Agent data (comprehensive)
            Agents = []
            AgentTree = []
            Departments = []
            Teams = []
            NetworkNodes = []

            // Metascripts and nodes
            Metascripts = []
            Nodes = []
            CurrentMetascript = None
            MonacoEditorContent = ""

            // UI state
            SelectedAgent = None
            SelectedAgentId = None
            SelectedNode = None
            SelectedNodeId = None
            ActiveChat = None
            ChatMessages = []
            ChatHistory = []

            // UI generation
            GenerationPrompt = ""
            GenerationInProgress = false
            GeneratedComponents = []

            // Filters and search
            SearchQuery = ""
            StatusFilter = None
            DepartmentFilter = None

            // Real-time updates and connection
            ConnectionStatus = "Disconnected"
            WebSocketConnected = false
            SemanticKernelReady = false
            LastUpdate = DateTime.Now

            // Loading and error handling
            IsLoading = false
            Error = None
            Errors = []
        }, [LoadAgentsCmd; LoadDepartmentsCmd; LoadTeamsCmd; LoadNetworkNodesCmd; ConnectWebSocketCmd]
    
    /// Sample data for development
    module SampleData =
        
        let sampleAgents = [
            {
                Id = "ui-dev-001"
                Name = "React UI Agent"
                Type = "UIAgent"
                Status = Active
                Department = "UI Development"
                Team = Some "UI Development Team"
                Capabilities = ["react"; "typescript"; "three.js"; "webgpu"]
                CurrentTask = Some "Generating network visualization component"
                Performance = 0.95
                LastActivity = DateTime.Now.AddMinutes(-2.0)
            }
            {
                Id = "design-001"
                Name = "Visual Design Agent"
                Type = "DesignAgent"
                Status = Active
                Department = "Design"
                Team = Some "Design Team"
                Capabilities = ["visual_design"; "branding"; "tars_theme"]
                CurrentTask = Some "Creating TARS visual specifications"
                Performance = 0.88
                LastActivity = DateTime.Now.AddMinutes(-1.0)
            }
            {
                Id = "ux-001"
                Name = "UX Research Agent"
                Type = "UXAgent"
                Status = Busy
                Department = "UX"
                Team = Some "UX Team"
                Capabilities = ["user_research"; "accessibility"; "interaction_design"]
                CurrentTask = Some "Conducting accessibility compliance review"
                Performance = 0.92
                LastActivity = DateTime.Now.AddMinutes(-0.5)
            }
            {
                Id = "ai-001"
                Name = "LLM Research Agent"
                Type = "AIAgent"
                Status = Active
                Department = "AI Research"
                Team = Some "AI Research Team"
                Capabilities = ["llm_optimization"; "model_analysis"; "performance_tuning"]
                CurrentTask = Some "Optimizing inference performance"
                Performance = 0.97
                LastActivity = DateTime.Now
            }
        ]
        
        let sampleDepartments = [
            {
                Name = "UI Development"
                Description = "User interface development and visualization"
                Head = "ui-dev-lead-001"
                Teams = ["UI Development Team"; "Design Team"; "UX Team"]
                AgentCount = 8
                Status = "Active"
            }
            {
                Name = "AI Research"
                Description = "Advanced AI research and development"
                Head = "ai-research-lead-001"
                Teams = ["AI Research Team"; "Innovation Team"]
                AgentCount = 6
                Status = "Active"
            }
            {
                Name = "Development"
                Description = "Core software development"
                Head = "dev-head-001"
                Teams = ["Core Engine Team"; "Infrastructure Team"]
                AgentCount = 12
                Status = "Active"
            }
        ]
        
        let sampleNetworkNodes = [
            {
                Id = "tars-core"
                Name = "TARS Core"
                Type = "core"
                Position = (0.0, 0.0, 0.0)
                Connections = ["ui-dev"; "ai-research"; "development"]
                Status = Active
                Metadata = Map.ofList [("version", "2.0" :> obj); ("uptime", "99.9%" :> obj)]
            }
            {
                Id = "ui-dev"
                Name = "UI Development"
                Type = "department"
                Position = (-200.0, 100.0, 0.0)
                Connections = ["tars-core"; "ui-dev-team"; "design-team"]
                Status = Active
                Metadata = Map.ofList [("agents", 8 :> obj)]
            }
            {
                Id = "ai-research"
                Name = "AI Research"
                Type = "department"
                Position = (200.0, 100.0, 0.0)
                Connections = ["tars-core"; "ai-research-team"]
                Status = Active
                Metadata = Map.ofList [("agents", 6 :> obj)]
            }
        ]
