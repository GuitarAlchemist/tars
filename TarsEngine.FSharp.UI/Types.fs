namespace TarsEngine.FSharp.UI

open System
open Thoth.Json

/// Core types for the TARS UI application
module Types =
    
    /// Application state
    type Model = {
        CurrentPage: Page
        Agents: AgentTreeNode list
        Metascripts: MetascriptInfo list
        Nodes: NodeInfo list
        ChatHistory: ChatMessage list
        CurrentMetascript: string option
        IsLoading: bool
        Error: string option
        WebSocketConnected: bool
        SemanticKernelReady: bool
        MonacoEditorContent: string
        SelectedAgent: string option
        SelectedNode: string option
    }
    
    /// Application pages
    and Page =
        | Dashboard
        | Agents
        | Metascripts
        | Nodes
        | Chat
        | Editor of string
    
    /// Agent tree structure for hierarchical display
    and AgentTreeNode = {
        Id: string
        Name: string
        Type: string
        Status: AgentStatus
        Children: AgentTreeNode list
        Capabilities: string list
        LastActivity: DateTime option
        Metrics: Map<string, obj>
    }
    
    /// Agent status enumeration
    and AgentStatus =
        | Idle
        | Active
        | Busy
        | Error of string
        | Offline
    
    /// Metascript information
    and MetascriptInfo = {
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
    and NodeInfo = {
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
    
    /// Node status enumeration
    and NodeStatus =
        | Online
        | Offline
        | Degraded
        | Maintenance
    
    /// Chat message structure
    and ChatMessage = {
        Id: string
        Content: string
        IsFromUser: bool
        Timestamp: DateTime
        MessageType: MessageType
        Metadata: Map<string, obj> option
    }
    
    /// Chat message types
    and MessageType =
        | Text
        | Code
        | Error
        | System
        | AgentResponse
    
    /// Application messages/events
    type Msg =
        | NavigateTo of Page
        | LoadAgents
        | LoadMetascripts
        | LoadNodes
        | AgentsLoaded of AgentTreeNode list
        | MetascriptsLoaded of MetascriptInfo list
        | NodesLoaded of NodeInfo list
        | SendChatMessage of string
        | ChatMessageReceived of ChatMessage
        | SelectAgent of string
        | SelectNode of string
        | SelectMetascript of string
        | RunMetascript of string
        | StopMetascript of string
        | UpdateMonacoContent of string
        | SaveMetascript of string * string
        | WebSocketConnected
        | WebSocketDisconnected
        | WebSocketMessage of string
        | SemanticKernelInitialized
        | Error of string
        | ClearError
        | Refresh
