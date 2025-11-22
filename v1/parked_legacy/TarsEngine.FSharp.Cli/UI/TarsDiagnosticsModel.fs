namespace TarsEngine.FSharp.Cli.UI

open System

/// TARS Diagnostics Model Types and Data Structures
module TarsDiagnosticsModel =

    // TARS-SPECIFIC MODEL
    type TarsSubsystem = {
        Name: string
        Status: SubsystemStatus
        HealthPercentage: float
        ActiveComponents: int
        ProcessingRate: float
        MemoryUsage: int64
        LastActivity: DateTime
        Dependencies: string list
        Metrics: Map<string, obj>
    }

    and SubsystemStatus = 
        | Operational 
        | Degraded 
        | Critical 
        | Offline
        | Evolving

    type TarsDiagnosticsModel = {
        // ALL TARS SUBSYSTEMS - Comprehensive List
        AllSubsystems: TarsSubsystem list

        // System State
        OverallTarsHealth: float
        ActiveAgents: int
        ProcessingTasks: int
        IsLoading: bool
        Error: string option
        LastUpdate: DateTime

        // UI State
        SelectedSubsystem: string option
        ShowDetails: bool
        ViewMode: ViewMode
        AutoRefresh: bool
    }

    and ViewMode = 
        | Overview 
        | Detailed 
        | Performance 
        | Architecture

    // TARS-SPECIFIC MESSAGES
    type TarsMsg =
        | LoadTarsSubsystems
        | TarsSubsystemsLoaded of TarsSubsystem list
        | TarsError of string
        | SelectSubsystem of string
        | ToggleDetails
        | ChangeViewMode of ViewMode
        | ToggleAutoRefresh
        | RefreshTars
        | SubsystemStatusChanged of string * SubsystemStatus
        | NavigateToHome
        | NavigateToView of ViewMode
        | ClearSubsystemSelection

    // INIT - Pure function to create TARS model
    let init () : TarsDiagnosticsModel =
        {
            AllSubsystems = []
            OverallTarsHealth = 0.0
            ActiveAgents = 0
            ProcessingTasks = 0
            IsLoading = true
            Error = None
            LastUpdate = DateTime.MinValue
            SelectedSubsystem = None
            ShowDetails = false
            ViewMode = Overview
            AutoRefresh = true
        }

    // PURE VIEW HELPERS
    let statusColor = function
        | Operational -> "#28a745"
        | Degraded -> "#ffc107" 
        | Critical -> "#dc3545"
        | Offline -> "#6c757d"
        | Evolving -> "#17a2b8"

    let statusIcon = function
        | Operational -> "✅"
        | Degraded -> "⚠️"
        | Critical -> "❌"
        | Offline -> "⭕"
        | Evolving -> "🔄"

    let formatMetric (value: obj) =
        match value with
        | :? int as i -> sprintf "%d" i
        | :? float as f -> sprintf "%.2f" f
        | :? string as s -> s
        | _ -> sprintf "%A" value
