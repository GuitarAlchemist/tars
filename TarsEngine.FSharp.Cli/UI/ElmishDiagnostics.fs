namespace TarsEngine.FSharp.Cli.UI

open System
open TarsEngine.FSharp.Cli.Diagnostics.TarsRealDiagnostics

/// HTML Virtual DOM for Elmish
module Html =
    type HtmlElement =
        | Element of string * (string * string) list * HtmlElement list
        | Text of string

    let div attrs children = Element("div", attrs, children)
    let h1 attrs children = Element("h1", attrs, children)
    let h2 attrs children = Element("h2", attrs, children)
    let h3 attrs children = Element("h3", attrs, children)
    let h4 attrs children = Element("h4", attrs, children)
    let p attrs children = Element("p", attrs, children)
    let span attrs children = Element("span", attrs, children)
    let button attrs children = Element("button", attrs, children)
    let input attrs = Element("input", attrs, [])
    let label attrs children = Element("label", attrs, children)
    let select attrs children = Element("select", attrs, children)
    let option attrs children = Element("option", attrs, children)
    let ul attrs children = Element("ul", attrs, children)
    let li attrs children = Element("li", attrs, children)
    let text str = Text(str)

/// Elmish Command helpers
module Cmd =
    type Cmd<'msg> = 'msg list

    let none : Cmd<'msg> = []
    let ofMsg (msg: 'msg) : Cmd<'msg> = [msg]

    module OfAsync =
        let perform (task: 'a -> Async<'b>) (arg: 'a) (ofSuccess: 'b -> 'msg) : Cmd<'msg> =
            // In a real implementation, this would execute the async and dispatch the result
            []

/// Elmish Subscription helpers
module Sub =
    type Sub<'msg> = unit

    let none : Sub<'msg> = ()
    let interval (ms: int) (msg: 'msg) : Sub<'msg> = ()

/// Elmish Program helpers
module Program =
    type Program<'model, 'msg> = {
        Init: unit -> 'model * Cmd<'msg>
        Update: 'msg -> 'model -> 'model * Cmd<'msg>
        View: 'model -> ('msg -> unit) -> Html.HtmlElement
        Subscriptions: 'model -> Sub<'msg>
    }

    let mkProgram init update view = {
        Init = init
        Update = update
        View = view
        Subscriptions = fun _ -> Sub.none
    }

    let withSubscription subs program = { program with Subscriptions = subs }

open Html

/// Pure Elmish MVU Architecture for TARS Diagnostics
module ElmishDiagnostics =

    // MODEL
    type DiagnosticsModel = {
        SystemHealth: ComprehensiveDiagnostics option
        IsLoading: bool
        Error: string option
        LastUpdate: DateTime
        AutoRefresh: bool
        RefreshInterval: int
        SelectedView: DiagnosticsView
        FilterLevel: HealthLevel
        ShowDetails: bool
    }

    and DiagnosticsView =
        | Overview
        | GpuDetails
        | GitHealth
        | NetworkStatus
        | SystemResources

    and HealthLevel =
        | All
        | HealthyOnly
        | ProblemsOnly

    // MESSAGES
    type Msg =
        | LoadDiagnostics
        | DiagnosticsLoaded of ComprehensiveDiagnostics
        | DiagnosticsError of string
        | ToggleAutoRefresh
        | SetRefreshInterval of int
        | ChangeView of DiagnosticsView
        | SetFilter of HealthLevel
        | ToggleDetails
        | Refresh
        | Tick

    // INIT
    let init () : DiagnosticsModel * Cmd<Msg> =
        let model = {
            SystemHealth = None
            IsLoading = true
            Error = None
            LastUpdate = DateTime.MinValue
            AutoRefresh = true
            RefreshInterval = 5000
            SelectedView = Overview
            FilterLevel = All
            ShowDetails = false
        }
        model, Cmd.ofMsg LoadDiagnostics

    // UPDATE
    let update (msg: Msg) (model: DiagnosticsModel) : DiagnosticsModel * Cmd<Msg> =
        match msg with
        | LoadDiagnostics ->
            { model with IsLoading = true; Error = None }, 
            Cmd.OfAsync.perform getComprehensiveDiagnostics Environment.CurrentDirectory DiagnosticsLoaded

        | DiagnosticsLoaded diagnostics ->
            { model with 
                SystemHealth = Some diagnostics
                IsLoading = false
                Error = None
                LastUpdate = DateTime.Now }, 
            Cmd.none

        | DiagnosticsError error ->
            { model with 
                IsLoading = false
                Error = Some error }, 
            Cmd.none

        | ToggleAutoRefresh ->
            { model with AutoRefresh = not model.AutoRefresh }, 
            Cmd.none

        | SetRefreshInterval interval ->
            { model with RefreshInterval = interval }, 
            Cmd.none

        | ChangeView view ->
            { model with SelectedView = view }, 
            Cmd.none

        | SetFilter level ->
            { model with FilterLevel = level }, 
            Cmd.none

        | ToggleDetails ->
            { model with ShowDetails = not model.ShowDetails }, 
            Cmd.none

        | Refresh ->
            update LoadDiagnostics model

        | Tick ->
            if model.AutoRefresh then
                update LoadDiagnostics model
            else
                model, Cmd.none

    // VIEW HELPERS
    let healthColor score =
        if score > 90.0 then "#28a745"
        elif score > 75.0 then "#ffc107"
        else "#dc3545"

    let healthIcon score =
        if score > 90.0 then "✅"
        elif score > 75.0 then "⚠️"
        else "❌"

    let formatBytes (bytes: int64) =
        let kb = bytes / 1024L
        let mb = kb / 1024L
        let gb = mb / 1024L
        if gb > 0L then sprintf "%d GB" gb
        elif mb > 0L then sprintf "%d MB" mb
        else sprintf "%d KB" kb

    // VIEW COMPONENTS
    let viewHeader (model: DiagnosticsModel) (dispatch: Msg -> unit) =
        div [ ("class", "diagnostics-header") ] [
            h1 [] [ text "🧠 TARS Real-time Diagnostics" ]
            div [ ("class", "header-controls") ] [
                button [ 
                    ("class", "btn btn-primary")
                    ("onclick", fun _ -> dispatch Refresh)
                ] [ text "🔄 Refresh" ]
                
                label [] [
                    input [ 
                        ("type", "checkbox")
                        ("checked", model.AutoRefresh.ToString().ToLower())
                        ("onchange", fun _ -> dispatch ToggleAutoRefresh)
                    ] []
                    text " Auto-refresh"
                ]
                
                select [
                    ("onchange", fun e -> 
                        let interval = Int32.Parse(e.target?value)
                        dispatch (SetRefreshInterval interval))
                ] [
                    option [ ("value", "1000") ] [ text "1s" ]
                    option [ ("value", "5000"); ("selected", "true") ] [ text "5s" ]
                    option [ ("value", "10000") ] [ text "10s" ]
                    option [ ("value", "30000") ] [ text "30s" ]
                ]
            ]
            
            match model.SystemHealth with
            | Some diagnostics ->
                div [ ("class", "health-summary") ] [
                    div [ 
                        ("class", "health-score")
                        ("style", sprintf "color: %s" (healthColor diagnostics.OverallHealthScore))
                    ] [
                        text (sprintf "%.1f%%" diagnostics.OverallHealthScore)
                    ]
                    div [ ("class", "last-update") ] [
                        text (sprintf "Last update: %s" (model.LastUpdate.ToString("HH:mm:ss")))
                    ]
                ]
            | None -> div [] []
        ]

    let viewNavigation (model: DiagnosticsModel) (dispatch: Msg -> unit) =
        div [ ("class", "diagnostics-nav") ] [
            ul [ ("class", "nav-list") ] [
                li [ 
                    ("class", if model.SelectedView = Overview then "nav-item active" else "nav-item")
                    ("onclick", fun _ -> dispatch (ChangeView Overview))
                ] [ text "🏠 Overview" ]
                
                li [ 
                    ("class", if model.SelectedView = GpuDetails then "nav-item active" else "nav-item")
                    ("onclick", fun _ -> dispatch (ChangeView GpuDetails))
                ] [ text "🎮 GPU" ]
                
                li [ 
                    ("class", if model.SelectedView = GitHealth then "nav-item active" else "nav-item")
                    ("onclick", fun _ -> dispatch (ChangeView GitHealth))
                ] [ text "📂 Git" ]
                
                li [ 
                    ("class", if model.SelectedView = NetworkStatus then "nav-item active" else "nav-item")
                    ("onclick", fun _ -> dispatch (ChangeView NetworkStatus))
                ] [ text "🌐 Network" ]
                
                li [ 
                    ("class", if model.SelectedView = SystemResources then "nav-item active" else "nav-item")
                    ("onclick", fun _ -> dispatch (ChangeView SystemResources))
                ] [ text "💻 System" ]
            ]
        ]

    let viewOverview (diagnostics: ComprehensiveDiagnostics) (dispatch: Msg -> unit) =
        div [ ("class", "overview-grid") ] [
            // Overall Health Card
            div [ ("class", "health-card") ] [
                h3 [] [ text "System Health" ]
                div [ 
                    ("class", "health-gauge")
                    ("style", sprintf "color: %s" (healthColor diagnostics.OverallHealthScore))
                ] [
                    text (sprintf "%.1f%%" diagnostics.OverallHealthScore)
                    text (healthIcon diagnostics.OverallHealthScore)
                ]
            ]
            
            // GPU Summary
            div [ ("class", "summary-card") ] [
                h4 [] [ text "🎮 GPU Status" ]
                for gpu in diagnostics.GpuInfo do
                    div [ ("class", "gpu-item") ] [
                        text gpu.Name
                        span [ ("class", "gpu-cuda") ] [
                            text (if gpu.CudaSupported then " ✅ CUDA" else " ❌ No CUDA")
                        ]
                    ]
            ]
            
            // Git Summary
            div [ ("class", "summary-card") ] [
                h4 [] [ text "📂 Repository" ]
                div [] [
                    text (if diagnostics.GitHealth.IsRepository then "✅ Git Repository" else "❌ Not a Git repo")
                ]
                div [] [
                    text (if diagnostics.GitHealth.IsClean then "✅ Clean" else "⚠️ Has changes")
                ]
                div [] [
                    text (sprintf "📝 %d commits" diagnostics.GitHealth.Commits)
                ]
            ]
            
            // Network Summary
            div [ ("class", "summary-card") ] [
                h4 [] [ text "🌐 Network" ]
                div [] [
                    text (if diagnostics.NetworkDiagnostics.IsConnected then "✅ Connected" else "❌ Disconnected")
                ]
                div [] [
                    text (sprintf "🕐 DNS: %dms" diagnostics.NetworkDiagnostics.DnsResolutionTime)
                ]
                match diagnostics.NetworkDiagnostics.PingLatency with
                | Some latency -> div [] [ text (sprintf "📡 Ping: %.1fms" latency) ]
                | None -> div [] []
            ]
        ]

    let viewGpuDetails (diagnostics: ComprehensiveDiagnostics) (dispatch: Msg -> unit) =
        div [ ("class", "gpu-details") ] [
            h2 [] [ text "🎮 GPU Information" ]
            if diagnostics.GpuInfo.IsEmpty then
                div [ ("class", "no-data") ] [ text "No GPU information available" ]
            else
                for gpu in diagnostics.GpuInfo do
                    div [ ("class", "gpu-card") ] [
                        h3 [] [ text gpu.Name ]
                        div [ ("class", "gpu-specs") ] [
                            div [] [ text (sprintf "Memory: %s" (formatBytes gpu.MemoryTotal)) ]
                            div [] [ text (sprintf "Used: %s" (formatBytes gpu.MemoryUsed)) ]
                            div [] [ text (sprintf "Free: %s" (formatBytes gpu.MemoryFree)) ]
                            div [] [ 
                                text "CUDA Support: "
                                span [ ("class", if gpu.CudaSupported then "status-good" else "status-bad") ] [
                                    text (if gpu.CudaSupported then "✅ Yes" else "❌ No")
                                ]
                            ]
                            match gpu.Temperature with
                            | Some temp -> div [] [ text (sprintf "🌡️ Temperature: %.1f°C" temp) ]
                            | None -> div [] []
                            match gpu.UtilizationGpu with
                            | Some util -> div [] [ text (sprintf "⚡ Utilization: %.1f%%" util) ]
                            | None -> div [] []
                        ]
                    ]
        ]

    let viewGitHealth (diagnostics: ComprehensiveDiagnostics) (dispatch: Msg -> unit) =
        div [ ("class", "git-details") ] [
            h2 [] [ text "📂 Git Repository Health" ]
            div [ ("class", "git-card") ] [
                div [ ("class", "git-status") ] [
                    div [] [
                        text "Repository: "
                        span [ ("class", if diagnostics.GitHealth.IsRepository then "status-good" else "status-bad") ] [
                            text (if diagnostics.GitHealth.IsRepository then "✅ Valid" else "❌ Invalid")
                        ]
                    ]
                    
                    if diagnostics.GitHealth.IsRepository then
                        div [] [
                            text "Status: "
                            span [ ("class", if diagnostics.GitHealth.IsClean then "status-good" else "status-warning") ] [
                                text (if diagnostics.GitHealth.IsClean then "✅ Clean" else "⚠️ Has changes")
                            ]
                        ]
                        div [] [ text (sprintf "📝 Total commits: %d" diagnostics.GitHealth.Commits) ]
                        div [] [ text (sprintf "📤 Unstaged changes: %d" diagnostics.GitHealth.UnstagedChanges) ]
                        div [] [ text (sprintf "📋 Staged changes: %d" diagnostics.GitHealth.StagedChanges) ]
                        
                        match diagnostics.GitHealth.CurrentBranch with
                        | Some branch -> div [] [ text (sprintf "🌿 Current branch: %s" branch) ]
                        | None -> div [] []
                        
                        match diagnostics.GitHealth.RemoteUrl with
                        | Some url -> div [] [ text (sprintf "🔗 Remote: %s" url) ]
                        | None -> div [] []
                ]
            ]
        ]

    let viewNetworkStatus (diagnostics: ComprehensiveDiagnostics) (dispatch: Msg -> unit) =
        div [ ("class", "network-details") ] [
            h2 [] [ text "🌐 Network Diagnostics" ]
            div [ ("class", "network-card") ] [
                div [] [
                    text "Connection: "
                    span [ ("class", if diagnostics.NetworkDiagnostics.IsConnected then "status-good" else "status-bad") ] [
                        text (if diagnostics.NetworkDiagnostics.IsConnected then "✅ Connected" else "❌ Disconnected")
                    ]
                ]
                
                match diagnostics.NetworkDiagnostics.PublicIpAddress with
                | Some ip -> div [] [ text (sprintf "🌍 Public IP: %s" ip) ]
                | None -> div [] []
                
                div [] [ text (sprintf "🕐 DNS Resolution: %dms" diagnostics.NetworkDiagnostics.DnsResolutionTime) ]
                
                match diagnostics.NetworkDiagnostics.PingLatency with
                | Some latency -> 
                    div [] [ 
                        text (sprintf "📡 Ping Latency: %.1fms" latency)
                        span [ ("class", if latency < 50.0 then "status-good" elif latency < 100.0 then "status-warning" else "status-bad") ] [
                            text (if latency < 50.0 then " ✅" elif latency < 100.0 then " ⚠️" else " ❌")
                        ]
                    ]
                | None -> div [] []
                
                div [] [ text (sprintf "🔗 Active connections: %d" diagnostics.NetworkDiagnostics.ActiveConnections) ]
                
                div [] [
                    text "Network interfaces: "
                    for iface in diagnostics.NetworkDiagnostics.NetworkInterfaces do
                        span [ ("class", "interface-tag") ] [ text iface ]
                ]
            ]
        ]

    let viewSystemResources (diagnostics: ComprehensiveDiagnostics) (dispatch: Msg -> unit) =
        div [ ("class", "system-details") ] [
            h2 [] [ text "💻 System Resources" ]
            div [ ("class", "system-grid") ] [
                div [ ("class", "resource-card") ] [
                    h4 [] [ text "🔥 CPU" ]
                    div [] [ text (sprintf "Usage: %.1f%%" diagnostics.SystemResources.CpuUsagePercent) ]
                    div [] [ text (sprintf "Cores: %d" diagnostics.SystemResources.CpuCoreCount) ]
                    div [] [ text (sprintf "Frequency: %d MHz" diagnostics.SystemResources.CpuFrequency) ]
                ]
                
                div [ ("class", "resource-card") ] [
                    h4 [] [ text "🧠 Memory" ]
                    div [] [ text (sprintf "Used: %s" (formatBytes diagnostics.SystemResources.MemoryUsedBytes)) ]
                    div [] [ text (sprintf "Total: %s" (formatBytes diagnostics.SystemResources.MemoryTotalBytes)) ]
                    div [] [ text (sprintf "Available: %s" (formatBytes diagnostics.SystemResources.MemoryAvailableBytes)) ]
                ]
                
                div [ ("class", "resource-card") ] [
                    h4 [] [ text "💾 Disk" ]
                    div [] [ text (sprintf "Used: %s" (formatBytes diagnostics.SystemResources.DiskUsedBytes)) ]
                    div [] [ text (sprintf "Total: %s" (formatBytes diagnostics.SystemResources.DiskTotalBytes)) ]
                    div [] [ text (sprintf "Free: %s" (formatBytes diagnostics.SystemResources.DiskFreeBytes)) ]
                ]
                
                div [ ("class", "resource-card") ] [
                    h4 [] [ text "⚙️ Processes" ]
                    div [] [ text (sprintf "Running: %d" diagnostics.SystemResources.ProcessCount) ]
                    div [] [ text (sprintf "Uptime: %.1f hours" (diagnostics.SystemResources.Uptime / 3600.0)) ]
                ]
            ]
        ]

    // MAIN VIEW
    let view (model: DiagnosticsModel) (dispatch: Msg -> unit) =
        div [ ("class", "elmish-diagnostics") ] [
            viewHeader model dispatch
            
            if model.IsLoading then
                div [ ("class", "loading") ] [
                    div [ ("class", "spinner") ] []
                    text "Loading diagnostics..."
                ]
            else
                match model.Error with
                | Some error ->
                    div [ ("class", "error") ] [
                        text (sprintf "❌ Error: %s" error)
                        button [ ("onclick", fun _ -> dispatch Refresh) ] [ text "Retry" ]
                    ]
                | None ->
                    match model.SystemHealth with
                    | Some diagnostics ->
                        div [ ("class", "diagnostics-content") ] [
                            viewNavigation model dispatch
                            div [ ("class", "main-content") ] [
                                match model.SelectedView with
                                | Overview -> viewOverview diagnostics dispatch
                                | GpuDetails -> viewGpuDetails diagnostics dispatch
                                | GitHealth -> viewGitHealth diagnostics dispatch
                                | NetworkStatus -> viewNetworkStatus diagnostics dispatch
                                | SystemResources -> viewSystemResources diagnostics dispatch
                            ]
                        ]
                    | None ->
                        div [ ("class", "no-data") ] [ text "No diagnostics data available" ]
        ]

    // SUBSCRIPTIONS
    let subscriptions (model: DiagnosticsModel) =
        if model.AutoRefresh then
            Sub.interval model.RefreshInterval Tick
        else
            Sub.none

    // PROGRAM
    let program =
        Program.mkProgram init update view
        |> Program.withSubscription subscriptions
