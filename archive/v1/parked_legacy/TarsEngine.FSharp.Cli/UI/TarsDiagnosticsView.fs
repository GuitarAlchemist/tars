namespace TarsEngine.FSharp.Cli.UI

open TarsHtml
open TarsDiagnosticsModel

/// TARS Diagnostics View Components
module TarsDiagnosticsView =

    // ENHANCED BREADCRUMB NAVIGATION COMPONENT - Real Elmish with Dynamic State
    let viewBreadcrumbs (model: TarsDiagnosticsModel) (dispatch: TarsMsg -> unit) =
        let viewModeText = function
            | Overview -> "Subsystems Overview"
            | Detailed -> "Detailed Analysis"
            | Performance -> "Performance Metrics"
            | Architecture -> "System Architecture"

        let viewModeIcon = function
            | Overview -> "🔍"
            | Detailed -> "📊"
            | Performance -> "⚡"
            | Architecture -> "🏗️"

        div [ className "tars-breadcrumbs" ] [
            // Breadcrumb status indicator
            div [ className "breadcrumb-status" ] [
                span [ className "status-icon" ] [
                    text (match model.SelectedSubsystem with
                          | Some _ -> "🔍"
                          | None -> "🏠")
                ]
                span [ className "status-text" ] [
                    text (match model.SelectedSubsystem with
                          | Some name -> sprintf "Viewing %s Details" name
                          | None -> sprintf "Browsing %s" (viewModeText model.ViewMode))
                ]
            ]
            div [ className "breadcrumb-container" ] [
                // Home breadcrumb - Always clickable with REAL JavaScript
                span [
                    className (if model.ViewMode = Overview && model.SelectedSubsystem.IsNone then "breadcrumb-item home current" else "breadcrumb-item home")
                    onClick "dispatch('ClearSubsystemSelection')"
                ] [
                    text "🧠 TARS Diagnostics"
                ]

                // View mode breadcrumb (always show current view)
                span [ className "breadcrumb-separator" ] [ text " > " ]
                span [
                    className (if model.SelectedSubsystem.IsNone then "breadcrumb-item view current" else "breadcrumb-item view")
                    onClick "dispatch('ClearSubsystemSelection')"
                ] [
                    text (sprintf "%s %s" (viewModeIcon model.ViewMode) (viewModeText model.ViewMode))
                ]

                // Subsystem breadcrumb (if selected)
                match model.SelectedSubsystem with
                | Some subsystemName ->
                    span [ className "breadcrumb-separator" ] [ text " > " ]
                    span [ className "breadcrumb-item subsystem current" ] [
                        text (sprintf "🔧 %s" subsystemName)
                    ]
                    // Add close button for subsystem with REAL JavaScript
                    span [
                        className "breadcrumb-close"
                        onClick "dispatch('ClearSubsystemSelection')"
                    ] [
                        text " ✕"
                    ]
                | None -> empty
            ]
        ]

    // ELMISH VIEW COMPONENTS - Pure functions with Real Message Dispatching
    let viewTarsHeader (model: TarsDiagnosticsModel) (dispatch: TarsMsg -> unit) =
        div [ className "tars-header" ] [
            h1 [] [ text "🧠 TARS Subsystem Diagnostics" ]
            div [ className "tars-health-summary" ] [
                div [ className "overall-health" ] [
                    span [ className "health-score" ] [
                        text (sprintf "%.1f%%" model.OverallTarsHealth)
                    ]
                    span [ className "health-label" ] [ text "TARS Health" ]
                ]
                div [ className "tars-stats" ] [
                    div [ className "stat" ] [
                        span [ className "stat-value" ] [ text (string model.ActiveAgents) ]
                        span [ className "stat-label" ] [ text "Active Agents" ]
                    ]
                    div [ className "stat" ] [
                        span [ className "stat-value" ] [ text (string model.ProcessingTasks) ]
                        span [ className "stat-label" ] [ text "Processing Tasks" ]
                    ]
                    div [ className "stat" ] [
                        span [ className "stat-value" ] [ text (model.LastUpdate.ToString("HH:mm:ss")) ]
                        span [ className "stat-label" ] [ text "Last Update" ]
                    ]
                ]
            ]
            // Add dark mode toggle with REAL JavaScript
            button [
                className "dark-mode-toggle"
                onClick "toggleDarkMode()"
            ] [
                text "☀️ Light Mode"
            ]
        ]

    let viewSubsystemCard (subsystem: TarsSubsystem) (isSelected: bool) (dispatch: TarsMsg -> unit) =
        div [
            className (sprintf "subsystem-card clickable %s %s"
                (subsystem.Name.ToLower())
                (if isSelected then "selected" else ""))
            onClick (sprintf "dispatch('SelectSubsystem', '%s')" subsystem.Name)
        ] [
            div [ className "subsystem-header" ] [
                h3 [] [ text subsystem.Name ]
                span [
                    className "status-indicator"
                    style (sprintf "color: %s" (statusColor subsystem.Status))
                ] [
                    text (sprintf "%s %A" (statusIcon subsystem.Status) subsystem.Status)
                ]
            ]
            div [ className "subsystem-metrics" ] [
                div [ className "metric" ] [
                    span [ className "metric-label" ] [ text "Health:" ]
                    span [ className "metric-value" ] [ text (sprintf "%.1f%%" subsystem.HealthPercentage) ]
                ]
                div [ className "metric" ] [
                    span [ className "metric-label" ] [ text "Components:" ]
                    span [ className "metric-value" ] [ text (string subsystem.ActiveComponents) ]
                ]
                div [ className "metric" ] [
                    span [ className "metric-label" ] [ text "Rate:" ]
                    span [ className "metric-value" ] [ text (sprintf "%.1f/sec" subsystem.ProcessingRate) ]
                ]
                div [ className "metric" ] [
                    span [ className "metric-label" ] [ text "Memory:" ]
                    span [ className "metric-value" ] [ text (sprintf "%.1f MB" (float subsystem.MemoryUsage / 1024.0 / 1024.0)) ]
                ]
            ]
            if isSelected then
                div [ className "subsystem-details" ] [
                    h4 [] [ text "Dependencies:" ]
                    ul [] [
                        for dep in subsystem.Dependencies do
                            li [] [ text dep ]
                    ]
                    h4 [] [ text "Detailed Metrics:" ]
                    div [ className "detailed-metrics" ] [
                        for kvp in subsystem.Metrics do
                            div [ className "detailed-metric" ] [
                                span [ className "metric-name" ] [ text kvp.Key ]
                                span [ className "metric-value" ] [ text (formatMetric kvp.Value) ]
                            ]
                    ]
                ]
            else
                empty
        ]
