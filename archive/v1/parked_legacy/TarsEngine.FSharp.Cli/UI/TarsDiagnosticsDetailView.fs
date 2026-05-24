namespace TarsEngine.FSharp.Cli.UI

open TarsHtml
open TarsDiagnosticsModel

/// TARS Diagnostics Detail View Components
module TarsDiagnosticsDetailView =

    // SUBSYSTEM DETAIL VIEW COMPONENT
    let viewSubsystemDetail (subsystem: TarsSubsystem) (dispatch: TarsMsg -> unit) =
        div [ className "subsystem-detail-view" ] [
            // Detail Header
            div [ className "detail-header" ] [
                div [ className "detail-title-section" ] [
                    h2 [] [ text (sprintf "🔧 %s" subsystem.Name) ]
                    span [
                        className "detail-status"
                        style (sprintf "color: %s" (statusColor subsystem.Status))
                    ] [
                        text (sprintf "%s %A" (statusIcon subsystem.Status) subsystem.Status)
                    ]
                ]
                button [
                    className "close-detail-btn"
                    onClick "clearSubsystemSelection()"
                ] [
                    text "✕ Close"
                ]
            ]

            // Detailed Metrics Grid
            div [ className "detail-metrics-grid" ] [
                div [ className "metric-card primary" ] [
                    div [ className "metric-icon" ] [ text "💚" ]
                    div [ className "metric-content" ] [
                        div [ className "metric-value large" ] [ text (sprintf "%.1f%%" subsystem.HealthPercentage) ]
                        div [ className "metric-label" ] [ text "System Health" ]
                    ]
                ]
                div [ className "metric-card" ] [
                    div [ className "metric-icon" ] [ text "⚙️" ]
                    div [ className "metric-content" ] [
                        div [ className "metric-value" ] [ text (string subsystem.ActiveComponents) ]
                        div [ className "metric-label" ] [ text "Active Components" ]
                    ]
                ]
                div [ className "metric-card" ] [
                    div [ className "metric-icon" ] [ text "⚡" ]
                    div [ className "metric-content" ] [
                        div [ className "metric-value" ] [ text (sprintf "%.1f/s" subsystem.ProcessingRate) ]
                        div [ className "metric-label" ] [ text "Processing Rate" ]
                    ]
                ]
                div [ className "metric-card" ] [
                    div [ className "metric-icon" ] [ text "💾" ]
                    div [ className "metric-content" ] [
                        div [ className "metric-value" ] [ text (sprintf "%.1f GB" (float subsystem.MemoryUsage / 1024.0 / 1024.0 / 1024.0)) ]
                        div [ className "metric-label" ] [ text "Memory Usage" ]
                    ]
                ]
            ]

            // Dependencies Section
            div [ className "detail-section" ] [
                h3 [] [ text "🔗 Dependencies" ]
                div [ className "dependencies-grid" ] [
                    for dep in subsystem.Dependencies do
                        div [ className "dependency-item" ] [
                            span [ className "dependency-icon" ] [ text "🔗" ]
                            span [ className "dependency-name" ] [ text dep ]
                        ]
                ]
            ]

            // Advanced Metrics Section
            div [ className "detail-section" ] [
                h3 [] [ text "📊 Advanced Metrics" ]
                div [ className "advanced-metrics" ] [
                    for kvp in subsystem.Metrics do
                        div [ className "advanced-metric-row" ] [
                            span [ className "metric-name" ] [ text kvp.Key ]
                            span [ className "metric-value" ] [ text (formatMetric kvp.Value) ]
                        ]
                ]
            ]
        ]

    let viewTarsOverview (model: TarsDiagnosticsModel) (dispatch: TarsMsg -> unit) =
        match model.SelectedSubsystem with
        | Some selectedName ->
            // Show detailed view for selected subsystem
            match model.AllSubsystems |> List.tryFind (fun s -> s.Name = selectedName) with
            | Some subsystem -> viewSubsystemDetail subsystem dispatch
            | None ->
                div [ className "error-message" ] [
                    text (sprintf "❌ Subsystem '%s' not found" selectedName)
                ]
        | None ->
            // Show overview grid
            div [ className "tars-overview" ] [
                div [ className "subsystems-grid" ] [
                    if model.AllSubsystems.IsEmpty then
                        div [ className "loading-message" ] [
                            text "🔄 Loading all TARS subsystems..."
                        ]
                    else
                        for subsystem in model.AllSubsystems do
                            TarsDiagnosticsView.viewSubsystemCard subsystem false dispatch
                ]
            ]
