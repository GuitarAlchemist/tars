namespace TarsEngine.FSharp.Cli.UI

open TarsHtml
open TarsDiagnosticsModel
open TarsDiagnosticsUpdate
open TarsDiagnosticsView
open TarsDiagnosticsDetailView

/// Pure Elmish TARS Subsystem Diagnostics - Real MVU Architecture
module TarsElmishDiagnostics =

    // MAIN ELMISH VIEW - Pure function with Real Message Dispatching
    let view (model: TarsDiagnosticsModel) (dispatch: TarsMsg -> unit) =
        div [ className "tars-diagnostics-elmish" ] [
            TarsDiagnosticsView.viewTarsHeader model dispatch
            TarsDiagnosticsView.viewBreadcrumbs model dispatch

            if model.IsLoading then
                div [ className "loading-tars" ] [
                    div [ className "spinner" ] []
                    text "Loading TARS subsystems..."
                ]
            else
                match model.Error with
                | Some error ->
                    div [ className "error-tars" ] [
                        text (sprintf "❌ TARS Error: %s" error)
                        button [ onClick "dispatch(JSON.stringify({ 'Case': 'RefreshTars', 'Fields': [] }))" ] [ text "Retry" ]
                    ]
                | None ->
                    TarsDiagnosticsDetailView.viewTarsOverview model dispatch
        ]

    // REAL ELMISH PROGRAM - True MVU Architecture
    let createTarsElmishProgram () =
        {|
            Init = TarsDiagnosticsModel.init
            Update = TarsDiagnosticsUpdate.update
            View = view
        |}
