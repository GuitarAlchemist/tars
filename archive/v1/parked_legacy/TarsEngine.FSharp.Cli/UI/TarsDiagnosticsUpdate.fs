namespace TarsEngine.FSharp.Cli.UI

open System
open TarsDiagnosticsModel

/// TARS Diagnostics Update Logic
module TarsDiagnosticsUpdate =

    // UPDATE - Pure state transitions for TARS
    let update (msg: TarsMsg) (model: TarsDiagnosticsModel) : TarsDiagnosticsModel =
        match msg with
        | LoadTarsSubsystems ->
            { model with IsLoading = true; Error = None }
            
        | TarsSubsystemsLoaded subsystems ->
            let overallHealth =
                if subsystems.IsEmpty then 0.0
                else
                    subsystems
                    |> List.map (fun s -> s.HealthPercentage)
                    |> List.average

            let totalAgents =
                subsystems
                |> List.sumBy (fun s -> s.ActiveComponents)

            { model with
                AllSubsystems = subsystems
                OverallTarsHealth = overallHealth
                ActiveAgents = totalAgents
                IsLoading = false
                Error = None
                LastUpdate = DateTime.Now }
                
        | TarsError error ->
            { model with IsLoading = false; Error = Some error }
            
        | SelectSubsystem name ->
            { model with SelectedSubsystem = Some name }
            
        | ToggleDetails ->
            { model with ShowDetails = not model.ShowDetails }
            
        | ChangeViewMode mode ->
            { model with ViewMode = mode }
            
        | ToggleAutoRefresh ->
            { model with AutoRefresh = not model.AutoRefresh }
            
        | RefreshTars ->
            { model with IsLoading = true; Error = None }
            
        | SubsystemStatusChanged (name, status) ->
            // Update specific subsystem status in the list
            let updatedSubsystems =
                model.AllSubsystems
                |> List.map (fun s ->
                    if s.Name = name then { s with Status = status } else s)

            { model with AllSubsystems = updatedSubsystems }

        | NavigateToHome ->
            { model with ViewMode = Overview; SelectedSubsystem = None }

        | NavigateToView mode ->
            { model with ViewMode = mode; SelectedSubsystem = None }

        | ClearSubsystemSelection ->
            { model with SelectedSubsystem = None }
