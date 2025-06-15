namespace TarsEngine.FSharp.DynamicUI

open Fable.React
open Fable.React.Props
open Types
open AgentTeams
open State

module View =
    
    /// Render the main application view
    let view (model: Model) (dispatch: Msg -> unit) =
        div [ 
            Style [ 
                Height "100vh"
                Width "100vw"
                Background "#000"
                Color "#00ff88"
                FontFamily "monospace"
                Position "relative"
                Overflow "hidden"
            ] 
        ] [
            // 3D Canvas Container (Three.js will be injected here)
            div [ 
                Id "threejs-container"
                Style [ 
                    Width "100%"
                    Height "100%"
                    Position "absolute"
                    Top "0"
                    Left "0"
                    ZIndex 1
                ] 
            ] []
            
            // Agent Teams Status Panel
            getAgentTeamStatusComponent model.AgentTeams
            
            // Dynamic Components Overlay
            div [ 
                Style [ 
                    Position "absolute"
                    Top "0"
                    Left "0"
                    Width "100%"
                    Height "100%"
                    PointerEvents "none"
                    ZIndex 10
                ] 
            ] [
                for comp in model.DynamicComponents do
                    match comp.ReactElement with
                    | Some element -> 
                        div [ 
                            Key comp.Id
                            Style [ PointerEvents "auto" ]
                            ClassName "dynamic-component"
                        ] [ element ]
                    | None -> ()
            ]
            
            // Control Panel
            div [ 
                Style [ 
                    Position "absolute"
                    Bottom "20px"
                    Left "20px"
                    Background "rgba(0, 0, 0, 0.9)"
                    Padding "20px"
                    BorderRadius "10px"
                    Border "1px solid #00ff88"
                    ZIndex 100
                ] 
            ] [
                h4 [] [ str "🎮 Dynamic UI Control" ]
                
                if not model.IsGenerating then
                    button [ 
                        OnClick (fun _ -> dispatch StartDynamicGeneration)
                        Style [ 
                            Background "#00ff88"
                            Color "#000"
                            Padding "10px 20px"
                            Border "none"
                            BorderRadius "5px"
                            Cursor "pointer"
                            FontWeight "bold"
                            Margin "5px"
                        ]
                    ] [ str "🚀 Start Dynamic Generation" ]
                else
                    button [ 
                        OnClick (fun _ -> dispatch StopDynamicGeneration)
                        Style [ 
                            Background "#ff4444"
                            Color "#fff"
                            Padding "10px 20px"
                            Border "none"
                            BorderRadius "5px"
                            Cursor "pointer"
                            FontWeight "bold"
                            Margin "5px"
                        ]
                    ] [ str "⏹️ Stop Generation" ]
                
                div [ Style [ MarginTop "10px"; FontSize "12px" ] ] [
                    str $"Status: {if model.IsGenerating then "🔄 GENERATING" else "⏸️ STOPPED"}"
                ]
            ]
            
            // Generation Log
            div [ 
                Style [ 
                    Position "absolute"
                    Bottom "20px"
                    Right "20px"
                    Width "400px"
                    Height "200px"
                    Background "rgba(0, 0, 0, 0.9)"
                    Padding "15px"
                    BorderRadius "10px"
                    Border "1px solid #00ff88"
                    Overflow "auto"
                    ZIndex 100
                    FontSize "11px"
                ] 
            ] [
                h4 [] [ str "📜 Generation Log" ]
                div [] [
                    for logEntry in model.GenerationLog |> List.rev ->
                        div [ Style [ MarginBottom "3px"; LineHeight "1.3" ] ] [ str logEntry ]
                ]
            ]
            
            // Performance Metrics
            div [ 
                Style [ 
                    Position "absolute"
                    Top "20px"
                    Right "20px"
                    Background "rgba(0, 0, 0, 0.9)"
                    Padding "15px"
                    BorderRadius "10px"
                    Border "1px solid #00ff88"
                    ZIndex 100
                    FontSize "12px"
                ] 
            ] [
                h4 [] [ str "⚡ Live Performance" ]
                div [] [ str $"FPS: {model.PerformanceMetrics.FPS:F1}" ]
                div [] [ str $"Components: {model.PerformanceMetrics.ComponentCount}" ]
                div [] [ str $"Active Agents: {model.PerformanceMetrics.ActiveAgents}" ]
                div [] [ str $"Memory: {model.PerformanceMetrics.MemoryUsage:F1}MB" ]
                div [] [ str $"WebGPU: {if model.PerformanceMetrics.WebGPUActive then "✅ Active" else "❌ Inactive"}" ]
                div [] [ str $"Last Update: {model.LastUpdate.ToString("HH:mm:ss")}" ]
            ]
            
            // TARS Agents Status
            div [ 
                Style [ 
                    Position "absolute"
                    Top "200px"
                    Right "20px"
                    Background "rgba(0, 0, 0, 0.9)"
                    Padding "15px"
                    BorderRadius "10px"
                    Border "1px solid #00ff88"
                    ZIndex 100
                    FontSize "12px"
                    MaxWidth "300px"
                ] 
            ] [
                h4 [] [ str "🤖 TARS Agent Tree" ]
                for agent in model.TarsAgents ->
                    div [ 
                        Style [ 
                            Margin "8px 0"
                            Padding "8px"
                            BorderLeft $"3px solid {if agent.Status = Running then "#00ff88" else "#666"}"
                            PaddingLeft "10px"
                            Background $"rgba(0, 255, 136, {if agent.Status = Running then "0.1" else "0.05"})"
                            BorderRadius "3px"
                            Cursor "pointer"
                        ]
                        OnClick (fun _ -> dispatch (ToggleAgent agent.Id))
                    ] [
                        div [ Style [ FontWeight "bold" ] ] [ 
                            str $"{if agent.Status = Running then "🟢" else "⚫"} {agent.Name}" 
                        ]
                        div [ Style [ FontSize "10px"; Color "#aaa" ] ] [ 
                            str $"{agent.Department} | Tasks: {agent.TaskCount}" 
                        ]
                    ]
            ]
            
            // Requirements Panel
            div [ 
                Style [ 
                    Position "absolute"
                    Top "400px"
                    Left "20px"
                    Background "rgba(0, 0, 0, 0.9)"
                    Padding "15px"
                    BorderRadius "10px"
                    Border "1px solid #00ff88"
                    ZIndex 100
                    FontSize "12px"
                ] 
            ] [
                h4 [] [ str "📋 Scene Requirements" ]
                div [] [ str $"Agent Tree: {model.SceneRequirements.AgentTreeVisible}" ]
                div [] [ str $"Connections: {model.SceneRequirements.ShowConnections}" ]
                div [] [ str $"Style: {model.SceneRequirements.VisualizationStyle}" ]
                div [] [ str $"Interaction: {model.SceneRequirements.InteractionMode}" ]
                div [] [ str $"Performance: {model.SceneRequirements.PerformanceMode}" ]
                div [] [ str $"Update Rate: {model.SceneRequirements.UpdateFrequency} Hz" ]
            ]
        ]
    
    /// Render loading screen
    let loadingView () =
        div [ 
            Style [ 
                Height "100vh"
                Width "100vw"
                Background "#000"
                Color "#00ff88"
                FontFamily "monospace"
                Display "flex"
                AlignItems "center"
                JustifyContent "center"
                FlexDirection "column"
            ] 
        ] [
            h1 [] [ str "🚀 TARS Dynamic UI" ]
            div [] [ str "Initializing agent teams..." ]
            div [ Style [ MarginTop "20px"; FontSize "12px" ] ] [
                str "F# + Fable + Elmish + Three.js WebGPU"
            ]
        ]
