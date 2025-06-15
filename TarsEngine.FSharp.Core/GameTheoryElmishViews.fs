namespace TarsEngine.FSharp.Core

open System
open TarsEngine.FSharp.Core.GameTheoryElmishModels
open TarsEngine.FSharp.Core.ModernGameTheory

/// Simple HTML element representation
type HtmlElement = {
    Tag: string
    Attributes: Map<string, string>
    Content: string
    Children: HtmlElement list
}

/// Simple HTML builders
module Html =
    let div attrs children = { Tag = "div"; Attributes = Map.ofList attrs; Content = ""; Children = children }
    let h1 attrs content = { Tag = "h1"; Attributes = Map.ofList attrs; Content = content; Children = [] }
    let h3 attrs content = { Tag = "h3"; Attributes = Map.ofList attrs; Content = content; Children = [] }
    let h4 attrs content = { Tag = "h4"; Attributes = Map.ofList attrs; Content = content; Children = [] }
    let p attrs content = { Tag = "p"; Attributes = Map.ofList attrs; Content = content; Children = [] }
    let span attrs content = { Tag = "span"; Attributes = Map.ofList attrs; Content = content; Children = [] }
    let button attrs content = { Tag = "button"; Attributes = Map.ofList attrs; Content = content; Children = [] }
    let str content = content

/// Text-based Views for Modern Game Theory UI (Console/Terminal representation)
module GameTheoryElmishViews =

    /// Render agent information as text
    let renderAgentText (agent: AgentUIState) (isSelected: bool) =
        let marker = if isSelected then "►" else " "
        let status = if agent.IsActive then "🟢" else "🔴"
        sprintf "%s %s %s - %A (Conf: %.1f%%, Perf: %.3f) %s"
            marker status agent.AgentId agent.GameTheoryModel
            (agent.ConfidenceLevel * 100.0) agent.PerformanceScore
            (agent.LastUpdate.ToString("HH:mm:ss"))

    /// Render coordination panel as text
    let renderCoordinationText (coordination: CoordinationUIState) =
        let trendIcon = if coordination.IsImproving then "📈" else "📉"
        let statusIcon =
            match coordination.EquilibriumStatus with
            | "Stable" -> "✅"
            | "Converging" -> "🔄"
            | _ -> "⚠️"

        [
            "🤝 MULTI-AGENT COORDINATION ANALYSIS"
            "======================================"
            sprintf "Average Coordination: %.3f %s" coordination.AverageCoordination trendIcon
            sprintf "Trend: %+.3f" coordination.CoordinationTrend
            sprintf "Status: %s %s" statusIcon coordination.EquilibriumStatus
            sprintf "Active Agents: %d" coordination.ActiveAgents
            sprintf "Recommended Model: %s" coordination.RecommendedModel
            ""
        ]

    /// Render system overview as text
    let renderSystemOverview (state: GameTheoryUIState) =
        [
            "🎯 TARS MODERN GAME THEORY SYSTEM OVERVIEW"
            "=========================================="
            ""
            sprintf "System Performance: %.1f%%" (UIHelpers.calculateSystemPerformance state * 100.0)
            sprintf "Equilibrium Status: %s" (UIHelpers.getEquilibriumStatus state)
            sprintf "Active Agents: %d" (UIHelpers.getActiveAgentsCount state)
            sprintf "Real-time Mode: %s" (if state.IsRealTimeMode then "🔴 LIVE" else "⏸️ STATIC")
            sprintf "Last Update: %s" (state.LastDataUpdate.ToString("yyyy-MM-dd HH:mm:ss"))
            ""
        ]

    /// Render agents list as text
    let renderAgentsList (agents: AgentUIState list) (selectedAgent: string option) =
        [
            "👥 MULTI-AGENT SYSTEM"
            "====================="
            ""
        ] @ [
            for agent in agents do
                renderAgentText agent (selectedAgent = Some agent.AgentId)
        ] @ [
            ""
            sprintf "Total Agents: %d" agents.Length
            sprintf "Active Agents: %d" (agents |> List.filter (_.IsActive) |> List.length)
            ""
        ]

    /// Render 3D visualization status as text
    let render3DVisualizationText (threeDState: ThreeDVisualizationState) =
        [
            "🌌 3D GAME THEORY SPACE VISUALIZATION"
            "====================================="
            ""
            sprintf "Geometry: %s" threeDState.SpaceGeometry
            let (x,y,z) = threeDState.CameraPosition
            sprintf "Camera Position: (%.1f, %.1f, %.1f)" x y z
            sprintf "Agent Positions: %d tracked" threeDState.AgentPositions.Count
            sprintf "Animation Speed: %.1fx" threeDState.AnimationSpeed
            sprintf "Show Trajectories: %s" (if threeDState.ShowTrajectories then "✅" else "❌")
            sprintf "Interstellar Mode: %s" (if threeDState.InterstellarMode then "🚀 ACTIVE" else "❌ INACTIVE")
            ""
            if threeDState.InterstellarMode then
                "🚀 INTERSTELLAR MODE FEATURES:"
                "• WebGPU-powered 3D rendering"
                "• Real-time agent trajectory tracking"
                "• Coordination flow animations"
                "• Interstellar movie-style effects"
                ""
        ]

    /// Render analysis results as text
    let renderAnalysisResults (results: string list) (isLoading: bool) =
        if isLoading then
            [
                "🔄 RUNNING COMPREHENSIVE ANALYSIS..."
                "===================================="
                ""
                "Please wait while the system analyzes:"
                "• Multi-agent coordination patterns"
                "• Game theory model performance"
                "• Equilibrium convergence rates"
                "• Strategic interaction dynamics"
                ""
            ]
        elif results.IsEmpty then
            [
                "📊 GAME THEORY ANALYSIS"
                "======================="
                ""
                "No analysis results yet."
                "Click 'Run Analysis' to start comprehensive evaluation."
                ""
            ]
        else
            [
                "📊 ANALYSIS RESULTS"
                "==================="
                ""
            ] @ results @ [""]

    /// Render available game theory models as text
    let renderGameTheoryModels (selectedModel: GameTheoryModel option) =
        [
            "🎲 AVAILABLE GAME THEORY MODELS"
            "==============================="
            ""
            "• Quantal Response Equilibrium (QRE) - Bounded rationality"
            "• Cognitive Hierarchy - Iterative strategic thinking"
            "• No-Regret Learning - Adaptive learning algorithms"
            "• Correlated Equilibrium - Coordination mechanisms"
            "• Evolutionary Game Theory - Population dynamics"
            "• Mean Field Games - Large-scale interactions"
            ""
            sprintf "Currently Selected: %s"
                (match selectedModel with
                 | Some model -> sprintf "%A" model
                 | None -> "None")
            ""
        ]

    /// Update the coordination tab content to include models
    let updateCoordinationContent (state: GameTheoryUIState) =
        renderCoordinationText state.Coordination @
        renderGameTheoryModels state.Visualization.SelectedModel

    /// Main view function - renders state as text for console/terminal display
    let renderView (state: GameTheoryUIState) : string list =
        let content =
            match state.ActiveTab with
            | "overview" -> renderSystemOverview state @ renderCoordinationText state.Coordination
            | "agents" -> renderAgentsList state.Agents state.SelectedAgent
            | "coordination" -> updateCoordinationContent state
            | "analysis" -> renderAnalysisResults state.AnalysisResults state.IsLoading
            | "3d" -> render3DVisualizationText state.ThreeD
            | _ -> ["Unknown tab: " + state.ActiveTab]

        let header = [
            "🎲 TARS MODERN GAME THEORY INTERFACE"
            "====================================="
            ""
            sprintf "Mode: %s | Tab: %s | Update: %s"
                (if state.IsRealTimeMode then "🔴 LIVE" else "⏸️ STATIC")
                state.ActiveTab
                (state.LastDataUpdate.ToString("HH:mm:ss"))
            ""
        ]

        let errorSection =
            match state.ErrorMessage with
            | Some error -> ["❌ ERROR: " + error; ""]
            | None -> []

        let footer = [
            ""
            "📋 AVAILABLE TABS: overview | agents | coordination | analysis | 3d"
            "🎮 CONTROLS: Real-time mode, Analysis tools, 3D visualization"
            sprintf "⚙️ Update Interval: %dms | Performance: %.1f%%"
                state.UpdateInterval
                (UIHelpers.calculateSystemPerformance state * 100.0)
            ""
        ]

        header @ errorSection @ content @ footer

    /// Simple console renderer
    let renderToConsole (state: GameTheoryUIState) : unit =
        let lines = renderView state
        System.Console.Clear()
        for line in lines do
            printfn "%s" line
