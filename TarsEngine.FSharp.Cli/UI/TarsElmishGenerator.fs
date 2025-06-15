namespace TarsEngine.FSharp.Cli.UI

open System
open System.Text
open System.Text.RegularExpressions

// ===============================
// TARS AI-Driven Elmish UI Generator
// ===============================

module TarsElmishGenerator =
    
    // --- DSL Types ---
    type UIComponent =
        | Header of string
        | Table of string list * string
        | Button of string * string
        | LineChart of string
        | BarChart of string
        | PieChart of string
        | HtmlRaw of string
        | ThreeJS of string             // 3D scene bind
        | VexFlow of string             // VexFlow music notation bind
        | D3Visualization of string     // D3.js visualization bind
        | MetricsPanel of string        // TARS metrics panel
        | ChatPanel of string           // TARS AI chat panel
        | ThoughtFlow of string         // TARS thought flow visualization
        | ProjectsPanel of string       // TARS projects panel
        | AgentTeams of string          // TARS agent teams visualization
        | DiagnosticsPanel of string    // TARS diagnostics panel
    
    /// DSL Root
    type ElmishUIBlock = {
        ViewId: string
        Title: string
        Components: UIComponent list
        FeedbackEnabled: bool
        RealTimeUpdates: bool
    }
    
    /// Feedback for UI evolution
    type UIFeedback = {
        ViewId: string
        UsabilityIssues: string list
        ProposedChanges: string list
        UserRating: float
        PerformanceMetrics: Map<string, float>
    }
    
    // --- DSL Parser ---
    let parseDsl (dslText: string) : ElmishUIBlock =
        let lines = dslText.Split('\n') |> Array.map (_.Trim()) |> Array.filter (fun l -> not (String.IsNullOrWhiteSpace(l)))
        let mutable viewId = ""
        let mutable title = ""
        let mutable components = []
        let mutable feedbackEnabled = false
        let mutable realTimeUpdates = false
        let mutable inUiBlock = false
        
        for line in lines do
            if line.StartsWith("ui {") then
                inUiBlock <- true
            elif line = "}" then
                inUiBlock <- false
            elif inUiBlock then
                if line.StartsWith("view_id:") then
                    viewId <- line.Substring(8).Trim().Trim('"')
                elif line.StartsWith("title:") then
                    title <- line.Substring(6).Trim().Trim('"')
                elif line.StartsWith("feedback_enabled:") then
                    feedbackEnabled <- line.Substring(17).Trim() = "true"
                elif line.StartsWith("real_time_updates:") then
                    realTimeUpdates <- line.Substring(18).Trim() = "true"
                elif line.StartsWith("header ") then
                    components <- Header(line.Substring(7).Trim().Trim('"')) :: components
                elif line.StartsWith("table bind(") then
                    let inner = line.Substring(11).TrimEnd(')')
                    components <- Table (["Agent"; "Score"; "Reflection"], inner) :: components
                elif line.StartsWith("button ") then
                    let parts = line.Split([|" on "|], StringSplitOptions.None)
                    if parts.Length >= 2 then
                        components <- Button(parts.[0].Substring(7).Trim().Trim('"'), parts.[1].Trim()) :: components
                elif line.StartsWith("line_chart bind(") then
                    components <- LineChart(line.Substring(16).TrimEnd(')')) :: components
                elif line.StartsWith("bar_chart bind(") then
                    components <- BarChart(line.Substring(15).TrimEnd(')')) :: components
                elif line.StartsWith("pie_chart bind(") then
                    components <- PieChart(line.Substring(15).TrimEnd(')')) :: components
                elif line.StartsWith("html ") then
                    components <- HtmlRaw(line.Substring(5).Trim().Trim('"')) :: components
                elif line.StartsWith("threejs bind(") then
                    components <- ThreeJS(line.Substring(13).TrimEnd(')')) :: components
                elif line.StartsWith("vexflow bind(") then
                    components <- VexFlow(line.Substring(13).TrimEnd(')')) :: components
                elif line.StartsWith("d3_viz bind(") then
                    components <- D3Visualization(line.Substring(12).TrimEnd(')')) :: components
                elif line.StartsWith("metrics_panel bind(") then
                    components <- MetricsPanel(line.Substring(19).TrimEnd(')')) :: components
                elif line.StartsWith("chat_panel bind(") then
                    components <- ChatPanel(line.Substring(16).TrimEnd(')')) :: components
                elif line.StartsWith("thought_flow bind(") then
                    components <- ThoughtFlow(line.Substring(18).TrimEnd(')')) :: components
                elif line.StartsWith("projects_panel bind(") then
                    components <- ProjectsPanel(line.Substring(20).TrimEnd(')')) :: components
                elif line.StartsWith("agent_teams bind(") then
                    components <- AgentTeams(line.Substring(17).TrimEnd(')')) :: components
                elif line.StartsWith("diagnostics_panel bind(") then
                    components <- DiagnosticsPanel(line.Substring(23).TrimEnd(')')) :: components
        
        { 
            ViewId = if String.IsNullOrEmpty(viewId) then "GeneratedView" else viewId
            Title = if String.IsNullOrEmpty(title) then "TARS Generated UI" else title
            Components = List.rev components
            FeedbackEnabled = feedbackEnabled
            RealTimeUpdates = realTimeUpdates
        }
    
    // --- Elmish Code Generator ---
    let generateElmishView (ui: ElmishUIBlock) : string =
        let sb = StringBuilder()
        
        // Module header
        sb.AppendLine($"namespace TarsEngine.FSharp.Cli.UI.Generated") |> ignore
        sb.AppendLine("") |> ignore
        sb.AppendLine("open System") |> ignore
        sb.AppendLine("open Elmish") |> ignore
        sb.AppendLine("open TarsEngine.FSharp.Cli.UI.ElmishHelpers") |> ignore
        sb.AppendLine("open TarsEngine.FSharp.Cli.UI.TarsInterop") |> ignore
        sb.AppendLine("open TarsEngine.FSharp.Cli.CognitivePsychology") |> ignore
        sb.AppendLine("open TarsEngine.FSharp.Cli.BeliefPropagation") |> ignore
        sb.AppendLine("open TarsEngine.FSharp.Cli.Projects") |> ignore
        sb.AppendLine("") |> ignore
        sb.AppendLine($"module {ui.ViewId} =") |> ignore
        sb.AppendLine("") |> ignore
        
        // Model type
        sb.AppendLine("    type Model = {") |> ignore
        sb.AppendLine("        Title: string") |> ignore
        sb.AppendLine("        IsLoading: bool") |> ignore
        sb.AppendLine("        LastUpdate: DateTime") |> ignore
        sb.AppendLine("        Data: Map<string, obj>") |> ignore
        if ui.FeedbackEnabled then
            sb.AppendLine("        Feedback: UIFeedback option") |> ignore
        if ui.RealTimeUpdates then
            sb.AppendLine("        WebSocketConnected: bool") |> ignore
        sb.AppendLine("    }") |> ignore
        sb.AppendLine("") |> ignore
        
        // Message type
        sb.AppendLine("    type Message =") |> ignore
        sb.AppendLine("        | UpdateData of string * obj") |> ignore
        sb.AppendLine("        | SetLoading of bool") |> ignore
        sb.AppendLine("        | Refresh") |> ignore
        if ui.FeedbackEnabled then
            sb.AppendLine("        | SubmitFeedback of string") |> ignore
        if ui.RealTimeUpdates then
            sb.AppendLine("        | WebSocketConnected of bool") |> ignore
        sb.AppendLine("") |> ignore
        
        // Init function
        sb.AppendLine("    let init () =") |> ignore
        sb.AppendLine("        {") |> ignore
        sb.AppendLine($"            Title = \"{ui.Title}\"") |> ignore
        sb.AppendLine("            IsLoading = false") |> ignore
        sb.AppendLine("            LastUpdate = DateTime.Now") |> ignore
        sb.AppendLine("            Data = Map.empty") |> ignore
        if ui.FeedbackEnabled then
            sb.AppendLine("            Feedback = None") |> ignore
        if ui.RealTimeUpdates then
            sb.AppendLine("            WebSocketConnected = false") |> ignore
        sb.AppendLine("        }, Cmd.ofMsg Refresh") |> ignore
        sb.AppendLine("") |> ignore
        
        // Update function
        sb.AppendLine("    let update msg model =") |> ignore
        sb.AppendLine("        match msg with") |> ignore
        sb.AppendLine("        | UpdateData (key, value) ->") |> ignore
        sb.AppendLine("            { model with Data = Map.add key value model.Data; LastUpdate = DateTime.Now }, Cmd.none") |> ignore
        sb.AppendLine("        | SetLoading loading ->") |> ignore
        sb.AppendLine("            { model with IsLoading = loading }, Cmd.none") |> ignore
        sb.AppendLine("        | Refresh ->") |> ignore
        sb.AppendLine("            { model with LastUpdate = DateTime.Now }, Cmd.none") |> ignore
        if ui.FeedbackEnabled then
            sb.AppendLine("        | SubmitFeedback feedback ->") |> ignore
            sb.AppendLine("            // TODO: Process feedback for UI evolution") |> ignore
            sb.AppendLine("            model, Cmd.none") |> ignore
        if ui.RealTimeUpdates then
            sb.AppendLine("        | WebSocketConnected connected ->") |> ignore
            sb.AppendLine("            { model with WebSocketConnected = connected }, Cmd.none") |> ignore
        sb.AppendLine("") |> ignore
        
        // View function
        sb.AppendLine("    let view model dispatch =") |> ignore
        sb.AppendLine("        div [ Class \"tars-generated-ui\" ] [") |> ignore
        sb.AppendLine("            // Header") |> ignore
        sb.AppendLine("            div [ Class \"ui-header\" ] [") |> ignore
        sb.AppendLine($"                h1 [] [ text \"{ui.Title}\" ]") |> ignore
        sb.AppendLine("                div [ Class \"ui-status\" ] [") |> ignore
        sb.AppendLine("                    span [ Class \"last-update\" ] [ text (sprintf \"Last Update: %s\" (model.LastUpdate.ToString(\"HH:mm:ss\"))) ]") |> ignore
        if ui.RealTimeUpdates then
            sb.AppendLine("                    span [ Class (if model.WebSocketConnected then \"status-connected\" else \"status-disconnected\") ] [") |> ignore
            sb.AppendLine("                        text (if model.WebSocketConnected then \" Connected\" else \" Disconnected\")") |> ignore
            sb.AppendLine("                    ]") |> ignore
        sb.AppendLine("                ]") |> ignore
        sb.AppendLine("            ]") |> ignore
        sb.AppendLine("            ") |> ignore
        sb.AppendLine("            // Components") |> ignore
        sb.AppendLine("            div [ Class \"ui-components\" ] [") |> ignore
        
        for comp in ui.Components do
            match comp with
            | Header title ->
                sb.AppendLine($"                h2 [ Class \"component-header\" ] [ text \"{title}\" ]") |> ignore
            | Table (headers, bind) ->
                sb.AppendLine("                div [ Class \"table-component\" ] [") |> ignore
                sb.AppendLine("                    // TODO: Implement table with data binding") |> ignore
                sb.AppendLine($"                    text \"Table: {bind}\"") |> ignore
                sb.AppendLine("                ]") |> ignore
            | Button (label, handler) ->
                sb.AppendLine($"                button [ Class \"btn btn-primary\"; OnClick (fun _ -> dispatch Refresh) ] [ text \"{label}\" ]") |> ignore
            | LineChart bind ->
                sb.AppendLine($"                div [ Class \"chart-component\" ] [ text \"Line Chart: {bind}\" ]") |> ignore
            | BarChart bind ->
                sb.AppendLine($"                div [ Class \"chart-component\" ] [ text \"Bar Chart: {bind}\" ]") |> ignore
            | PieChart bind ->
                sb.AppendLine($"                div [ Class \"chart-component\" ] [ text \"Pie Chart: {bind}\" ]") |> ignore
            | HtmlRaw content ->
                sb.AppendLine($"                div [ Class \"html-raw\" ] [ text \"{content}\" ]") |> ignore
            | ThreeJS bind ->
                sb.AppendLine($"                div [ Class \"threejs-component\"; OnClick (fun _ -> TarsInterop.Three.initScene(\"{bind}\")) ] [ text \"3D Scene: {bind}\" ]") |> ignore
            | VexFlow bind ->
                sb.AppendLine($"                div [ Class \"vexflow-component\"; OnClick (fun _ -> TarsInterop.VexFlow.renderNotation(\"{bind}\")) ] [ text \"Music Notation: {bind}\" ]") |> ignore
            | D3Visualization bind ->
                sb.AppendLine($"                div [ Class \"d3-component\" ] [ text \"D3 Visualization: {bind}\" ]") |> ignore
            | MetricsPanel bind ->
                sb.AppendLine($"                div [ Class \"metrics-panel\" ] [ text \"Metrics Panel: {bind}\" ]") |> ignore
            | ChatPanel bind ->
                sb.AppendLine($"                div [ Class \"chat-panel\" ] [ text \"Chat Panel: {bind}\" ]") |> ignore
            | ThoughtFlow bind ->
                sb.AppendLine($"                div [ Class \"thought-flow\" ] [ text \"Thought Flow: {bind}\" ]") |> ignore
            | ProjectsPanel bind ->
                sb.AppendLine($"                div [ Class \"projects-panel\" ] [ text \"Projects Panel: {bind}\" ]") |> ignore
            | AgentTeams bind ->
                sb.AppendLine($"                div [ Class \"agent-teams\" ] [ text \"Agent Teams: {bind}\" ]") |> ignore
            | DiagnosticsPanel bind ->
                sb.AppendLine($"                div [ Class \"diagnostics-panel\" ] [ text \"Diagnostics Panel: {bind}\" ]") |> ignore
        
        sb.AppendLine("            ]") |> ignore
        
        if ui.FeedbackEnabled then
            sb.AppendLine("            ") |> ignore
            sb.AppendLine("            // Feedback Section") |> ignore
            sb.AppendLine("            div [ Class \"feedback-section\" ] [") |> ignore
            sb.AppendLine("                h3 [] [ text \"UI Feedback\" ]") |> ignore
            sb.AppendLine("                button [ Class \"btn btn-secondary\"; OnClick (fun _ -> dispatch (SubmitFeedback \"Good UI\")) ] [ text \"Submit Feedback\" ]") |> ignore
            sb.AppendLine("            ]") |> ignore
        
        sb.AppendLine("        ]") |> ignore
        sb.AppendLine("") |> ignore
        
        sb.ToString()
    
    // --- Entry Point ---
    let processUiDsl (dslText: string) : string =
        let ast = parseDsl dslText
        generateElmishView ast
    
    // --- Feedback Processing ---
    let processFeedback (feedback: UIFeedback) (originalDsl: string) : string =
        // TODO: Implement AI-driven DSL evolution based on feedback
        // This would use local LLM to modify the DSL based on usability issues and proposed changes
        originalDsl
    
    // --- Extract UI Block from .trsx ---
    let extractUiBlockFromTrsx (trsxContent: string) : string option =
        let uiBlockPattern = @"ui\s*\{[^}]*\}"
        let regex = Regex(uiBlockPattern, RegexOptions.Singleline)
        let matches = regex.Matches(trsxContent)
        if matches.Count > 0 then
            Some(matches.[0].Value)
        else
            None
