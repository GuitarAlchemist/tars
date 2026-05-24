namespace TarsEngine.FSharp.Cli.Core

open System
open TarsEngine.FSharp.Cli.Core.MultiAgentDomain
open TarsEngine.FSharp.Cli.Core.UnifiedFormatting

// ============================================================================
// COMPOSABLE HTML COMPONENTS - REPLACES LARGE EMBEDDED STRINGS
// ============================================================================

module HtmlComponents =

    // ============================================================================
    // CORE HTML BUILDING TYPES
    // ============================================================================

    type HtmlAttribute = string * string
    type HtmlElement = {
        Tag: string
        Attributes: HtmlAttribute list
        Content: HtmlContent
    }
    and HtmlContent =
        | Text of string
        | Elements of HtmlElement list
        | Mixed of (string * HtmlElement list)

    // ============================================================================
    // HTML BUILDING FUNCTIONS
    // ============================================================================

    let attr (name: string) (value: string) : HtmlAttribute = (name, value)
    let class_ (value: string) = attr "class" value
    let id_ (value: string) = attr "id" value
    let onclick (value: string) = attr "onclick" value
    let style (value: string) = attr "style" value

    let element (tag: string) (attributes: HtmlAttribute list) (content: HtmlContent) : HtmlElement =
        { Tag = tag; Attributes = attributes; Content = content }

    let div attrs content = element "div" attrs content
    let span attrs content = element "span" attrs content
    let h1 attrs content = element "h1" attrs content
    let h2 attrs content = element "h2" attrs content
    let h3 attrs content = element "h3" attrs content
    let h4 attrs content = element "h4" attrs content
    let h5 attrs content = element "h5" attrs content
    let button attrs content = element "button" attrs content
    let p attrs content = element "p" attrs content

    let text (value: string) : HtmlContent = Text value
    let elements (elems: HtmlElement list) : HtmlContent = Elements elems
    let mixed (textParts: string list) (elems: HtmlElement list) : HtmlContent = 
        Mixed (String.Join("", textParts), elems)

    // ============================================================================
    // HTML RENDERING
    // ============================================================================

    let rec renderAttributes (attrs: HtmlAttribute list) : string =
        attrs
        |> List.map (fun (name, value) -> $"""{name}="{value}"""")
        |> String.concat " "
        |> fun s -> if s = "" then "" else " " + s

    let rec renderContent (content: HtmlContent) : string =
        match content with
        | Text text -> text
        | Elements elems -> elems |> List.map renderElement |> String.concat ""
        | Mixed (text, elems) -> text + (elems |> List.map renderElement |> String.concat "")

    and renderElement (elem: HtmlElement) : string =
        let attrsStr = renderAttributes elem.Attributes
        let contentStr = renderContent elem.Content
        $"<{elem.Tag}{attrsStr}>{contentStr}</{elem.Tag}>"

    // ============================================================================
    // SPECIALIZED COMPONENTS
    // ============================================================================

    /// Agent card component
    let agentCard (agent: UnifiedAgent) : HtmlElement =
        let (_, colorClass) = agentSpecializationWithColor agent.Specialization
        let (statusText, statusColor) = agentStatusWithColor agent.Status
        let (x, y, z) = agent.Position3D
        
        div [
            class_ $"agent-card {colorClass}"
            attr "data-agent-id" agent.Id
        ] (elements [
            div [ class_ "agent-header" ] (elements [
                h4 [] (text agent.Name)
                span [ class_ "agent-type" ] (text (agentSpecialization agent.Specialization))
            ])
            div [ class_ "agent-details" ] (elements [
                div [ class_ "position" ] (text $"Position: {position3D agent.Position3D}")
                div [ class_ "game-theory" ] (text $"Strategy: {gameTheoryModel agent.GameTheoryProfile}")
                div [ class_ "department" ] (text $"Dept: {agent.Department |> Option.defaultValue "None"}")
                div [ class_ "status" ] (text $"Status: {statusText}")
                div [ class_ "progress" ] (text $"Progress: {formatProgress agent.Progress}")
            ])
            div [ class_ "agent-actions" ] (elements [
                button [ onclick $"selectAgent('{agent.Id}')" ] (text "Select")
                button [ onclick $"focusAgent('{agent.Id}')" ] (text "Focus")
                button [ onclick $"analyzeAgent('{agent.Id}')" ] (text "Analyze")
            ])
        ])

    /// Department summary component
    let departmentSummary (dept: UnifiedDepartment) : HtmlElement =
        div [ class_ "dept-summary" ] (elements [
            h5 [] (text dept.Name)
            span [ class_ "dept-info" ] (text $"{dept.Agents.Length} agents")
            span [ class_ "dept-type" ] (text (departmentType dept.DepartmentType))
            div [ class_ "dept-details" ] (elements [
                div [] (text $"Position: {position3D dept.Position3D}")
                div [] (text $"Communication: {communicationProtocol dept.CommunicationProtocol}")
                div [] (text $"Strategy: {gameTheoryStrategy dept.GameTheoryStrategy}")
            ])
        ])

    /// Status item component
    let statusItem (label: string) (value: string) (valueColor: string option) : HtmlElement =
        div [ class_ "status-item" ] (elements [
            span [] (text label)
            span [ 
                class_ "status-value"
                match valueColor with Some color -> style $"color: {color}" | None -> style ""
            ] (text value)
        ])

    /// Progress bar component
    let progressBar (progress: float) (label: string option) : HtmlElement =
        let percentage = progress * 100.0
        div [ class_ "progress-container" ] (elements [
            match label with
            | Some lbl -> div [ class_ "progress-label" ] (text lbl)
            | None -> div [] (text "")
            div [ class_ "progress-bar" ] (elements [
                div [ 
                    class_ "progress-fill"
                    style $"width: {percentage:F1}%"
                ] (text "")
            ])
            div [ class_ "progress-text" ] (text $"{percentage:F1}%")
        ])

    /// Metric display component
    let metricDisplay (title: string) (value: string) (trend: string option) : HtmlElement =
        div [ class_ "metric-display" ] (elements [
            div [ class_ "metric-title" ] (text title)
            div [ class_ "metric-value" ] (text value)
            match trend with
            | Some t -> div [ class_ "metric-trend" ] (text t)
            | None -> div [] (text "")
        ])

    /// Communication log entry component
    let communicationEntry (fromAgent: string) (toAgent: string) (message: string) (timestamp: DateTime) (success: bool) : HtmlElement =
        div [ class_ "comm-entry" ] (elements [
            div [ class_ "comm-header" ] (elements [
                span [ class_ "comm-from" ] (text fromAgent)
                span [ class_ "comm-arrow" ] (text " → ")
                span [ class_ "comm-to" ] (text toAgent)
                span [ class_ "comm-status" ] (text (if success then "✅" else "❌"))
            ])
            div [ class_ "comm-message" ] (text message)
            div [ class_ "comm-timestamp" ] (text (timestamp.ToString("HH:mm:ss")))
        ])

    /// Control button component
    let controlButton (label: string) (action: string) (buttonType: string option) : HtmlElement =
        button [
            class_ $"control-button {buttonType |> Option.defaultValue "default"}"
            onclick action
        ] (text label)

    /// Grid container component
    let gridContainer (className: string) (items: HtmlElement list) : HtmlElement =
        div [ class_ $"grid-container {className}" ] (elements items)

    /// Panel component with header
    let panel (title: string) (content: HtmlElement list) (panelType: string option) : HtmlElement =
        div [ class_ $"panel {panelType |> Option.defaultValue "default"}" ] (elements [
            div [ class_ "panel-header" ] (elements [
                h4 [] (text title)
            ])
            div [ class_ "panel-content" ] (elements content)
        ])

    // ============================================================================
    // LAYOUT COMPONENTS
    // ============================================================================

    /// Main layout structure
    let mainLayout (title: string) (sidebar: HtmlElement list) (mainContent: HtmlElement list) : HtmlElement =
        div [ class_ "container" ] (elements [
            div [ class_ "main-area" ] (elements [
                div [ class_ "header" ] (elements [
                    h1 [] (text title)
                    div [ class_ "header-subtitle" ] (text "TARS Adaptive Multi-Agent Reasoning System")
                ])
                div [ class_ "main-content" ] (elements mainContent)
            ])
            div [ class_ "sidebar" ] (elements sidebar)
        ])

    /// CSS styles component
    let cssStyles : string = """
        <style>
        body {
            margin: 0;
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            color: #fff;
            font-family: 'Consolas', monospace;
            padding: 20px;
        }

        .container {
            display: grid;
            grid-template-columns: 1fr 300px;
            gap: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }

        .main-area {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .header {
            text-align: center;
            padding: 20px;
            background: rgba(0,0,0,0.3);
            border-radius: 10px;
        }

        .header h1 {
            margin: 0;
            color: #4a9eff;
        }

        .header-subtitle {
            margin-top: 10px;
            opacity: 0.8;
        }

        .grid-container {
            display: grid;
            gap: 15px;
        }

        .grid-container.agents {
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        }

        .agent-card {
            background: linear-gradient(145deg, #2a2a3e, #1e1e32);
            border-radius: 10px;
            padding: 15px;
            border: 2px solid transparent;
            transition: all 0.3s ease;
            cursor: pointer;
        }

        .agent-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(74, 158, 255, 0.3);
        }

        .agent-card.green { border-color: #00ff88; }
        .agent-card.blue { border-color: #4a9eff; }
        .agent-card.red { border-color: #ff6b6b; }
        .agent-card.purple { border-color: #9b59b6; }
        .agent-card.yellow { border-color: #ffaa00; }
        .agent-card.cyan { border-color: #00ffff; }

        .agent-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }

        .agent-header h4 {
            margin: 0;
            color: #fff;
        }

        .agent-type {
            background: rgba(74, 158, 255, 0.2);
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
        }

        .agent-details {
            margin: 10px 0;
            font-size: 0.9em;
            opacity: 0.8;
        }

        .agent-details > div {
            margin: 5px 0;
        }

        .agent-actions {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }

        .agent-actions button {
            background: linear-gradient(145deg, #4a9eff, #357abd);
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.8em;
            transition: all 0.2s ease;
        }

        .agent-actions button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(74, 158, 255, 0.4);
        }

        .sidebar {
            background: rgba(0,0,0,0.4);
            border-radius: 10px;
            padding: 20px;
            height: fit-content;
        }

        .panel {
            margin-bottom: 20px;
        }

        .panel-header h4 {
            margin: 0 0 10px 0;
            color: #4a9eff;
        }

        .status-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }

        .status-value {
            color: #00ff88;
            font-weight: bold;
        }

        .dept-summary {
            background: rgba(255,255,255,0.05);
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
        }

        .dept-summary h5 {
            margin: 0 0 5px 0;
            color: #4a9eff;
        }

        .progress-container {
            margin: 10px 0;
        }

        .progress-bar {
            width: 100%;
            height: 8px;
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
            overflow: hidden;
            margin: 5px 0;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #00ff88, #4a9eff);
            transition: width 0.3s ease;
        }

        .progress-text {
            font-size: 0.8em;
            text-align: center;
            opacity: 0.8;
        }

        .control-button {
            background: linear-gradient(145deg, #4a9eff, #357abd);
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px;
            transition: all 0.2s ease;
        }

        .control-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(74, 158, 255, 0.4);
        }

        .comm-entry {
            background: rgba(255,255,255,0.05);
            padding: 8px;
            border-radius: 5px;
            margin: 5px 0;
            border-left: 3px solid #00ff88;
        }

        .comm-header {
            display: flex;
            align-items: center;
            font-weight: bold;
            margin-bottom: 5px;
        }

        .comm-from { color: #4a9eff; }
        .comm-to { color: #ff6b6b; }
        .comm-arrow { color: #ccc; }
        .comm-status { margin-left: auto; }

        .comm-message {
            font-size: 0.9em;
            opacity: 0.9;
            margin: 5px 0;
        }

        .comm-timestamp {
            font-size: 0.8em;
            opacity: 0.6;
            text-align: right;
        }

        .metric-display {
            background: rgba(255,255,255,0.05);
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
            text-align: center;
        }

        .metric-title {
            font-size: 0.8em;
            opacity: 0.8;
            margin-bottom: 5px;
        }

        .metric-value {
            font-size: 1.2em;
            font-weight: bold;
            color: #00ff88;
        }

        .metric-trend {
            font-size: 0.8em;
            margin-top: 5px;
        }

        @keyframes pulse {
            0% { opacity: 0.7; }
            50% { opacity: 1; }
            100% { opacity: 0.7; }
        }

        .agent-card.selected {
            animation: pulse 2s infinite;
            border-color: #00ff88 !important;
        }
        </style>
    """

    /// Complete HTML document
    let htmlDocument (title: string) (content: HtmlElement) : string =
        $"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    {cssStyles}
</head>
<body>
    {renderElement content}
    <script>
        console.log('🎯 TARS Composable HTML Components Active!');
        
        function selectAgent(agentId) {
            document.querySelectorAll('.agent-card').forEach(card => {
                card.classList.remove('selected');
            });

            const selectedCard = document.querySelector(`[data-agent-id="${agentId}"]`);
            if (selectedCard) {
                selectedCard.classList.add('selected');
            }

            console.log('Selected agent:', agentId);
        }
        
        function focusAgent(agentId) {
            const card = document.querySelector(`[data-agent-id="${agentId}"]`);
            if (card) {
                card.scrollIntoView({ behavior: 'smooth', block: 'center' });
                card.style.transform = 'scale(1.05)';
                setTimeout(() => card.style.transform = 'scale(1)', 500);
            }

            console.log('Focused agent:', agentId);
        }
        
        function analyzeAgent(agentId) {
            console.log('Analyzing agent:', agentId);
            // Integration point for concept analysis
        }
    </script>
</body>
</html>"""
