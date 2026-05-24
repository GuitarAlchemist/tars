namespace TarsEngine.FSharp.Cli.UI

// ============================================================================
// ELMISH HTML HELPERS - F# FUNCTIONAL REACTIVE PROGRAMMING
// ============================================================================

module ElmishHelpers =
    
    type HtmlAttribute =
        | Class of string
        | Type of string
        | Placeholder of string
        | OnClick of (obj -> unit)
        | OnKeyPress of (obj -> unit)
    
    type HtmlElement = {
        Tag: string
        Attributes: HtmlAttribute list
        Children: HtmlNode list
    }
    
    and HtmlNode =
        | Element of HtmlElement
        | Text of string
    
    // HTML Element Builders
    let div attrs children = Element { Tag = "div"; Attributes = attrs; Children = children }
    let h1 attrs children = Element { Tag = "h1"; Attributes = attrs; Children = children }
    let h5 attrs children = Element { Tag = "h5"; Attributes = attrs; Children = children }
    let p attrs children = Element { Tag = "p"; Attributes = attrs; Children = children }
    let span attrs children = Element { Tag = "span"; Attributes = attrs; Children = children }
    let i attrs children = Element { Tag = "i"; Attributes = attrs; Children = children }
    let strong attrs children = Element { Tag = "strong"; Attributes = attrs; Children = children }
    let input attrs = Element { Tag = "input"; Attributes = attrs; Children = [] }
    let button attrs children = Element { Tag = "button"; Attributes = attrs; Children = children }
    let text content = Text content
    
    // Render HTML to string
    let rec renderAttribute = function
        | Class value -> sprintf "class=\"%s\"" value
        | Type value -> sprintf "type=\"%s\"" value
        | Placeholder value -> sprintf "placeholder=\"%s\"" value
        | OnClick _ -> "onclick=\"handleClick(event)\""
        | OnKeyPress _ -> "onkeypress=\"handleKeyPress(event)\""
    
    let rec renderNode = function
        | Text content -> content
        | Element element ->
            let attrs = 
                element.Attributes 
                |> List.map renderAttribute 
                |> String.concat " "
            let children = 
                element.Children 
                |> List.map renderNode 
                |> String.concat ""
            
            if List.isEmpty element.Children && element.Tag = "input" then
                sprintf "<%s %s />" element.Tag attrs
            else
                sprintf "<%s %s>%s</%s>" element.Tag attrs children element.Tag
    
    // Generate complete HTML page with Elmish runtime
    let generateElmishPage (title: string) (bodyContent: HtmlNode) =
        sprintf """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>%s</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        body { 
            background: #0d1117; color: #c9d1d9; 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        .tars-elmish-dashboard { padding: 20px; }
        .metric-card { 
            background: #161b22; border: 1px solid #30363d; 
            border-radius: 8px; padding: 20px; margin-bottom: 20px;
        }
        .metric-value { font-size: 1.5rem; font-weight: bold; color: #58a6ff; }
        .metric-label { color: #8b949e; font-size: 0.9rem; }
        .real-data { 
            background: #238636; color: white; 
            padding: 4px 8px; border-radius: 4px; font-size: 0.8rem;
        }
        .status-connected { color: #238636; }
        .status-disconnected { color: #da3633; }
        .thought-flow-container { 
            max-height: 300px; overflow-y: auto; 
            padding-right: 5px;
        }
        .thought-node { 
            background: #21262d; border-left: 3px solid #58a6ff; 
            padding: 10px; margin: 5px 0; border-radius: 4px;
        }
        .thought-content { font-weight: 500; margin-bottom: 5px; }
        .thought-meta { font-size: 0.8rem; color: #8b949e; }
        .thought-confidence { margin-right: 10px; }
        .projects-container { 
            max-height: 250px; overflow-y: auto; 
            padding-right: 5px;
        }
        .project-item { 
            background: #21262d; border: 1px solid #30363d; 
            padding: 10px; margin: 5px 0; border-radius: 4px;
        }
        .project-name { font-weight: 500; margin-bottom: 3px; }
        .project-description { font-size: 0.9rem; color: #8b949e; }
        .chat-container { 
            background: #161b22; border: 1px solid #30363d; 
            border-radius: 8px; height: 400px; display: flex; flex-direction: column;
        }
        .chat-messages { 
            flex: 1; padding: 15px; overflow-y: auto; 
        }
        .chat-input { 
            padding: 15px; border-top: 1px solid #30363d; 
            display: flex; gap: 10px;
        }
        .message { 
            margin-bottom: 10px; padding: 8px 12px; 
            border-radius: 6px; max-width: 80%%;
        }
        .message.user { 
            background: #1f6feb; color: white; 
            margin-left: auto;
        }
        .message.assistant { 
            background: #21262d; border-left: 3px solid #58a6ff; 
        }
        .message.loading { 
            background: #21262d; border-left: 3px solid #ffc107; 
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0%% { opacity: 1; }
            50%% { opacity: 0.5; }
            100%% { opacity: 1; }
        }
        .dashboard-header { margin-bottom: 30px; }
        .system-status { 
            display: flex; gap: 20px; align-items: center; 
            margin-top: 10px;
        }
        .last-update { color: #8b949e; font-size: 0.9rem; }
        
        /* Scrollbar styling */
        .thought-flow-container::-webkit-scrollbar,
        .projects-container::-webkit-scrollbar,
        .chat-messages::-webkit-scrollbar {
            width: 6px;
        }
        .thought-flow-container::-webkit-scrollbar-track,
        .projects-container::-webkit-scrollbar-track,
        .chat-messages::-webkit-scrollbar-track {
            background: #161b22;
        }
        .thought-flow-container::-webkit-scrollbar-thumb,
        .projects-container::-webkit-scrollbar-thumb,
        .chat-messages::-webkit-scrollbar-thumb {
            background: #30363d; border-radius: 3px;
        }
        .thought-flow-container::-webkit-scrollbar-thumb:hover,
        .projects-container::-webkit-scrollbar-thumb:hover,
        .chat-messages::-webkit-scrollbar-thumb:hover {
            background: #58a6ff;
        }
    </style>
</head>
<body>
    <div id="elmish-app">
        %s
    </div>
    
    <script>
        // Elmish Runtime - Handle F# functional reactive events
        let currentModel = null;
        let dispatch = null;
        
        function handleClick(event) {
            console.log('Elmish Click Event:', event);
            // F# dispatch will be injected here
        }
        
        function handleKeyPress(event) {
            console.log('Elmish KeyPress Event:', event);
            // F# dispatch will be injected here
        }
        
        // WebSocket connection for real-time Elmish updates
        function connectElmishWebSocket() {
            try {
                const ws = new WebSocket('ws://localhost:9876/ws');
                
                ws.onopen = function() {
                    console.log('Elmish WebSocket connected');
                    updateElmishModel({ type: 'WebSocketConnected', data: true });
                };
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    console.log('Elmish WebSocket message:', data);
                    updateElmishModel(data);
                };
                
                ws.onclose = function() {
                    console.log('Elmish WebSocket disconnected');
                    updateElmishModel({ type: 'WebSocketConnected', data: false });
                    setTimeout(connectElmishWebSocket, 3000);
                };
            } catch (error) {
                console.error('Elmish WebSocket connection failed:', error);
            }
        }
        
        function updateElmishModel(message) {
            // This will be replaced with real F# Elmish dispatch
            console.log('Elmish Model Update:', message);
        }
        
        // Initialize Elmish runtime
        document.addEventListener('DOMContentLoaded', function() {
            console.log('TARS Elmish Application Initialized');
            connectElmishWebSocket();
            
            // Auto-refresh every 5 seconds
            setInterval(() => {
                updateElmishModel({ type: 'RefreshAllSystems' });
            }, 5000);
        });
    </script>
</body>
</html>""" title (renderNode bodyContent)
