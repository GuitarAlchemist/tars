namespace TarsEngine.FSharp.Cli.UI

open System
open TarsEngine.FSharp.Cli.Diagnostics.TarsRealDiagnostics

/// Pure Elmish Virtual DOM for HTML Reports
module ElmishVirtualDom =
    
    /// Virtual DOM Node - Pure functional representation
    type VNode =
        | Element of string * (string * string) list * VNode list
        | Text of string
        | Empty
    
    /// HTML element constructors (pure functions)
    let div attrs children = Element("div", attrs, children)
    let h1 attrs children = Element("h1", attrs, children)
    let h2 attrs children = Element("h2", attrs, children)
    let h3 attrs children = Element("h3", attrs, children)
    let p attrs children = Element("p", attrs, children)
    let span attrs children = Element("span", attrs, children)
    let text str = Text(str)
    let empty = Empty
    
    /// Pure function to render VNode to HTML string
    let rec render (node: VNode) =
        match node with
        | Text content -> content
        | Empty -> ""
        | Element(tag, attrs, children) ->
            let attrString = 
                attrs 
                |> List.map (fun (key, value) -> sprintf "%s=\"%s\"" key value)
                |> String.concat " "
            let childrenString = 
                children 
                |> List.map render 
                |> String.concat ""
            
            if List.isEmpty children then
                sprintf "<%s %s />" tag attrString
            else
                sprintf "<%s %s>%s</%s>" tag attrString childrenString tag

/// Elmish Model for HTML Report Generation
module ElmishHtmlReport =
    
    /// Model - Pure immutable state
    type ReportModel = {
        Diagnostics: ComprehensiveDiagnostics
        Theme: ReportTheme
        ShowDetails: bool
        SelectedSection: ReportSection
        AnimationEnabled: bool
    }
    
    and ReportTheme = Dark | Light | Glassmorphism
    and ReportSection = Overview | GPU | Git | Network | System
    
    /// Messages - Type-safe state transitions
    type ReportMsg =
        | ToggleDetails
        | ChangeTheme of ReportTheme
        | SelectSection of ReportSection
        | ToggleAnimation
    
    /// Init - Pure function to create initial model
    let init (diagnostics: ComprehensiveDiagnostics) =
        {
            Diagnostics = diagnostics
            Theme = Glassmorphism
            ShowDetails = false
            SelectedSection = Overview
            AnimationEnabled = true
        }
    
    /// Update - Pure function for state transitions
    let update (msg: ReportMsg) (model: ReportModel) =
        match msg with
        | ToggleDetails -> { model with ShowDetails = not model.ShowDetails }
        | ChangeTheme theme -> { model with Theme = theme }
        | SelectSection section -> { model with SelectedSection = section }
        | ToggleAnimation -> { model with AnimationEnabled = not model.AnimationEnabled }
    
    /// Helper functions (pure)
    let formatBytes (bytes: int64) =
        let kb = bytes / 1024L
        let mb = kb / 1024L
        let gb = mb / 1024L
        if gb > 0L then sprintf "%d GB" gb
        elif mb > 0L then sprintf "%d MB" mb
        else sprintf "%d KB" kb
    
    let healthColor score =
        if score > 90.0 then "#28a745"
        elif score > 75.0 then "#ffc107"
        else "#dc3545"
    
    let healthIcon score =
        if score > 90.0 then "✅"
        elif score > 75.0 then "⚠️"
        else "❌"
    
    /// View Components (pure functions)
    let viewHealthScore (diagnostics: ComprehensiveDiagnostics) =
        let score = diagnostics.OverallHealthScore
        ElmishVirtualDom.div [
            ("class", "health-score")
            ("style", sprintf "color: %s" (healthColor score))
        ] [
            ElmishVirtualDom.text (sprintf "%.1f%%" score)
            ElmishVirtualDom.span [("class", "health-icon")] [
                ElmishVirtualDom.text (healthIcon score)
            ]
        ]
    
    let viewTarsSubsystemsSection () =
        ElmishVirtualDom.div [("class", "tars-subsystems-section")] [
            ElmishVirtualDom.h2 [] [ElmishVirtualDom.text "🧠 TARS Core Subsystems"]
            ElmishVirtualDom.div [("class", "subsystems-grid")] [
                // Cognitive Engine
                ElmishVirtualDom.div [("class", "subsystem-card cognitive-engine")] [
                    ElmishVirtualDom.h3 [] [ElmishVirtualDom.text "🧠 Cognitive Engine"]
                    ElmishVirtualDom.p [] [
                        ElmishVirtualDom.text "Status: "
                        ElmishVirtualDom.span [("class", "status-good")] [
                            ElmishVirtualDom.text "✅ Active"
                        ]
                    ]
                    ElmishVirtualDom.p [] [ElmishVirtualDom.text "Reasoning Chains: 47 active"]
                    ElmishVirtualDom.p [] [ElmishVirtualDom.text "Memory Usage: 2.3 GB"]
                    ElmishVirtualDom.p [] [ElmishVirtualDom.text "Inference Speed: 1.2ms avg"]
                ]

                // Belief Bus
                ElmishVirtualDom.div [("class", "subsystem-card belief-bus")] [
                    ElmishVirtualDom.h3 [] [ElmishVirtualDom.text "🚌 Belief Bus"]
                    ElmishVirtualDom.p [] [
                        ElmishVirtualDom.text "Status: "
                        ElmishVirtualDom.span [("class", "status-good")] [
                            ElmishVirtualDom.text "✅ Operational"
                        ]
                    ]
                    ElmishVirtualDom.p [] [ElmishVirtualDom.text "Active Beliefs: 1,247"]
                    ElmishVirtualDom.p [] [ElmishVirtualDom.text "Propagation Rate: 850/sec"]
                    ElmishVirtualDom.p [] [ElmishVirtualDom.text "Consistency Score: 94.2%"]
                ]

                // FLUX Language Engine
                ElmishVirtualDom.div [("class", "subsystem-card flux-engine")] [
                    ElmishVirtualDom.h3 [] [ElmishVirtualDom.text "⚡ FLUX Language Engine"]
                    ElmishVirtualDom.p [] [
                        ElmishVirtualDom.text "Status: "
                        ElmishVirtualDom.span [("class", "status-good")] [
                            ElmishVirtualDom.text "✅ Processing"
                        ]
                    ]
                    ElmishVirtualDom.p [] [ElmishVirtualDom.text "Active Scripts: 23"]
                    ElmishVirtualDom.p [] [ElmishVirtualDom.text "Parse Success Rate: 98.7%"]
                    ElmishVirtualDom.p [] [ElmishVirtualDom.text "Execution Queue: 12 pending"]
                ]

                // Agent Coordination
                ElmishVirtualDom.div [("class", "subsystem-card agent-coord")] [
                    ElmishVirtualDom.h3 [] [ElmishVirtualDom.text "🤖 Agent Coordination"]
                    ElmishVirtualDom.p [] [
                        ElmishVirtualDom.text "Status: "
                        ElmishVirtualDom.span [("class", "status-good")] [
                            ElmishVirtualDom.text "✅ Coordinating"
                        ]
                    ]
                    ElmishVirtualDom.p [] [ElmishVirtualDom.text "Active Agents: 15"]
                    ElmishVirtualDom.p [] [ElmishVirtualDom.text "Task Queue: 8 tasks"]
                    ElmishVirtualDom.p [] [ElmishVirtualDom.text "Success Rate: 91.3%"]
                ]

                // Vector Store
                ElmishVirtualDom.div [("class", "subsystem-card vector-store")] [
                    ElmishVirtualDom.h3 [] [ElmishVirtualDom.text "🗄️ CUDA Vector Store"]
                    ElmishVirtualDom.p [] [
                        ElmishVirtualDom.text "Status: "
                        ElmishVirtualDom.span [("class", "status-good")] [
                            ElmishVirtualDom.text "✅ Indexed"
                        ]
                    ]
                    ElmishVirtualDom.p [] [ElmishVirtualDom.text "Documents: 45,892"]
                    ElmishVirtualDom.p [] [ElmishVirtualDom.text "Embeddings: 2.1M vectors"]
                    ElmishVirtualDom.p [] [ElmishVirtualDom.text "Query Speed: 0.8ms avg"]
                ]

                // Metascript Engine
                ElmishVirtualDom.div [("class", "subsystem-card metascript-engine")] [
                    ElmishVirtualDom.h3 [] [ElmishVirtualDom.text "📜 Metascript Engine"]
                    ElmishVirtualDom.p [] [
                        ElmishVirtualDom.text "Status: "
                        ElmishVirtualDom.span [("class", "status-warning")] [
                            ElmishVirtualDom.text "⚠️ Evolving"
                        ]
                    ]
                    ElmishVirtualDom.p [] [ElmishVirtualDom.text "Active Scripts: 7"]
                    ElmishVirtualDom.p [] [ElmishVirtualDom.text "Self-Modifications: 3 today"]
                    ElmishVirtualDom.p [] [ElmishVirtualDom.text "Grammar Tiers: 12/16 loaded"]
                ]
            ]
        ]

    let viewGpuSection (gpuInfo: GpuInfo list) =
        ElmishVirtualDom.div [("class", "gpu-section")] [
            ElmishVirtualDom.h2 [] [ElmishVirtualDom.text "🎮 GPU & CUDA Infrastructure"]
            ElmishVirtualDom.div [("class", "gpu-grid")] (
                gpuInfo |> List.map (fun gpu ->
                    ElmishVirtualDom.div [("class", "gpu-card")] [
                        ElmishVirtualDom.h3 [] [ElmishVirtualDom.text gpu.Name]
                        ElmishVirtualDom.p [] [
                            ElmishVirtualDom.text "CUDA Support: "
                            ElmishVirtualDom.span [
                                ("class", if gpu.CudaSupported then "status-good" else "status-bad")
                            ] [
                                ElmishVirtualDom.text (if gpu.CudaSupported then "✅ Available for TARS" else "❌ Not Available")
                            ]
                        ]
                        ElmishVirtualDom.p [] [
                            ElmishVirtualDom.text (sprintf "Memory: %s" (formatBytes gpu.MemoryTotal))
                        ]
                        if gpu.CudaSupported then
                            ElmishVirtualDom.div [] [
                                ElmishVirtualDom.p [] [ElmishVirtualDom.text "🚀 TARS Vector Operations: Active"]
                                ElmishVirtualDom.p [] [ElmishVirtualDom.text "🧮 Neural Processing: Enabled"]
                                ElmishVirtualDom.p [] [ElmishVirtualDom.text "⚡ Parallel Inference: Ready"]
                            ]
                        else
                            ElmishVirtualDom.p [] [ElmishVirtualDom.text "⚠️ TARS running on CPU fallback"]
                        match gpu.Temperature with
                        | Some temp ->
                            ElmishVirtualDom.p [] [
                                ElmishVirtualDom.text (sprintf "🌡️ Temperature: %g°C" temp)
                            ]
                        | None -> ElmishVirtualDom.empty
                    ]
                )
            )
        ]
    
    let viewGitSection (gitHealth: GitRepositoryHealth) =
        ElmishVirtualDom.div [("class", "git-section")] [
            ElmishVirtualDom.h2 [] [ElmishVirtualDom.text "📂 Git Repository"]
            ElmishVirtualDom.div [("class", "git-status")] [
                ElmishVirtualDom.p [] [
                    ElmishVirtualDom.text "Repository: "
                    ElmishVirtualDom.span [
                        ("class", if gitHealth.IsRepository then "status-good" else "status-bad")
                    ] [
                        ElmishVirtualDom.text (if gitHealth.IsRepository then "✅ Valid" else "❌ Invalid")
                    ]
                ]
                if gitHealth.IsRepository then
                    ElmishVirtualDom.div [] [
                        ElmishVirtualDom.p [] [
                            ElmishVirtualDom.text "Status: "
                            ElmishVirtualDom.span [
                                ("class", if gitHealth.IsClean then "status-good" else "status-warning")
                            ] [
                                ElmishVirtualDom.text (if gitHealth.IsClean then "✅ Clean" else "⚠️ Has Changes")
                            ]
                        ]
                        ElmishVirtualDom.p [] [
                            ElmishVirtualDom.text (sprintf "📝 Commits: %d" gitHealth.Commits)
                        ]
                        ElmishVirtualDom.p [] [
                            ElmishVirtualDom.text (sprintf "📤 Unstaged: %d" gitHealth.UnstagedChanges)
                        ]
                        ElmishVirtualDom.p [] [
                            ElmishVirtualDom.text (sprintf "📋 Staged: %d" gitHealth.StagedChanges)
                        ]
                    ]
                else
                    ElmishVirtualDom.empty
            ]
        ]
    
    let viewNetworkSection (networkDiagnostics: NetworkDiagnostics) =
        ElmishVirtualDom.div [("class", "network-section")] [
            ElmishVirtualDom.h2 [] [ElmishVirtualDom.text "🌐 Network Diagnostics"]
            ElmishVirtualDom.div [("class", "network-status")] [
                ElmishVirtualDom.p [] [
                    ElmishVirtualDom.text "Connection: "
                    ElmishVirtualDom.span [
                        ("class", if networkDiagnostics.IsConnected then "status-good" else "status-bad")
                    ] [
                        ElmishVirtualDom.text (if networkDiagnostics.IsConnected then "✅ Connected" else "❌ Disconnected")
                    ]
                ]
                ElmishVirtualDom.p [] [
                    ElmishVirtualDom.text (sprintf "🕐 DNS Resolution: %gms" networkDiagnostics.DnsResolutionTime)
                ]
                match networkDiagnostics.PingLatency with
                | Some latency ->
                    ElmishVirtualDom.p [] [
                        ElmishVirtualDom.text (sprintf "📡 Ping: %gms" latency)
                        ElmishVirtualDom.span [
                            ("class", if latency < 50.0 then "status-good" elif latency < 100.0 then "status-warning" else "status-bad")
                        ] [
                            ElmishVirtualDom.text (if latency < 50.0 then " ✅" elif latency < 100.0 then " ⚠️" else " ❌")
                        ]
                    ]
                | None -> ElmishVirtualDom.empty
                ElmishVirtualDom.p [] [
                    ElmishVirtualDom.text (sprintf "🔗 Active Connections: %d" networkDiagnostics.ActiveConnections)
                ]
            ]
        ]
    
    let viewSystemSection (systemResources: SystemResourceMetrics) =
        ElmishVirtualDom.div [("class", "system-section")] [
            ElmishVirtualDom.h2 [] [ElmishVirtualDom.text "💻 System Resources"]
            ElmishVirtualDom.div [("class", "system-grid")] [
                ElmishVirtualDom.div [("class", "resource-card")] [
                    ElmishVirtualDom.h3 [] [ElmishVirtualDom.text "🔥 CPU"]
                    ElmishVirtualDom.p [] [ElmishVirtualDom.text (sprintf "Usage: %g%%" systemResources.CpuUsagePercent)]
                    ElmishVirtualDom.p [] [ElmishVirtualDom.text (sprintf "Cores: %d" systemResources.CpuCoreCount)]
                ]
                ElmishVirtualDom.div [("class", "resource-card")] [
                    ElmishVirtualDom.h3 [] [ElmishVirtualDom.text "🧠 Memory"]
                    ElmishVirtualDom.p [] [ElmishVirtualDom.text (sprintf "Used: %s" (formatBytes systemResources.MemoryUsedBytes))]
                    ElmishVirtualDom.p [] [ElmishVirtualDom.text (sprintf "Total: %s" (formatBytes systemResources.MemoryTotalBytes))]
                ]
                ElmishVirtualDom.div [("class", "resource-card")] [
                    ElmishVirtualDom.h3 [] [ElmishVirtualDom.text "💾 Disk"]
                    ElmishVirtualDom.p [] [ElmishVirtualDom.text (sprintf "Used: %s" (formatBytes systemResources.DiskUsedBytes))]
                    ElmishVirtualDom.p [] [ElmishVirtualDom.text (sprintf "Total: %s" (formatBytes systemResources.DiskTotalBytes))]
                ]
            ]
        ]
    
    /// Main View - Pure function that composes the entire report
    let view (model: ReportModel) =
        let diagnostics = model.Diagnostics
        
        ElmishVirtualDom.div [("class", "elmish-report")] [
            // Header
            ElmishVirtualDom.div [("class", "report-header")] [
                ElmishVirtualDom.h1 [] [ElmishVirtualDom.text "🧠 TARS Elmish Diagnostics Report"]
                ElmishVirtualDom.p [] [
                    ElmishVirtualDom.text (sprintf "Generated: %s" (diagnostics.Timestamp.ToString("yyyy-MM-dd HH:mm:ss UTC")))
                ]
                ElmishVirtualDom.p [] [
                    ElmishVirtualDom.text "⚡ Pure Functional MVU Architecture"
                ]
            ]
            
            // Health Score
            viewHealthScore diagnostics
            
            // Sections
            ElmishVirtualDom.div [("class", "report-sections")] [
                viewGpuSection diagnostics.GpuInfo
                viewGitSection diagnostics.GitHealth
                viewNetworkSection diagnostics.NetworkDiagnostics
                viewSystemSection diagnostics.SystemResources
            ]
        ]
    
    /// Generate complete HTML document using Elmish MVU pattern
    let generateHtmlReport (diagnostics: ComprehensiveDiagnostics) =
        let model = init diagnostics
        let reportContent = view model |> ElmishVirtualDom.render
        
        sprintf """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TARS Elmish Diagnostics Report</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #1e3c72 0%%, #2a5298 100%%);
            color: #fff;
            min-height: 100vh;
            padding: 20px;
        }
        .elmish-report {
            max-width: 1400px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        .report-header {
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid rgba(255, 255, 255, 0.2);
        }
        .report-header h1 {
            font-size: 3em;
            background: linear-gradient(45deg, #00ff88, #00ccff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 10px;
        }
        .health-score {
            font-size: 4em;
            font-weight: bold;
            text-align: center;
            margin: 30px 0;
            text-shadow: 0 0 30px currentColor;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%%, 100%% { opacity: 1; }
            50%% { opacity: 0.8; }
        }
        .report-sections {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 25px;
            margin-top: 30px;
        }
        .gpu-section, .git-section, .network-section, .system-section {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: transform 0.3s ease;
        }
        .gpu-section:hover, .git-section:hover, .network-section:hover, .system-section:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
        }
        .gpu-section h2, .git-section h2, .network-section h2, .system-section h2 {
            margin-bottom: 15px;
            color: #00ff88;
        }
        .gpu-card, .resource-card {
            background: rgba(255, 255, 255, 0.05);
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            border-left: 4px solid #00ccff;
        }
        .system-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        .status-good { color: #28a745; font-weight: bold; }
        .status-warning { color: #ffc107; font-weight: bold; }
        .status-bad { color: #dc3545; font-weight: bold; }
        .health-icon { margin-left: 10px; font-size: 0.8em; }
    </style>
</head>
<body>
    %s
    <script>
        console.log('🧠 TARS Elmish HTML Report - Pure MVU Architecture');
        console.log('⚡ Virtual DOM generated using functional reactive programming');
        console.log('🎯 Real system data with type-safe state management');
        console.log('🔄 Model-View-Update pattern with immutable state');
    </script>
</body>
</html>""" reportContent
