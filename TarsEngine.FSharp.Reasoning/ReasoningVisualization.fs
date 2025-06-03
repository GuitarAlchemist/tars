namespace TarsEngine.FSharp.Reasoning

open System
open System.Text
open System.Collections.Generic
open Microsoft.Extensions.Logging

/// Visualization types for reasoning
type VisualizationType =
    | ReasoningTree         // Hierarchical tree structure
    | ThoughtGraph         // Network graph of connections
    | TemporalFlow         // Time-based progression
    | ConfidenceHeatmap    // Confidence distribution
    | AlternativeExplorer  // Interactive path comparison

/// Visualization node
type VisualizationNode = {
    Id: string
    Label: string
    Type: string
    Content: string
    Confidence: float
    Position: (float * float) option
    Style: Map<string, string>
    Metadata: Map<string, obj>
}

/// Visualization edge
type VisualizationEdge = {
    Id: string
    Source: string
    Target: string
    Label: string option
    Weight: float
    Style: Map<string, string>
    Metadata: Map<string, obj>
}

/// Complete visualization
type ReasoningVisualization = {
    VisualizationId: string
    Type: VisualizationType
    Title: string
    Description: string
    Nodes: VisualizationNode list
    Edges: VisualizationEdge list
    Layout: string
    InteractiveFeatures: string list
    RenderingHints: Map<string, obj>
    CreatedAt: DateTime
}

/// Interactive visualization feature
type InteractiveFeature =
    | StepInspection of string      // Detailed step examination
    | AlternativeExploration        // Explore different paths
    | ConfidenceDrilling           // Investigate confidence sources
    | ReasoningReplay              // Step through process
    | CollaborativeAnnotation      // Team review and annotation

/// Visualization export format
type ExportFormat =
    | SVG
    | PNG
    | HTML
    | JSON
    | Mermaid
    | GraphViz

/// Interface for reasoning visualization
type IReasoningVisualization =
    abstract member CreateVisualization: ChainOfThought -> VisualizationType -> ReasoningVisualization
    abstract member RenderToString: ReasoningVisualization -> ExportFormat -> string
    abstract member CreateInteractiveHtml: ReasoningVisualization -> string
    abstract member GenerateMermaidDiagram: ChainOfThought -> string

/// Reasoning visualization implementation
type ReasoningVisualizationEngine(logger: ILogger<ReasoningVisualizationEngine>) =
    
    /// Generate color based on confidence level
    let getConfidenceColor (confidence: float) =
        match confidence with
        | c when c >= 0.8 -> "#2E7D32"  // Dark green
        | c when c >= 0.6 -> "#FBC02D"  // Yellow
        | c when c >= 0.4 -> "#F57C00"  // Orange
        | _ -> "#D32F2F"                // Red
    
    /// Generate step type icon
    let getStepTypeIcon (stepType: ReasoningStepType) =
        match stepType with
        | Observation -> "👁️"
        | Hypothesis -> "💡"
        | Deduction -> "🔍"
        | Induction -> "📊"
        | Abduction -> "🎯"
        | Causal -> "🔗"
        | Analogical -> "🔄"
        | Meta -> "🤔"
        | Synthesis -> "🔀"
        | Validation -> "✅"
    
    /// Create reasoning tree visualization
    let createReasoningTree (chain: ChainOfThought) =
        let nodes = 
            chain.Steps
            |> List.mapi (fun i step ->
                {
                    Id = step.Id
                    Label = $"{getStepTypeIcon step.StepType} Step {step.StepNumber}"
                    Type = step.StepType.ToString()
                    Content = step.Content
                    Confidence = step.Confidence
                    Position = Some (float i * 150.0, float step.StepNumber * 100.0)
                    Style = Map.ofList [
                        ("fill", getConfidenceColor step.Confidence)
                        ("stroke", "#333")
                        ("strokeWidth", "2")
                    ]
                    Metadata = Map.ofList [
                        ("stepNumber", box step.StepNumber)
                        ("complexityScore", box step.ComplexityScore)
                        ("processingTime", box step.ProcessingTime.TotalMilliseconds)
                    ]
                })
        
        let edges = 
            chain.Steps
            |> List.pairwise
            |> List.map (fun (prev, curr) ->
                {
                    Id = $"{prev.Id}->{curr.Id}"
                    Source = prev.Id
                    Target = curr.Id
                    Label = Some "leads to"
                    Weight = 1.0
                    Style = Map.ofList [
                        ("stroke", "#666")
                        ("strokeWidth", "1.5")
                        ("markerEnd", "url(#arrowhead)")
                    ]
                    Metadata = Map.empty
                })
        
        {
            VisualizationId = Guid.NewGuid().ToString()
            Type = ReasoningTree
            Title = $"Reasoning Tree: {chain.Problem}"
            Description = $"Hierarchical view of {chain.Steps.Length} reasoning steps"
            Nodes = nodes
            Edges = edges
            Layout = "tree"
            InteractiveFeatures = ["step_inspection"; "confidence_drilling"]
            RenderingHints = Map.ofList [
                ("direction", box "top-bottom")
                ("nodeSpacing", box 100)
                ("levelSpacing", box 150)
            ]
            CreatedAt = DateTime.UtcNow
        }
    
    /// Create thought graph visualization
    let createThoughtGraph (chain: ChainOfThought) =
        let nodes = 
            chain.Steps
            |> List.map (fun step ->
                {
                    Id = step.Id
                    Label = $"{getStepTypeIcon step.StepType}"
                    Type = step.StepType.ToString()
                    Content = step.Content
                    Confidence = step.Confidence
                    Position = None  // Let layout algorithm decide
                    Style = Map.ofList [
                        ("fill", getConfidenceColor step.Confidence)
                        ("radius", string (10 + int (step.Confidence * 20.0)))
                        ("stroke", "#333")
                    ]
                    Metadata = Map.ofList [
                        ("stepType", box step.StepType)
                        ("confidence", box step.Confidence)
                    ]
                })
        
        // Create edges based on dependencies and logical flow
        let edges = 
            [
                // Sequential edges
                yield! chain.Steps
                       |> List.pairwise
                       |> List.map (fun (prev, curr) ->
                           {
                               Id = $"seq_{prev.Id}_{curr.Id}"
                               Source = prev.Id
                               Target = curr.Id
                               Label = Some "sequence"
                               Weight = 0.5
                               Style = Map.ofList [("stroke", "#999"); ("strokeDasharray", "5,5")]
                               Metadata = Map.ofList [("type", box "sequential")]
                           })
                
                // Dependency edges
                yield! chain.Steps
                       |> List.collect (fun step ->
                           step.Dependencies
                           |> List.map (fun depId ->
                               {
                                   Id = $"dep_{depId}_{step.Id}"
                                   Source = depId
                                   Target = step.Id
                                   Label = Some "depends on"
                                   Weight = 1.0
                                   Style = Map.ofList [("stroke", "#333"); ("strokeWidth", "2")]
                                   Metadata = Map.ofList [("type", box "dependency")]
                               }))
            ]
        
        {
            VisualizationId = Guid.NewGuid().ToString()
            Type = ThoughtGraph
            Title = $"Thought Graph: {chain.Problem}"
            Description = $"Network view showing relationships between {chain.Steps.Length} reasoning steps"
            Nodes = nodes
            Edges = edges
            Layout = "force-directed"
            InteractiveFeatures = ["step_inspection"; "alternative_exploration"; "reasoning_replay"]
            RenderingHints = Map.ofList [
                ("forceStrength", box 0.3)
                ("linkDistance", box 100)
                ("chargeStrength", box -300)
            ]
            CreatedAt = DateTime.UtcNow
        }
    
    /// Create temporal flow visualization
    let createTemporalFlow (chain: ChainOfThought) =
        let nodes = 
            chain.Steps
            |> List.mapi (fun i step ->
                let timeOffset = float i * 50.0  // Simulate time progression
                {
                    Id = step.Id
                    Label = $"T{i+1}: {getStepTypeIcon step.StepType}"
                    Type = step.StepType.ToString()
                    Content = step.Content
                    Confidence = step.Confidence
                    Position = Some (timeOffset, step.Confidence * 200.0)
                    Style = Map.ofList [
                        ("fill", getConfidenceColor step.Confidence)
                        ("stroke", "#333")
                        ("opacity", string step.Confidence)
                    ]
                    Metadata = Map.ofList [
                        ("timeIndex", box i)
                        ("processingTime", box step.ProcessingTime.TotalMilliseconds)
                    ]
                })
        
        let edges = 
            chain.Steps
            |> List.pairwise
            |> List.map (fun (prev, curr) ->
                {
                    Id = $"flow_{prev.Id}_{curr.Id}"
                    Source = prev.Id
                    Target = curr.Id
                    Label = Some "→"
                    Weight = 1.0
                    Style = Map.ofList [
                        ("stroke", "#666")
                        ("strokeWidth", "2")
                        ("markerEnd", "url(#arrowhead)")
                    ]
                    Metadata = Map.empty
                })
        
        {
            VisualizationId = Guid.NewGuid().ToString()
            Type = TemporalFlow
            Title = $"Temporal Flow: {chain.Problem}"
            Description = $"Time-based progression of reasoning steps"
            Nodes = nodes
            Edges = edges
            Layout = "timeline"
            InteractiveFeatures = ["reasoning_replay"; "step_inspection"]
            RenderingHints = Map.ofList [
                ("timeAxis", box "horizontal")
                ("confidenceAxis", box "vertical")
                ("showTimeline", box true)
            ]
            CreatedAt = DateTime.UtcNow
        }
    
    /// Generate Mermaid diagram
    let generateMermaidDiagram (chain: ChainOfThought) =
        let sb = StringBuilder()
        
        sb.AppendLine("graph TD") |> ignore
        sb.AppendLine($"    classDef high fill:#2E7D32,stroke:#333,stroke-width:2px") |> ignore
        sb.AppendLine($"    classDef medium fill:#FBC02D,stroke:#333,stroke-width:2px") |> ignore
        sb.AppendLine($"    classDef low fill:#D32F2F,stroke:#333,stroke-width:2px") |> ignore
        sb.AppendLine() |> ignore
        
        // Add nodes
        for step in chain.Steps do
            let nodeId = $"S{step.StepNumber}"
            let label = $"{getStepTypeIcon step.StepType} {step.StepType}"
            let cssClass = 
                if step.Confidence >= 0.7 then "high"
                elif step.Confidence >= 0.5 then "medium"
                else "low"
            
            sb.AppendLine($"    {nodeId}[\"{label}\"]") |> ignore
            sb.AppendLine($"    class {nodeId} {cssClass}") |> ignore
        
        sb.AppendLine() |> ignore
        
        // Add edges
        for i in 0 .. chain.Steps.Length - 2 do
            let sourceId = $"S{chain.Steps.[i].StepNumber}"
            let targetId = $"S{chain.Steps.[i+1].StepNumber}"
            sb.AppendLine($"    {sourceId} --> {targetId}") |> ignore
        
        sb.ToString()
    
    interface IReasoningVisualization with
        
        member this.CreateVisualization(chain: ChainOfThought) (visualizationType: VisualizationType) =
            try
                logger.LogInformation($"Creating {visualizationType} visualization for chain: {chain.ChainId}")
                
                match visualizationType with
                | ReasoningTree -> createReasoningTree chain
                | ThoughtGraph -> createThoughtGraph chain
                | TemporalFlow -> createTemporalFlow chain
                | ConfidenceHeatmap -> 
                    // TODO: Implement confidence heatmap
                    createReasoningTree chain  // Fallback for now
                | AlternativeExplorer -> 
                    // TODO: Implement alternative explorer
                    createThoughtGraph chain  // Fallback for now
                    
            with
            | ex ->
                logger.LogError(ex, $"Error creating visualization for chain: {chain.ChainId}")
                {
                    VisualizationId = Guid.NewGuid().ToString()
                    Type = visualizationType
                    Title = "Error Visualization"
                    Description = $"Error creating visualization: {ex.Message}"
                    Nodes = []
                    Edges = []
                    Layout = "error"
                    InteractiveFeatures = []
                    RenderingHints = Map.empty
                    CreatedAt = DateTime.UtcNow
                }
        
        member this.RenderToString(visualization: ReasoningVisualization) (format: ExportFormat) =
            try
                match format with
                | JSON ->
                    // TODO: Implement JSON serialization
                    $"{{\"visualization\": \"{visualization.Title}\", \"nodes\": {visualization.Nodes.Length}, \"edges\": {visualization.Edges.Length}}}"
                
                | HTML ->
                    let sb = StringBuilder()
                    sb.AppendLine("<!DOCTYPE html>") |> ignore
                    sb.AppendLine("<html>") |> ignore
                    sb.AppendLine("<head>") |> ignore
                    sb.AppendLine($"<title>{visualization.Title}</title>") |> ignore
                    sb.AppendLine("</head>") |> ignore
                    sb.AppendLine("<body>") |> ignore
                    sb.AppendLine($"<h1>{visualization.Title}</h1>") |> ignore
                    sb.AppendLine($"<p>{visualization.Description}</p>") |> ignore
                    sb.AppendLine($"<p>Nodes: {visualization.Nodes.Length}, Edges: {visualization.Edges.Length}</p>") |> ignore
                    sb.AppendLine("</body>") |> ignore
                    sb.AppendLine("</html>") |> ignore
                    sb.ToString()
                
                | Mermaid ->
                    // Return Mermaid diagram syntax
                    let sb = StringBuilder()
                    sb.AppendLine("graph TD") |> ignore
                    for node in visualization.Nodes do
                        sb.AppendLine($"    {node.Id}[\"{node.Label}\"]") |> ignore
                    for edge in visualization.Edges do
                        sb.AppendLine($"    {edge.Source} --> {edge.Target}") |> ignore
                    sb.ToString()
                
                | _ ->
                    $"Export format {format} not yet implemented"
                    
            with
            | ex ->
                logger.LogError(ex, $"Error rendering visualization to {format}")
                $"Error rendering visualization: {ex.Message}"
        
        member this.CreateInteractiveHtml(visualization: ReasoningVisualization) =
            try
                let sb = StringBuilder()
                
                sb.AppendLine("<!DOCTYPE html>") |> ignore
                sb.AppendLine("<html>") |> ignore
                sb.AppendLine("<head>") |> ignore
                sb.AppendLine($"<title>Interactive {visualization.Title}</title>") |> ignore
                sb.AppendLine("<script src=\"https://d3js.org/d3.v7.min.js\"></script>") |> ignore
                sb.AppendLine("<style>") |> ignore
                sb.AppendLine(".node { cursor: pointer; }") |> ignore
                sb.AppendLine(".node:hover { opacity: 0.8; }") |> ignore
                sb.AppendLine(".edge { stroke: #999; stroke-width: 1.5px; }") |> ignore
                sb.AppendLine("#tooltip { position: absolute; background: rgba(0,0,0,0.8); color: white; padding: 10px; border-radius: 5px; pointer-events: none; }") |> ignore
                sb.AppendLine("</style>") |> ignore
                sb.AppendLine("</head>") |> ignore
                sb.AppendLine("<body>") |> ignore
                sb.AppendLine($"<h1>{visualization.Title}</h1>") |> ignore
                sb.AppendLine($"<p>{visualization.Description}</p>") |> ignore
                sb.AppendLine("<div id=\"visualization\"></div>") |> ignore
                sb.AppendLine("<div id=\"tooltip\" style=\"display: none;\"></div>") |> ignore
                
                // Add JavaScript for interactivity
                sb.AppendLine("<script>") |> ignore
                sb.AppendLine("// Interactive visualization code would go here") |> ignore
                sb.AppendLine($"console.log('Loaded visualization with {visualization.Nodes.Length} nodes and {visualization.Edges.Length} edges');") |> ignore
                sb.AppendLine("</script>") |> ignore
                
                sb.AppendLine("</body>") |> ignore
                sb.AppendLine("</html>") |> ignore
                
                sb.ToString()
                
            with
            | ex ->
                logger.LogError(ex, "Error creating interactive HTML")
                $"<html><body><h1>Error</h1><p>Error creating interactive visualization: {ex.Message}</p></body></html>"
        
        member this.GenerateMermaidDiagram(chain: ChainOfThought) =
            generateMermaidDiagram chain

/// Factory for creating reasoning visualization engines
module ReasoningVisualizationFactory =
    
    let create (logger: ILogger<ReasoningVisualizationEngine>) =
        new ReasoningVisualizationEngine(logger) :> IReasoningVisualization
