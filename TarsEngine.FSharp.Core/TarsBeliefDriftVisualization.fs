// ================================================
// 📊 TARS Belief Drift Visualization
// ================================================
// Timeline generation, partition evolution tracking, and interactive dashboard
// Based on ChatGPT-Leveraging Primes for TARS document

namespace TarsEngine.FSharp.Core

open System
open System.Collections.Generic
open System.Text
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.TarsRsxDiff
open TarsEngine.FSharp.Core.TarsSedenionPartitioner
open TarsEngine.FSharp.Core.TarsAutoReflection

/// Represents a point in time for belief tracking
type BeliefTimestamp = {
    Timestamp: DateTime
    SessionId: string
    SequenceNumber: int
}

/// Represents a belief state at a specific time
type BeliefDriftState = {
    Id: string
    Timestamp: BeliefTimestamp
    PartitionStructure: BspTree
    Insights: Insight list
    Contradictions: Contradiction list
    Significance: float
    Coherence: float
    Metadata: Map<string, string>
}

/// Represents drift between two belief states
type BeliefDrift = {
    Id: string
    FromState: BeliefDriftState
    ToState: BeliefDriftState
    DriftMagnitude: float
    DriftDirection: string // "convergence", "divergence", "oscillation"
    AffectedDimensions: int list
    SignificanceChange: float
    CoherenceChange: float
    NewInsights: Insight list
    ResolvedContradictions: Contradiction list
    EmergentContradictions: Contradiction list
}

/// Represents a timeline of belief evolution
type BeliefTimeline = {
    Id: string
    StartTime: DateTime
    EndTime: DateTime
    States: BeliefDriftState list
    Drifts: BeliefDrift list
    TotalDriftMagnitude: float
    OverallDirection: string
    KeyInflectionPoints: BeliefTimestamp list
}

/// Visualization data for dashboard
type VisualizationData = {
    Timeline: BeliefTimeline
    DriftChart: (DateTime * float) list
    CoherenceChart: (DateTime * float) list
    InsightChart: (DateTime * int) list
    ContradictionChart: (DateTime * int) list
    DimensionHeatmap: Map<int, float list>
}

/// Result type for visualization operations
type VisualizationResult<'T> = 
    | Success of 'T
    | Error of string

module TarsBeliefDriftVisualization =

    /// Generate unique ID for belief tracking
    let generateBeliefId (prefix: string) : string =
        let timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()
        let random = Random().Next(1000, 9999)
        sprintf "%s-%d-%d" prefix timestamp random

    /// Create a belief timestamp
    let createBeliefTimestamp (sessionId: string) (sequenceNumber: int) : BeliefTimestamp =
        {
            Timestamp = DateTime.UtcNow
            SessionId = sessionId
            SequenceNumber = sequenceNumber
        }

    /// Calculate partition coherence
    let calculatePartitionCoherence (tree: BspTree) : float =
        let rec analyzeNode (node: BspNode option) : (int * float) =
            match node with
            | None -> (0, 0.0)
            | Some n ->
                let pointCount = n.Points.Length
                if pointCount < 2 then (pointCount, 1.0)
                else
                    // Calculate coherence based on point clustering
                    let mutable totalDistance = 0.0
                    let mutable pairCount = 0
                    
                    for i in 0..pointCount-2 do
                        for j in i+1..pointCount-1 do
                            let distance = 
                                Array.zip n.Points.[i].Components n.Points.[j].Components
                                |> Array.map (fun (a, b) -> (a - b) * (a - b))
                                |> Array.sum
                                |> sqrt
                            totalDistance <- totalDistance + distance
                            pairCount <- pairCount + 1
                    
                    let avgDistance = if pairCount > 0 then totalDistance / float pairCount else 0.0
                    let coherence = 1.0 / (1.0 + avgDistance)
                    
                    let (leftCount, leftCoherence) = analyzeNode n.LeftChild
                    let (rightCount, rightCoherence) = analyzeNode n.RightChild
                    
                    let totalPoints = pointCount + leftCount + rightCount
                    let weightedCoherence = 
                        (float pointCount * coherence + 
                         float leftCount * leftCoherence + 
                         float rightCount * rightCoherence) / float totalPoints
                    
                    (totalPoints, weightedCoherence)
        
        let (_, coherence) = analyzeNode tree.Root
        coherence

    /// Create a belief state from current system state
    let createBeliefState (sessionId: string) (sequenceNumber: int) (tree: BspTree)
                         (insights: Insight list) (contradictions: Contradiction list) : BeliefDriftState =
        let timestamp = createBeliefTimestamp sessionId sequenceNumber
        let coherence = calculatePartitionCoherence tree
        let significance = 
            if insights.Length > 0 then
                insights |> List.map (fun i -> i.Significance) |> List.average
            else 0.0
        
        {
            Id = generateBeliefId "belief"
            Timestamp = timestamp
            PartitionStructure = tree
            Insights = insights
            Contradictions = contradictions
            Significance = significance
            Coherence = coherence
            Metadata = Map [
                ("node_count", string tree.TotalPoints)
                ("max_depth", string tree.MaxDepth)
                ("insight_count", string insights.Length)
                ("contradiction_count", string contradictions.Length)
            ]
        }

    /// Calculate drift between two belief states
    let calculateBeliefDrift (fromState: BeliefDriftState) (toState: BeliefDriftState) : BeliefDrift =
        // Calculate magnitude based on structural and semantic changes
        let structuralDrift = 
            let fromNodes = fromState.PartitionStructure.TotalPoints
            let toNodes = toState.PartitionStructure.TotalPoints
            abs(float toNodes - float fromNodes) / float (max fromNodes toNodes)
        
        let semanticDrift = abs(toState.Significance - fromState.Significance)
        let coherenceDrift = abs(toState.Coherence - fromState.Coherence)
        
        let driftMagnitude = (structuralDrift + semanticDrift + coherenceDrift) / 3.0
        
        // Determine drift direction
        let direction = 
            if toState.Coherence > fromState.Coherence && toState.Significance > fromState.Significance then
                "convergence"
            elif toState.Coherence < fromState.Coherence || toState.Significance < fromState.Significance then
                "divergence"
            else
                "oscillation"
        
        // Find new insights (not present in previous state)
        let newInsights = 
            toState.Insights 
            |> List.filter (fun insight -> 
                not (fromState.Insights |> List.exists (fun prev -> prev.Title = insight.Title)))
        
        // Find resolved contradictions
        let resolvedContradictions = 
            fromState.Contradictions
            |> List.filter (fun contradiction ->
                not (toState.Contradictions |> List.exists (fun curr -> curr.Description = contradiction.Description)))
        
        // Find emergent contradictions
        let emergentContradictions = 
            toState.Contradictions
            |> List.filter (fun contradiction ->
                not (fromState.Contradictions |> List.exists (fun prev -> prev.Description = contradiction.Description)))
        
        {
            Id = generateBeliefId "drift"
            FromState = fromState
            ToState = toState
            DriftMagnitude = driftMagnitude
            DriftDirection = direction
            AffectedDimensions = [0..15] // All sedenion dimensions potentially affected
            SignificanceChange = toState.Significance - fromState.Significance
            CoherenceChange = toState.Coherence - fromState.Coherence
            NewInsights = newInsights
            ResolvedContradictions = resolvedContradictions
            EmergentContradictions = emergentContradictions
        }

    /// Build belief timeline from sequence of states
    let buildBeliefTimeline (states: BeliefDriftState list) : BeliefTimeline =
        if states.Length < 2 then
            {
                Id = generateBeliefId "timeline"
                StartTime = DateTime.UtcNow
                EndTime = DateTime.UtcNow
                States = states
                Drifts = []
                TotalDriftMagnitude = 0.0
                OverallDirection = "stable"
                KeyInflectionPoints = []
            }
        else
            let sortedStates = states |> List.sortBy (fun s -> s.Timestamp.Timestamp)
            
            // Calculate drifts between consecutive states
            let drifts = 
                sortedStates
                |> List.pairwise
                |> List.map (fun (from, to_) -> calculateBeliefDrift from to_)
            
            let totalDriftMagnitude = drifts |> List.sumBy (fun d -> d.DriftMagnitude)
            
            // Determine overall direction
            let convergenceCount = drifts |> List.filter (fun d -> d.DriftDirection = "convergence") |> List.length
            let divergenceCount = drifts |> List.filter (fun d -> d.DriftDirection = "divergence") |> List.length
            
            let overallDirection = 
                if convergenceCount > divergenceCount then "convergence"
                elif divergenceCount > convergenceCount then "divergence"
                else "oscillation"
            
            // Find key inflection points (high drift magnitude)
            let avgDrift = if drifts.Length > 0 then totalDriftMagnitude / float drifts.Length else 0.0
            let inflectionPoints = 
                drifts
                |> List.filter (fun d -> d.DriftMagnitude > avgDrift * 1.5)
                |> List.map (fun d -> d.ToState.Timestamp)
            
            {
                Id = generateBeliefId "timeline"
                StartTime = sortedStates.Head.Timestamp.Timestamp
                EndTime = sortedStates |> List.last |> fun s -> s.Timestamp.Timestamp
                States = sortedStates
                Drifts = drifts
                TotalDriftMagnitude = totalDriftMagnitude
                OverallDirection = overallDirection
                KeyInflectionPoints = inflectionPoints
            }

    /// Generate visualization data for dashboard
    let generateVisualizationData (timeline: BeliefTimeline) : VisualizationData =
        let driftChart = 
            timeline.Drifts
            |> List.map (fun d -> (d.ToState.Timestamp.Timestamp, d.DriftMagnitude))
        
        let coherenceChart = 
            timeline.States
            |> List.map (fun s -> (s.Timestamp.Timestamp, s.Coherence))
        
        let insightChart = 
            timeline.States
            |> List.map (fun s -> (s.Timestamp.Timestamp, s.Insights.Length))
        
        let contradictionChart = 
            timeline.States
            |> List.map (fun s -> (s.Timestamp.Timestamp, s.Contradictions.Length))
        
        // Create dimension heatmap (simplified)
        let dimensionHeatmap = 
            [0..15]
            |> List.map (fun dim -> 
                let values = timeline.States |> List.map (fun _ -> Random().NextDouble())
                (dim, values))
            |> Map.ofList
        
        {
            Timeline = timeline
            DriftChart = driftChart
            CoherenceChart = coherenceChart
            InsightChart = insightChart
            ContradictionChart = contradictionChart
            DimensionHeatmap = dimensionHeatmap
        }

    /// Generate HTML dashboard for belief drift visualization
    let generateHtmlDashboard (visualizationData: VisualizationData) : string =
        let timeline = visualizationData.Timeline
        
        let html = StringBuilder()
        html.AppendLine("<!DOCTYPE html>") |> ignore
        html.AppendLine("<html>") |> ignore
        html.AppendLine("<head>") |> ignore
        html.AppendLine("    <title>TARS Belief Drift Visualization</title>") |> ignore
        html.AppendLine("    <style>") |> ignore
        html.AppendLine("        body { font-family: 'Segoe UI', sans-serif; margin: 20px; background: #0a0a0a; color: #e0e0e0; }") |> ignore
        html.AppendLine("        .header { text-align: center; margin-bottom: 30px; }") |> ignore
        html.AppendLine("        .metric-card { background: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 20px; margin: 10px; display: inline-block; min-width: 200px; }") |> ignore
        html.AppendLine("        .metric-title { font-size: 14px; color: #888; margin-bottom: 5px; }") |> ignore
        html.AppendLine("        .metric-value { font-size: 24px; font-weight: bold; color: #4CAF50; }") |> ignore
        html.AppendLine("        .chart-container { background: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 20px; margin: 20px 0; }") |> ignore
        html.AppendLine("        .timeline-item { background: #2a2a2a; border-left: 4px solid #4CAF50; padding: 15px; margin: 10px 0; }") |> ignore
        html.AppendLine("        .drift-positive { border-left-color: #4CAF50; }") |> ignore
        html.AppendLine("        .drift-negative { border-left-color: #f44336; }") |> ignore
        html.AppendLine("        .drift-neutral { border-left-color: #ff9800; }") |> ignore
        html.AppendLine("    </style>") |> ignore
        html.AppendLine("</head>") |> ignore
        html.AppendLine("<body>") |> ignore
        
        // Header
        html.AppendLine("    <div class='header'>") |> ignore
        html.AppendLine("        <h1>🧠 TARS Belief Drift Visualization</h1>") |> ignore
        let startTimeStr = timeline.StartTime.ToString("yyyy-MM-dd HH:mm")
        let endTimeStr = timeline.EndTime.ToString("yyyy-MM-dd HH:mm")
        html.AppendLine($"        <p>Timeline: {startTimeStr} - {endTimeStr}</p>") |> ignore
        html.AppendLine("    </div>") |> ignore
        
        // Metrics
        html.AppendLine("    <div class='metrics'>") |> ignore
        html.AppendLine("        <div class='metric-card'>") |> ignore
        html.AppendLine("            <div class='metric-title'>Total States</div>") |> ignore
        html.AppendLine($"            <div class='metric-value'>{timeline.States.Length}</div>") |> ignore
        html.AppendLine("        </div>") |> ignore
        html.AppendLine("        <div class='metric-card'>") |> ignore
        html.AppendLine("            <div class='metric-title'>Total Drift</div>") |> ignore
        html.AppendLine($"            <div class='metric-value'>{timeline.TotalDriftMagnitude:F3}</div>") |> ignore
        html.AppendLine("        </div>") |> ignore
        html.AppendLine("        <div class='metric-card'>") |> ignore
        html.AppendLine("            <div class='metric-title'>Overall Direction</div>") |> ignore
        html.AppendLine($"            <div class='metric-value'>{timeline.OverallDirection}</div>") |> ignore
        html.AppendLine("        </div>") |> ignore
        html.AppendLine("        <div class='metric-card'>") |> ignore
        html.AppendLine("            <div class='metric-title'>Inflection Points</div>") |> ignore
        html.AppendLine($"            <div class='metric-value'>{timeline.KeyInflectionPoints.Length}</div>") |> ignore
        html.AppendLine("        </div>") |> ignore
        html.AppendLine("    </div>") |> ignore
        
        // Timeline
        html.AppendLine("    <div class='chart-container'>") |> ignore
        html.AppendLine("        <h3>📈 Belief Evolution Timeline</h3>") |> ignore
        
        for drift in timeline.Drifts do
            let cssClass = 
                if drift.DriftMagnitude > 0.5 then "drift-positive"
                elif drift.DriftMagnitude < 0.2 then "drift-neutral"
                else "drift-negative"
            
            html.AppendLine($"        <div class='timeline-item {cssClass}'>") |> ignore
            let timestampStr = drift.ToState.Timestamp.Timestamp.ToString("HH:mm:ss")
            html.AppendLine($"            <strong>{timestampStr}</strong> - {drift.DriftDirection}") |> ignore
            html.AppendLine($"            <br>Magnitude: {drift.DriftMagnitude:F3}, Significance Δ: {drift.SignificanceChange:F3}") |> ignore
            html.AppendLine($"            <br>New Insights: {drift.NewInsights.Length}, Resolved Contradictions: {drift.ResolvedContradictions.Length}") |> ignore
            html.AppendLine("        </div>") |> ignore
        
        html.AppendLine("    </div>") |> ignore
        html.AppendLine("</body>") |> ignore
        html.AppendLine("</html>") |> ignore
        
        html.ToString()

    /// Test belief drift visualization
    let testBeliefDriftVisualization (logger: ILogger) : bool =
        try
            logger.LogInformation("🧪 Testing belief drift visualization")
            
            // Generate test belief states
            let sessionId = "test-session-" + string (Random().Next(1000, 9999))
            let mutable states = []
            
            for i in 1..5 do
                // Generate test data
                let random = Random(i * 42) // Different seed for each state
                let testVectors = 
                    [1..(10 + i * 5)]
                    |> List.map (fun _ -> Array.init 16 (fun _ -> random.NextDouble() * 2.0 - 1.0))
                
                match partitionChangeVectors testVectors 3 logger with
                | PartitionResult.Success tree ->
                    match performReflection tree logger with
                    | ReflectionResult.Success performance ->
                        // Create mock insights and contradictions
                        let insights = [
                            {
                                Id = generateBeliefId "insight"
                                Category = "test"
                                Title = $"Test Insight {i}"
                                Description = $"Generated insight for state {i}"
                                Significance = random.NextDouble()
                                SupportingEvidence = []
                                Recommendations = []
                                Timestamp = DateTime.UtcNow
                            }
                        ]
                        
                        let contradictions =
                            if i % 3 = 0 then
                                [{
                                    Id = generateBeliefId "contradiction"
                                    Type = "test"
                                    Description = $"Test contradiction {i}"
                                    Confidence = random.NextDouble()
                                    Evidence = []
                                    Timestamp = DateTime.UtcNow
                                }]
                            else []
                        
                        let state = createBeliefState sessionId i tree insights contradictions
                        states <- state :: states
                        
                        logger.LogInformation($"✅ Created belief state {i}: coherence={state.Coherence:F3}, significance={state.Significance:F3}")
                    | ReflectionResult.Error err ->
                        logger.LogWarning($"⚠️ Reflection failed for state {i}: {err}")
                | PartitionResult.Error err ->
                    logger.LogWarning($"⚠️ Partitioning failed for state {i}: {err}")
            
            if states.Length >= 2 then
                // Build timeline
                let timeline = buildBeliefTimeline (List.rev states)
                logger.LogInformation($"✅ Built timeline with {timeline.States.Length} states and {timeline.Drifts.Length} drifts")
                logger.LogInformation($"   Total drift magnitude: {timeline.TotalDriftMagnitude:F3}")
                logger.LogInformation($"   Overall direction: {timeline.OverallDirection}")
                
                // Generate visualization
                let visualizationData = generateVisualizationData timeline
                let htmlDashboard = generateHtmlDashboard visualizationData
                
                // Save dashboard to file
                let outputPath = "./belief_drift_dashboard.html"
                System.IO.File.WriteAllText(outputPath, htmlDashboard)
                logger.LogInformation($"✅ Generated HTML dashboard: {outputPath}")
                
                true
            else
                logger.LogWarning("⚠️ Insufficient belief states generated")
                false
                
        with
        | ex ->
            logger.LogError($"❌ Belief drift visualization test failed: {ex.Message}")
            false
