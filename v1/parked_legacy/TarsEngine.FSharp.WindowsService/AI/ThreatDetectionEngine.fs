namespace TarsEngine.FSharp.WindowsService.AI

open System
open System.Collections.Concurrent
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.WindowsService.Security

/// <summary>
/// AI-powered threat detection features
/// </summary>
type ThreatFeature = {
    Name: string
    Value: float
    Weight: float
    Description: string
}

/// <summary>
/// Threat detection model prediction
/// </summary>
type ThreatPrediction = {
    ThreatLevel: float // 0.0 to 1.0
    ThreatType: SecurityIncidentType
    Confidence: float
    Features: ThreatFeature list
    Reasoning: string list
    RecommendedActions: string list
}

/// <summary>
/// Behavioral pattern for threat analysis
/// </summary>
type BehavioralPattern = {
    PatternId: string
    PatternType: string
    Frequency: int
    LastSeen: DateTime
    Severity: float
    Indicators: string list
}

/// <summary>
/// AI-powered threat detection engine
/// Uses machine learning and behavioral analysis for advanced threat detection
/// </summary>
type ThreatDetectionEngine(logger: ILogger<ThreatDetectionEngine>) =
    
    let behavioralPatterns = ConcurrentDictionary<string, BehavioralPattern>()
    let threatHistory = ConcurrentQueue<(DateTime * ThreatPrediction)>()
    let mutable isRunning = false
    let mutable cancellationTokenSource = new CancellationTokenSource()
    
    // ML model weights (simplified neural network approach)
    let mutable modelWeights = Map [
        ("failed_attempts_rate", 0.8)
        ("request_frequency", 0.6)
        ("time_pattern_anomaly", 0.7)
        ("ip_reputation", 0.9)
        ("user_agent_suspicion", 0.5)
        ("geographic_anomaly", 0.4)
        ("token_manipulation", 0.95)
        ("endpoint_targeting", 0.6)
        ("payload_analysis", 0.7)
        ("behavioral_deviation", 0.8)
    ]
    
    /// Start the AI threat detection engine
    member this.StartAsync() = task {
        if not isRunning then
            isRunning <- true
            cancellationTokenSource <- new CancellationTokenSource()
            
            logger.LogInformation("🧠 Starting AI Threat Detection Engine...")
            logger.LogInformation("   🤖 Machine Learning: ACTIVE")
            logger.LogInformation("   📊 Behavioral Analysis: ENABLED")
            logger.LogInformation("   🔍 Pattern Recognition: ONLINE")
            
            // Start AI analysis loops
            let patternAnalysisTask = this.PatternAnalysisLoopAsync(cancellationTokenSource.Token)
            let modelTrainingTask = this.ModelTrainingLoopAsync(cancellationTokenSource.Token)
            
            logger.LogInformation("✅ AI Threat Detection Engine started successfully")
    }
    
    /// Stop the AI threat detection engine
    member this.StopAsync() = task {
        if isRunning then
            isRunning <- false
            cancellationTokenSource.Cancel()
            
            logger.LogInformation("⏹️ AI Threat Detection Engine stopped")
    }
    
    /// Analyze security incident using AI
    member this.AnalyzeThreat(incident: SecurityIncident, ?requestData: Map<string, obj>) = 
        try
            let features = this.ExtractThreatFeatures(incident, requestData)
            let prediction = this.PredictThreat(features)
            
            // Update behavioral patterns
            this.UpdateBehavioralPatterns(incident, prediction)
            
            // Store prediction history
            threatHistory.Enqueue((DateTime.UtcNow, prediction))
            
            logger.LogInformation("🧠 AI Threat Analysis Complete:")
            logger.LogInformation("   🎯 Threat Level: {ThreatLevel:F2}", prediction.ThreatLevel)
            logger.LogInformation("   🔍 Threat Type: {ThreatType}", prediction.ThreatType)
            logger.LogInformation("   📊 Confidence: {Confidence:F2}", prediction.Confidence)
            logger.LogInformation("   🧮 Features: {FeatureCount}", features.Length)
            
            Some prediction
            
        with
        | ex ->
            logger.LogError(ex, "❌ AI threat analysis failed for incident: {IncidentId}", incident.Id)
            None
    
    /// Extract threat features from security incident
    member private this.ExtractThreatFeatures(incident: SecurityIncident, requestData: Map<string, obj> option) =
        let features = ResizeArray<ThreatFeature>()
        
        // Time-based features
        let currentHour = DateTime.UtcNow.Hour
        let isOffHours = currentHour < 6 || currentHour > 22
        features.Add({
            Name = "time_pattern_anomaly"
            Value = if isOffHours then 0.8 else 0.2
            Weight = 0.7
            Description = "Activity during off-hours"
        })
        
        // Frequency-based features
        let recentIncidents = 
            threatHistory
            |> Seq.filter (fun (time, _) -> DateTime.UtcNow - time < TimeSpan.FromHours(1.0))
            |> Seq.length
        
        features.Add({
            Name = "request_frequency"
            Value = Math.Min(1.0, float recentIncidents / 10.0)
            Weight = 0.6
            Description = "Request frequency in last hour"
        })
        
        // IP reputation analysis
        let ipSuspicion = 
            match incident.IpAddress with
            | Some ip when ip.StartsWith("10.") || ip.StartsWith("192.168.") -> 0.1 // Internal IP
            | Some ip when ip.Contains("tor") || ip.Contains("proxy") -> 0.9 // Suspicious
            | Some _ -> 0.3 // External IP
            | None -> 0.5 // Unknown
        
        features.Add({
            Name = "ip_reputation"
            Value = ipSuspicion
            Weight = 0.9
            Description = "IP address reputation analysis"
        })
        
        // User agent analysis
        let userAgentSuspicion = 
            match incident.UserAgent with
            | Some ua when ua.Contains("bot") || ua.Contains("crawler") || ua.Contains("scanner") -> 0.9
            | Some ua when ua.Contains("curl") || ua.Contains("wget") || ua.Contains("python") -> 0.7
            | Some ua when ua.Length < 20 -> 0.6 // Very short user agent
            | Some _ -> 0.2 // Normal user agent
            | None -> 0.4 // Missing user agent
        
        features.Add({
            Name = "user_agent_suspicion"
            Value = userAgentSuspicion
            Weight = 0.5
            Description = "User agent analysis"
        })
        
        // Incident type specific features
        match incident.Type with
        | AuthenticationFailure ->
            features.Add({
                Name = "failed_attempts_rate"
                Value = 0.8
                Weight = 0.8
                Description = "Authentication failure pattern"
            })
        | TokenTampering ->
            features.Add({
                Name = "token_manipulation"
                Value = 0.95
                Weight = 0.95
                Description = "Token tampering detected"
            })
        | BruteForceAttack ->
            features.Add({
                Name = "failed_attempts_rate"
                Value = 1.0
                Weight = 0.9
                Description = "Brute force attack pattern"
            })
        | _ ->
            features.Add({
                Name = "general_threat"
                Value = 0.5
                Weight = 0.5
                Description = "General security incident"
            })
        
        // Behavioral deviation analysis
        let behavioralDeviation = this.CalculateBehavioralDeviation(incident)
        features.Add({
            Name = "behavioral_deviation"
            Value = behavioralDeviation
            Weight = 0.8
            Description = "Deviation from normal behavior patterns"
        })
        
        features |> List.ofSeq
    
    /// Predict threat using AI model
    member private this.PredictThreat(features: ThreatFeature list) =
        // Simple neural network-like calculation
        let weightedSum = 
            features
            |> List.sumBy (fun f -> 
                let modelWeight = modelWeights.TryFind(f.Name) |> Option.defaultValue 0.5
                f.Value * f.Weight * modelWeight)
        
        let normalizedScore = Math.Min(1.0, weightedSum / float features.Length)
        
        // Determine threat type based on feature analysis
        let threatType = 
            let tokenFeature = features |> List.tryFind (fun f -> f.Name = "token_manipulation")
            let bruteForceFeature = features |> List.tryFind (fun f -> f.Name = "failed_attempts_rate")
            let behavioralFeature = features |> List.tryFind (fun f -> f.Name = "behavioral_deviation")
            
            match tokenFeature, bruteForceFeature, behavioralFeature with
            | Some tf, _, _ when tf.Value > 0.8 -> TokenTampering
            | _, Some bf, _ when bf.Value > 0.8 -> BruteForceAttack
            | _, _, Some beh when beh.Value > 0.7 -> SuspiciousActivity
            | _ -> UnauthorizedAccess
        
        // Generate reasoning
        let reasoning = 
            features
            |> List.filter (fun f -> f.Value > 0.5)
            |> List.map (fun f -> $"High {f.Name}: {f.Value:F2} - {f.Description}")
        
        // Generate recommendations based on AI analysis
        let recommendations = this.GenerateAIRecommendations(threatType, normalizedScore, features)
        
        {
            ThreatLevel = normalizedScore
            ThreatType = threatType
            Confidence = Math.Min(0.95, normalizedScore + 0.1)
            Features = features
            Reasoning = reasoning
            RecommendedActions = recommendations
        }
    
    /// Generate AI-powered recommendations
    member private this.GenerateAIRecommendations(threatType: SecurityIncidentType, threatLevel: float, features: ThreatFeature list) =
        let recommendations = ResizeArray<string>()
        
        // Base recommendations by threat type
        match threatType with
        | TokenTampering ->
            recommendations.Add("Immediate JWT key rotation required")
            recommendations.Add("Implement token blacklisting mechanism")
            recommendations.Add("Enhanced token validation logging")
        | BruteForceAttack ->
            recommendations.Add("Implement progressive delay on failed attempts")
            recommendations.Add("Deploy CAPTCHA after 3 failed attempts")
            recommendations.Add("Consider IP-based rate limiting")
        | SuspiciousActivity ->
            recommendations.Add("Increase monitoring for this IP/user")
            recommendations.Add("Implement behavioral analysis alerts")
            recommendations.Add("Consider temporary access restrictions")
        | _ ->
            recommendations.Add("Enhanced monitoring recommended")
            recommendations.Add("Review access patterns")
        
        // AI-enhanced recommendations based on threat level
        if threatLevel > 0.8 then
            recommendations.Add("CRITICAL: Immediate human intervention required")
            recommendations.Add("Consider temporary service isolation")
            recommendations.Add("Activate incident response protocol")
        elif threatLevel > 0.6 then
            recommendations.Add("HIGH: Automated mitigation actions recommended")
            recommendations.Add("Increase security monitoring sensitivity")
        elif threatLevel > 0.4 then
            recommendations.Add("MEDIUM: Enhanced logging and monitoring")
            recommendations.Add("Consider preventive measures")
        
        // Feature-specific recommendations
        let ipFeature = features |> List.tryFind (fun f -> f.Name = "ip_reputation")
        match ipFeature with
        | Some ip when ip.Value > 0.7 ->
            recommendations.Add("Consider IP-based blocking or restrictions")
        | _ -> ()
        
        let timeFeature = features |> List.tryFind (fun f -> f.Name = "time_pattern_anomaly")
        match timeFeature with
        | Some time when time.Value > 0.6 ->
            recommendations.Add("Off-hours activity detected - consider time-based restrictions")
        | _ -> ()
        
        recommendations |> List.ofSeq
    
    /// Calculate behavioral deviation from normal patterns
    member private this.CalculateBehavioralDeviation(incident: SecurityIncident) =
        // Analyze deviation from established patterns
        let patternKey = $"{incident.Type}_{incident.IpAddress |> Option.defaultValue "unknown"}"
        
        match behavioralPatterns.TryGetValue(patternKey) with
        | true, pattern ->
            let timeSinceLastSeen = DateTime.UtcNow - pattern.LastSeen
            let frequencyDeviation = 
                if timeSinceLastSeen < TimeSpan.FromMinutes(5.0) then 0.9 // Very frequent
                elif timeSinceLastSeen < TimeSpan.FromHours(1.0) then 0.6 // Frequent
                else 0.3 // Normal
            
            Math.Min(1.0, frequencyDeviation + pattern.Severity * 0.3)
        | _ ->
            0.7 // New pattern - moderately suspicious
    
    /// Update behavioral patterns based on incident
    member private this.UpdateBehavioralPatterns(incident: SecurityIncident, prediction: ThreatPrediction) =
        let patternKey = $"{incident.Type}_{incident.IpAddress |> Option.defaultValue "unknown"}"
        
        let updatedPattern = 
            match behavioralPatterns.TryGetValue(patternKey) with
            | true, existing ->
                { existing with
                    Frequency = existing.Frequency + 1
                    LastSeen = DateTime.UtcNow
                    Severity = (existing.Severity + prediction.ThreatLevel) / 2.0
                    Indicators = (incident.Description :: existing.Indicators) |> List.take 10
                }
            | _ ->
                {
                    PatternId = patternKey
                    PatternType = incident.Type.ToString()
                    Frequency = 1
                    LastSeen = DateTime.UtcNow
                    Severity = prediction.ThreatLevel
                    Indicators = [incident.Description]
                }
        
        behavioralPatterns.[patternKey] <- updatedPattern
    
    /// Pattern analysis loop for continuous learning
    member private this.PatternAnalysisLoopAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogDebug("Starting AI pattern analysis loop")
            
            while not cancellationToken.IsCancellationRequested && isRunning do
                try
                    // Analyze behavioral patterns for anomalies
                    do! this.AnalyzeBehavioralPatternsAsync()
                    
                    // Clean up old patterns
                    do! this.CleanupOldPatternsAsync()
                    
                    // Wait for next analysis cycle
                    do! Task.Delay(TimeSpan.FromMinutes(5.0), cancellationToken)
                    
                with
                | :? OperationCanceledException ->
                    ()
                | ex ->
                    logger.LogWarning(ex, "Error in AI pattern analysis loop")
                    do! Task.Delay(TimeSpan.FromMinutes(5.0), cancellationToken)
                    
        with
        | :? OperationCanceledException ->
            logger.LogDebug("AI pattern analysis loop cancelled")
        | ex ->
            logger.LogError(ex, "AI pattern analysis loop failed")
    }
    
    /// Model training loop for continuous improvement
    member private this.ModelTrainingLoopAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogDebug("Starting AI model training loop")
            
            while not cancellationToken.IsCancellationRequested && isRunning do
                try
                    // Retrain model based on recent data
                    do! this.RetrainModelAsync()
                    
                    // Wait for next training cycle
                    do! Task.Delay(TimeSpan.FromHours(6.0), cancellationToken)
                    
                with
                | :? OperationCanceledException ->
                    ()
                | ex ->
                    logger.LogWarning(ex, "Error in AI model training loop")
                    do! Task.Delay(TimeSpan.FromHours(1.0), cancellationToken)
                    
        with
        | :? OperationCanceledException ->
            logger.LogDebug("AI model training loop cancelled")
        | ex ->
            logger.LogError(ex, "AI model training loop failed")
    }
    
    /// Analyze behavioral patterns for anomalies
    member private this.AnalyzeBehavioralPatternsAsync() = task {
        let anomalousPatterns = 
            behavioralPatterns.Values
            |> Seq.filter (fun p -> p.Severity > 0.7 && p.Frequency > 5)
            |> List.ofSeq
        
        for pattern in anomalousPatterns do
            logger.LogWarning("🔍 Anomalous behavioral pattern detected: {PatternId} - Severity: {Severity:F2}, Frequency: {Frequency}", 
                pattern.PatternId, pattern.Severity, pattern.Frequency)
    }
    
    /// Clean up old behavioral patterns
    member private this.CleanupOldPatternsAsync() = task {
        let cutoffTime = DateTime.UtcNow.AddDays(-7.0)
        
        let oldPatterns = 
            behavioralPatterns
            |> Seq.filter (fun kvp -> kvp.Value.LastSeen < cutoffTime)
            |> Seq.map (fun kvp -> kvp.Key)
            |> List.ofSeq
        
        for patternKey in oldPatterns do
            behavioralPatterns.TryRemove(patternKey) |> ignore
            logger.LogDebug("🗑️ Cleaned up old behavioral pattern: {PatternKey}", patternKey)
    }
    
    /// Retrain AI model based on recent threat data
    member private this.RetrainModelAsync() = task {
        let recentThreats = 
            threatHistory
            |> Seq.filter (fun (time, _) -> DateTime.UtcNow - time < TimeSpan.FromDays(7.0))
            |> Seq.map snd
            |> List.ofSeq
        
        if recentThreats.Length > 10 then
            // Simple weight adjustment based on prediction accuracy
            let avgThreatLevel = recentThreats |> List.averageBy (fun t -> t.ThreatLevel)
            let avgConfidence = recentThreats |> List.averageBy (fun t -> t.Confidence)
            
            // Adjust model weights based on performance
            if avgConfidence > 0.8 then
                logger.LogInformation("🧠 AI model performing well - maintaining current weights")
            else
                logger.LogInformation("🔄 Adjusting AI model weights based on recent performance")
                // Simple weight adjustment (in production, use proper ML training)
                modelWeights <- 
                    modelWeights
                    |> Map.map (fun _ weight -> weight * 0.95 + avgThreatLevel * 0.05)
            
            logger.LogInformation("🎯 AI model retrained with {SampleCount} recent threats", recentThreats.Length)
    }
    
    /// Get AI threat detection statistics
    member this.GetAIStatistics() =
        let recentThreats = 
            threatHistory
            |> Seq.filter (fun (time, _) -> DateTime.UtcNow - time < TimeSpan.FromHours(24.0))
            |> Seq.map snd
            |> List.ofSeq
        
        {|
            IsRunning = isRunning
            TotalPatterns = behavioralPatterns.Count
            RecentThreats = recentThreats.Length
            AverageThreatLevel = if recentThreats.IsEmpty then 0.0 else recentThreats |> List.averageBy (fun t -> t.ThreatLevel)
            AverageConfidence = if recentThreats.IsEmpty then 0.0 else recentThreats |> List.averageBy (fun t -> t.Confidence)
            ModelWeights = modelWeights
            LastAnalysis = DateTime.UtcNow
        |}
