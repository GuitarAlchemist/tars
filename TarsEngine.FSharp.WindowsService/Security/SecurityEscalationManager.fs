namespace TarsEngine.FSharp.WindowsService.Security

open System
open System.Collections.Concurrent
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.Logging
// open TarsEngine.FSharp.WindowsService.Monitoring

/// <summary>
/// Security incident severity levels
/// </summary>
type SecuritySeverity =
    | Low
    | Medium
    | High
    | Critical

/// <summary>
/// Security incident types
/// </summary>
type SecurityIncidentType =
    | AuthenticationFailure
    | AuthorizationViolation
    | TokenTampering
    | BruteForceAttack
    | SuspiciousActivity
    | DataBreach
    | SystemCompromise
    | ConfigurationViolation
    | UnauthorizedAccess
    | SecurityScanAlert

/// <summary>
/// Security incident details
/// </summary>
type SecurityIncident = {
    Id: string
    Type: SecurityIncidentType
    Severity: SecuritySeverity
    Title: string
    Description: string
    Source: string
    UserId: string option
    IpAddress: string option
    UserAgent: string option
    Timestamp: DateTime
    Evidence: Map<string, string>
    RequiresEscalation: bool
    EscalatedAt: DateTime option
    ResolvedAt: DateTime option
    DevSecOpsAgentNotified: bool
}

/// <summary>
/// DevSecOps agent escalation configuration
/// </summary>
type DevSecOpsEscalationConfig = {
    EnableEscalation: bool
    AutoEscalationThresholds: Map<SecuritySeverity, int>
    EscalationTimeouts: Map<SecuritySeverity, TimeSpan>
    AgentEndpoint: string option
    NotificationChannels: string list
    RequireManualApproval: bool
}

/// <summary>
/// Security escalation manager for TARS
/// Handles security incident detection, classification, and escalation to DevSecOps agent
/// </summary>
type SecurityEscalationManager(logger: ILogger<SecurityEscalationManager>) =
    
    let activeIncidents = ConcurrentDictionary<string, SecurityIncident>()
    let incidentHistory = ConcurrentQueue<SecurityIncident>()
    let mutable isRunning = false
    let mutable cancellationTokenSource = new CancellationTokenSource()
    
    // Default escalation configuration
    let mutable escalationConfig = {
        EnableEscalation = true
        AutoEscalationThresholds = Map [
            (Low, 10)      // 10 low severity incidents
            (Medium, 5)    // 5 medium severity incidents
            (High, 2)      // 2 high severity incidents
            (Critical, 1)  // 1 critical incident
        ]
        EscalationTimeouts = Map [
            (Low, TimeSpan.FromHours(4.0))
            (Medium, TimeSpan.FromHours(1.0))
            (High, TimeSpan.FromMinutes(15.0))
            (Critical, TimeSpan.FromMinutes(5.0))
        ]
        AgentEndpoint = Some "http://localhost:8080/api/devsecops/incident"
        NotificationChannels = ["console"; "eventlog"; "agent"]
        RequireManualApproval = false
    }
    
    /// Start the security escalation manager
    member this.StartAsync() = task {
        if not isRunning then
            isRunning <- true
            cancellationTokenSource <- new CancellationTokenSource()
            
            logger.LogInformation("🔐 Starting Security Escalation Manager...")
            
            // Start monitoring and escalation loops
            let monitoringTask = this.SecurityMonitoringLoopAsync(cancellationTokenSource.Token)
            let escalationTask = this.EscalationLoopAsync(cancellationTokenSource.Token)
            
            logger.LogInformation("✅ Security Escalation Manager started successfully")
    }
    
    /// Stop the security escalation manager
    member this.StopAsync() = task {
        if isRunning then
            isRunning <- false
            cancellationTokenSource.Cancel()
            
            logger.LogInformation("⏹️ Security Escalation Manager stopped")
    }
    
    /// Report a security incident
    member this.ReportSecurityIncident(incidentType: SecurityIncidentType, severity: SecuritySeverity, title: string, description: string, ?source: string, ?userId: string, ?ipAddress: string, ?userAgent: string, ?evidence: Map<string, string>) =
        let incident = {
            Id = Guid.NewGuid().ToString()
            Type = incidentType
            Severity = severity
            Title = title
            Description = description
            Source = source |> Option.defaultValue "Unknown"
            UserId = userId
            IpAddress = ipAddress
            UserAgent = userAgent
            Timestamp = DateTime.UtcNow
            Evidence = evidence |> Option.defaultValue Map.empty
            RequiresEscalation = this.ShouldEscalate(severity)
            EscalatedAt = None
            ResolvedAt = None
            DevSecOpsAgentNotified = false
        }
        
        activeIncidents.[incident.Id] <- incident
        incidentHistory.Enqueue(incident)
        
        // Log the incident
        this.LogSecurityIncident(incident)
        
        // Raise alert
        this.RaiseSecurityAlert(incident)
        
        // Check for immediate escalation
        if incident.RequiresEscalation then
            this.EscalateToDevSecOpsAsync(incident) |> ignore
        
        incident.Id
    
    /// Check if incident should be escalated based on severity
    member private this.ShouldEscalate(severity: SecuritySeverity) =
        match escalationConfig.AutoEscalationThresholds.TryFind(severity) with
        | Some threshold -> threshold <= 1 // Immediate escalation for critical/high
        | None -> false
    
    /// Log security incident
    member private this.LogSecurityIncident(incident: SecurityIncident) =
        let severityStr =
            match incident.Severity with
            | Low -> "LOW"
            | Medium -> "MEDIUM"
            | High -> "HIGH"
            | Critical -> "CRITICAL"

        let logLevel =
            match incident.Severity with
            | Low -> LogLevel.Information
            | Medium -> LogLevel.Warning
            | High -> LogLevel.Error
            | Critical -> LogLevel.Critical
        
        logger.Log(logLevel, "🚨 SECURITY INCIDENT [{Severity}] {Type}: {Title} - {Description} | Source: {Source} | User: {UserId} | IP: {IpAddress}", 
            severityStr, incident.Type, incident.Title, incident.Description, incident.Source, 
            incident.UserId |> Option.defaultValue "N/A", incident.IpAddress |> Option.defaultValue "N/A")
    
    /// Raise security alert
    member private this.RaiseSecurityAlert(incident: SecurityIncident) =
        let alertSeverity =
            match incident.Severity with
            | Low -> "Info"
            | Medium -> "Warning"
            | High -> "Error"
            | Critical -> "Critical"
        
        let alertData = Map [
            ("IncidentId", incident.Id)
            ("IncidentType", incident.Type.ToString())
            ("Source", incident.Source)
            ("UserId", incident.UserId |> Option.defaultValue "N/A")
            ("IpAddress", incident.IpAddress |> Option.defaultValue "N/A")
            ("Timestamp", incident.Timestamp.ToString("yyyy-MM-dd HH:mm:ss UTC"))
        ]
        
        logger.LogInformation("🚨 Security Alert [{Severity}]: {Title} - {Description}", alertSeverity, incident.Title, incident.Description)
    
    /// Escalate incident to DevSecOps agent
    member private this.EscalateToDevSecOpsAsync(incident: SecurityIncident) = task {
        try
            logger.LogWarning("🚨 Escalating security incident {IncidentId} to DevSecOps agent: {Title}", incident.Id, incident.Title)
            
            // Update incident status
            let escalatedIncident = { incident with EscalatedAt = Some DateTime.UtcNow; DevSecOpsAgentNotified = true }
            activeIncidents.[incident.Id] <- escalatedIncident
            
            // Send to DevSecOps agent via multiple channels
            do! this.NotifyDevSecOpsAgentAsync(escalatedIncident)
            
            // Raise critical alert for escalation
            let escalationData = Map [
                ("IncidentId", incident.Id)
                ("EscalatedAt", DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss UTC"))
                ("Severity", incident.Severity.ToString())
                ("Type", incident.Type.ToString())
            ]
            
            logger.LogCritical("🚨 Security Escalation: {Title} - Incident {IncidentId} escalated to DevSecOps", incident.Title, incident.Id)
            
            logger.LogCritical("🔴 SECURITY ESCALATION: Incident {IncidentId} escalated to DevSecOps agent", incident.Id)
            
        with
        | ex ->
            logger.LogError(ex, "❌ Failed to escalate security incident {IncidentId} to DevSecOps agent", incident.Id)
    }
    
    /// Notify DevSecOps agent through multiple channels
    member private this.NotifyDevSecOpsAgentAsync(incident: SecurityIncident) = task {
        // 1. Console notification (immediate)
        if escalationConfig.NotificationChannels |> List.contains "console" then
            Console.WriteLine($"🚨 SECURITY ESCALATION: {incident.Title}")
            Console.WriteLine($"   Incident ID: {incident.Id}")
            Console.WriteLine($"   Severity: {incident.Severity}")
            Console.WriteLine($"   Type: {incident.Type}")
            Console.WriteLine($"   Description: {incident.Description}")
            let timeStr = incident.Timestamp.ToString("yyyy-MM-dd HH:mm:ss")
            Console.WriteLine($"   Time: {timeStr} UTC")
            Console.WriteLine($"   DevSecOps Agent: Please investigate immediately!")
        
        // 2. Event log notification
        if escalationConfig.NotificationChannels |> List.contains "eventlog" then
            // TODO: Write to Windows Event Log
            ()
        
        // 3. Agent endpoint notification
        if escalationConfig.NotificationChannels |> List.contains "agent" then
            match escalationConfig.AgentEndpoint with
            | Some endpoint ->
                try
                    // TODO: Send HTTP request to DevSecOps agent endpoint
                    logger.LogInformation("📡 Notifying DevSecOps agent at endpoint: {Endpoint}", endpoint)
                with
                | ex ->
                    logger.LogError(ex, "❌ Failed to notify DevSecOps agent at endpoint: {Endpoint}", endpoint)
            | None ->
                logger.LogWarning("⚠️ DevSecOps agent endpoint not configured")
    }
    
    /// Security monitoring loop
    member private this.SecurityMonitoringLoopAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogDebug("Starting security monitoring loop")
            
            while not cancellationToken.IsCancellationRequested && isRunning do
                try
                    // Monitor for security patterns and anomalies
                    do! this.AnalyzeSecurityPatternsAsync()
                    
                    // Check for escalation timeouts
                    do! this.CheckEscalationTimeoutsAsync()
                    
                    // Wait for next monitoring cycle
                    do! Task.Delay(TimeSpan.FromMinutes(1.0), cancellationToken)
                    
                with
                | :? OperationCanceledException ->
                    ()
                | ex ->
                    logger.LogWarning(ex, "Error in security monitoring loop")
                    do! Task.Delay(TimeSpan.FromMinutes(1.0), cancellationToken)
                    
        with
        | :? OperationCanceledException ->
            logger.LogDebug("Security monitoring loop cancelled")
        | ex ->
            logger.LogError(ex, "Security monitoring loop failed")
    }
    
    /// Escalation loop
    member private this.EscalationLoopAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogDebug("Starting security escalation loop")
            
            while not cancellationToken.IsCancellationRequested && isRunning do
                try
                    // Process pending escalations
                    do! this.ProcessPendingEscalationsAsync()
                    
                    // Wait for next escalation cycle
                    do! Task.Delay(TimeSpan.FromMinutes(5.0), cancellationToken)
                    
                with
                | :? OperationCanceledException ->
                    ()
                | ex ->
                    logger.LogWarning(ex, "Error in security escalation loop")
                    do! Task.Delay(TimeSpan.FromMinutes(5.0), cancellationToken)
                    
        with
        | :? OperationCanceledException ->
            logger.LogDebug("Security escalation loop cancelled")
        | ex ->
            logger.LogError(ex, "Security escalation loop failed")
    }
    
    /// Analyze security patterns for anomalies
    member private this.AnalyzeSecurityPatternsAsync() = task {
        // Analyze recent incidents for patterns
        let recentIncidents = 
            activeIncidents.Values
            |> Seq.filter (fun i -> DateTime.UtcNow - i.Timestamp < TimeSpan.FromHours(1.0))
            |> List.ofSeq
        
        // Check for brute force patterns
        let authFailures = 
            recentIncidents
            |> List.filter (fun i -> i.Type = AuthenticationFailure)
            |> List.groupBy (fun i -> i.IpAddress)
            |> List.filter (fun (_, incidents) -> incidents.Length >= 5)
        
        for (ipAddress, incidents) in authFailures do
            let ipStr = ipAddress |> Option.defaultValue "Unknown"
            let incidentId = this.ReportSecurityIncident(
                BruteForceAttack,
                High,
                "Potential Brute Force Attack Detected",
                $"Multiple authentication failures from IP: {ipStr}",
                ?source = Some "SecurityPatternAnalysis",
                ?ipAddress = ipAddress,
                ?evidence = Some (Map ["FailureCount", incidents.Length.ToString(); "TimeWindow", "1 hour"])
            )
            logger.LogWarning("🔍 Brute force attack pattern detected: {IncidentId}", incidentId)
    }
    
    /// Check for escalation timeouts
    member private this.CheckEscalationTimeoutsAsync() = task {
        let now = DateTime.UtcNow
        
        let timedOutIncidents = 
            activeIncidents.Values
            |> Seq.filter (fun i -> 
                i.RequiresEscalation && 
                i.EscalatedAt.IsNone &&
                match escalationConfig.EscalationTimeouts.TryFind(i.Severity) with
                | Some timeout -> now - i.Timestamp > timeout
                | None -> false)
            |> List.ofSeq
        
        for incident in timedOutIncidents do
            logger.LogWarning("⏰ Security incident {IncidentId} escalation timeout reached", incident.Id)
            do! this.EscalateToDevSecOpsAsync(incident)
    }
    
    /// Process pending escalations
    member private this.ProcessPendingEscalationsAsync() = task {
        let pendingEscalations = 
            activeIncidents.Values
            |> Seq.filter (fun i -> i.RequiresEscalation && i.EscalatedAt.IsNone)
            |> List.ofSeq
        
        for incident in pendingEscalations do
            do! this.EscalateToDevSecOpsAsync(incident)
    }
    
    /// Get security incident statistics
    member this.GetSecurityStatistics() =
        let totalIncidents = incidentHistory.Count
        let activeCount = activeIncidents.Count
        let escalatedCount = activeIncidents.Values |> Seq.filter (fun i -> i.EscalatedAt.IsSome) |> Seq.length
        
        let severityBreakdown = 
            activeIncidents.Values
            |> Seq.groupBy (fun i -> i.Severity)
            |> Seq.map (fun (severity, incidents) -> (severity.ToString(), Seq.length incidents))
            |> Map.ofSeq
        
        {|
            TotalIncidents = totalIncidents
            ActiveIncidents = activeCount
            EscalatedIncidents = escalatedCount
            SeverityBreakdown = severityBreakdown
            LastIncident = if activeIncidents.IsEmpty then None else Some (activeIncidents.Values |> Seq.maxBy (fun i -> i.Timestamp))
        |}
    
    /// Resolve security incident
    member this.ResolveIncident(incidentId: string, resolution: string) =
        match activeIncidents.TryGetValue(incidentId) with
        | true, incident ->
            let resolvedIncident = { incident with ResolvedAt = Some DateTime.UtcNow }
            activeIncidents.TryRemove(incidentId) |> ignore
            logger.LogInformation("✅ Security incident resolved: {IncidentId} - {Resolution}", incidentId, resolution)
            true
        | _ ->
            logger.LogWarning("⚠️ Attempted to resolve unknown security incident: {IncidentId}", incidentId)
            false
