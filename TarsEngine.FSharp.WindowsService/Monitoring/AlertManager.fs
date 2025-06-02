namespace TarsEngine.FSharp.WindowsService.Monitoring

open System
open System.Collections.Concurrent
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.WindowsService.Core.ServiceConfiguration

/// <summary>
/// Alert severity levels
/// </summary>
type AlertSeverity =
    | Info
    | Warning
    | Error
    | Critical

/// <summary>
/// Alert status enumeration
/// </summary>
type AlertStatus =
    | Active
    | Acknowledged
    | Resolved
    | Suppressed

/// <summary>
/// Alert escalation level
/// </summary>
type EscalationLevel =
    | Level1 // Initial alert
    | Level2 // First escalation
    | Level3 // Second escalation
    | Level4 // Final escalation

/// <summary>
/// Alert definition
/// </summary>
type Alert = {
    Id: string
    Source: string
    Severity: AlertSeverity
    Status: AlertStatus
    Title: string
    Message: string
    Timestamp: DateTime
    LastUpdated: DateTime
    EscalationLevel: EscalationLevel
    EscalationCount: int
    AcknowledgedBy: string option
    ResolvedBy: string option
    Metadata: Map<string, obj>
    Tags: string list
}

/// <summary>
/// Alert rule configuration
/// </summary>
type AlertRule = {
    Id: string
    Name: string
    Description: string
    Source: string
    Condition: string
    Severity: AlertSeverity
    Threshold: float
    EvaluationInterval: TimeSpan
    EscalationInterval: TimeSpan
    MaxEscalations: int
    AutoResolve: bool
    AutoResolveTimeout: TimeSpan
    Enabled: bool
    Tags: string list
}

/// <summary>
/// Alert notification configuration
/// </summary>
type NotificationConfig = {
    Type: NotificationType
    Recipients: string list
    Template: string
    EscalationLevels: EscalationLevel list
    Enabled: bool
}

/// <summary>
/// Notification types
/// </summary>
and NotificationType =
    | Email
    | SMS
    | Slack
    | Teams
    | Webhook
    | EventLog

/// <summary>
/// Alert statistics
/// </summary>
type AlertStatistics = {
    TotalAlerts: int64
    ActiveAlerts: int
    ResolvedAlerts: int64
    AcknowledgedAlerts: int
    SuppressedAlerts: int
    AlertsBySource: Map<string, int>
    AlertsBySeverity: Map<AlertSeverity, int>
    AverageResolutionTimeMinutes: float
    EscalationRate: float
    AutoResolveRate: float
}

/// <summary>
/// Intelligent alert management system with escalation and notification
/// </summary>
type AlertManager(logger: ILogger<AlertManager>) =
    
    let activeAlerts = ConcurrentDictionary<string, Alert>()
    let alertHistory = ConcurrentQueue<Alert>()
    let alertRules = ConcurrentDictionary<string, AlertRule>()
    let notificationConfigs = ConcurrentDictionary<string, NotificationConfig>()
    let alertStatistics = ConcurrentDictionary<string, int64>()
    
    let mutable isRunning = false
    let mutable cancellationTokenSource: CancellationTokenSource option = None
    let mutable processingTask: Task option = None
    let mutable escalationTask: Task option = None
    
    let maxAlertHistory = 50000
    
    /// Start the alert manager
    member this.StartAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogInformation("Starting alert manager...")
            
            cancellationTokenSource <- Some (CancellationTokenSource.CreateLinkedTokenSource(cancellationToken))
            isRunning <- true
            
            // Initialize default alert rules
            this.InitializeDefaultAlertRules()
            
            // Initialize default notification configs
            this.InitializeDefaultNotificationConfigs()
            
            // Start alert processing loop
            let processingLoop = this.AlertProcessingLoopAsync(cancellationTokenSource.Value.Token)
            processingTask <- Some processingLoop
            
            // Start escalation loop
            let escalationLoop = this.EscalationLoopAsync(cancellationTokenSource.Value.Token)
            escalationTask <- Some escalationLoop
            
            logger.LogInformation("Alert manager started successfully")
            
        with
        | ex ->
            logger.LogError(ex, "Failed to start alert manager")
            isRunning <- false
            raise
    }
    
    /// Stop the alert manager
    member this.StopAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogInformation("Stopping alert manager...")
            
            isRunning <- false
            
            // Cancel all operations
            match cancellationTokenSource with
            | Some cts -> cts.Cancel()
            | None -> ()
            
            // Wait for tasks to complete
            let tasks = [
                match processingTask with Some t -> [t] | None -> []
                match escalationTask with Some t -> [t] | None -> []
            ] |> List.concat |> Array.ofList
            
            if tasks.Length > 0 then
                try
                    do! Task.WhenAll(tasks).WaitAsync(TimeSpan.FromSeconds(10.0), cancellationToken)
                with
                | :? TimeoutException ->
                    logger.LogWarning("Alert manager tasks did not complete within timeout")
                | ex ->
                    logger.LogWarning(ex, "Error waiting for alert manager tasks to complete")
            
            // Cleanup
            match cancellationTokenSource with
            | Some cts -> 
                cts.Dispose()
                cancellationTokenSource <- None
            | None -> ()
            
            processingTask <- None
            escalationTask <- None
            
            logger.LogInformation("Alert manager stopped successfully")
            
        with
        | ex ->
            logger.LogError(ex, "Error stopping alert manager")
    }
    
    /// Raise an alert
    member this.RaiseAlert(source: string, severity: string, message: string, metadata: Map<string, obj>) = task {
        try
            let alertSeverity = 
                match severity.ToLower() with
                | "info" -> Info
                | "warning" -> Warning
                | "error" -> Error
                | "critical" -> Critical
                | _ -> Warning
            
            let alert = {
                Id = Guid.NewGuid().ToString()
                Source = source
                Severity = alertSeverity
                Status = Active
                Title = $"{source} Alert"
                Message = message
                Timestamp = DateTime.UtcNow
                LastUpdated = DateTime.UtcNow
                EscalationLevel = Level1
                EscalationCount = 0
                AcknowledgedBy = None
                ResolvedBy = None
                Metadata = metadata
                Tags = []
            }
            
            // Check for duplicate alerts
            let existingAlert = 
                activeAlerts.Values
                |> Seq.tryFind (fun a -> a.Source = source && a.Message = message && a.Status = Active)
            
            match existingAlert with
            | Some existing ->
                // Update existing alert
                let updatedAlert = { existing with LastUpdated = DateTime.UtcNow; EscalationCount = existing.EscalationCount + 1 }
                activeAlerts.[existing.Id] <- updatedAlert
                logger.LogDebug($"Updated existing alert: {existing.Id}")
                Ok existing.Id
            
            | None ->
                // Create new alert
                activeAlerts.[alert.Id] <- alert
                
                // Update statistics
                this.UpdateStatistics("TotalAlerts", 1L)
                this.UpdateStatistics($"Alerts_{alertSeverity}", 1L)
                this.UpdateStatistics($"AlertsFrom_{source}", 1L)
                
                // Send notification
                do! this.SendNotificationAsync(alert)

                logger.LogInformation($"Alert raised: {alert.Id} - {alert.Severity} - {alert.Message}")
                return Ok alert.Id

        with
        | ex ->
            logger.LogError(ex, $"Failed to raise alert from {source}")
            return Error ex.Message
    }
    
    /// Acknowledge an alert
    member this.AcknowledgeAlert(alertId: string, acknowledgedBy: string) =
        try
            match activeAlerts.TryGetValue(alertId) with
            | true, alert ->
                let updatedAlert = { 
                    alert with 
                        Status = Acknowledged
                        AcknowledgedBy = Some acknowledgedBy
                        LastUpdated = DateTime.UtcNow 
                }
                activeAlerts.[alertId] <- updatedAlert
                
                logger.LogInformation($"Alert acknowledged: {alertId} by {acknowledgedBy}")
                Ok ()
            
            | false, _ ->
                let error = $"Alert not found: {alertId}"
                logger.LogWarning(error)
                Error error
                
        with
        | ex ->
            logger.LogError(ex, $"Failed to acknowledge alert: {alertId}")
            Error ex.Message
    
    /// Resolve an alert
    member this.ResolveAlert(alertId: string, resolvedBy: string) =
        try
            match activeAlerts.TryGetValue(alertId) with
            | true, alert ->
                let updatedAlert = { 
                    alert with 
                        Status = Resolved
                        ResolvedBy = Some resolvedBy
                        LastUpdated = DateTime.UtcNow 
                }
                
                // Move to history
                alertHistory.Enqueue(updatedAlert)
                activeAlerts.TryRemove(alertId) |> ignore
                
                // Keep history size manageable
                while alertHistory.Count > maxAlertHistory do
                    alertHistory.TryDequeue() |> ignore
                
                // Update statistics
                this.UpdateStatistics("ResolvedAlerts", 1L)
                
                // Calculate resolution time
                let resolutionTime = updatedAlert.LastUpdated - updatedAlert.Timestamp
                this.RecordResolutionTime(resolutionTime)
                
                logger.LogInformation($"Alert resolved: {alertId} by {resolvedBy} in {resolutionTime.TotalMinutes:F1} minutes")
                Ok ()
            
            | false, _ ->
                let error = $"Alert not found: {alertId}"
                logger.LogWarning(error)
                Error error
                
        with
        | ex ->
            logger.LogError(ex, $"Failed to resolve alert: {alertId}")
            Error ex.Message
    
    /// Suppress an alert
    member this.SuppressAlert(alertId: string, suppressedBy: string) =
        try
            match activeAlerts.TryGetValue(alertId) with
            | true, alert ->
                let updatedAlert = { 
                    alert with 
                        Status = Suppressed
                        LastUpdated = DateTime.UtcNow 
                }
                activeAlerts.[alertId] <- updatedAlert
                
                this.UpdateStatistics("SuppressedAlerts", 1L)
                
                logger.LogInformation($"Alert suppressed: {alertId} by {suppressedBy}")
                Ok ()
            
            | false, _ ->
                let error = $"Alert not found: {alertId}"
                logger.LogWarning(error)
                Error error
                
        with
        | ex ->
            logger.LogError(ex, $"Failed to suppress alert: {alertId}")
            Error ex.Message
    
    /// Initialize default alert rules
    member private this.InitializeDefaultAlertRules() =
        let defaultRules = [
            {
                Id = "HighCpuUsage"
                Name = "High CPU Usage"
                Description = "CPU usage exceeds threshold"
                Source = "SystemResources"
                Condition = "CpuUsagePercent > 80"
                Severity = Warning
                Threshold = 80.0
                EvaluationInterval = TimeSpan.FromMinutes(1.0)
                EscalationInterval = TimeSpan.FromMinutes(15.0)
                MaxEscalations = 3
                AutoResolve = true
                AutoResolveTimeout = TimeSpan.FromMinutes(5.0)
                Enabled = true
                Tags = ["performance"; "cpu"]
            }
            {
                Id = "HighMemoryUsage"
                Name = "High Memory Usage"
                Description = "Memory usage exceeds threshold"
                Source = "SystemResources"
                Condition = "MemoryUsagePercent > 85"
                Severity = Warning
                Threshold = 85.0
                EvaluationInterval = TimeSpan.FromMinutes(1.0)
                EscalationInterval = TimeSpan.FromMinutes(15.0)
                MaxEscalations = 3
                AutoResolve = true
                AutoResolveTimeout = TimeSpan.FromMinutes(5.0)
                Enabled = true
                Tags = ["performance"; "memory"]
            }
            {
                Id = "ServiceFailure"
                Name = "Service Failure"
                Description = "Critical service component failure"
                Source = "ServiceAvailability"
                Condition = "ServiceDown = true"
                Severity = Critical
                Threshold = 1.0
                EvaluationInterval = TimeSpan.FromMinutes(1.0)
                EscalationInterval = TimeSpan.FromMinutes(5.0)
                MaxEscalations = 5
                AutoResolve = false
                AutoResolveTimeout = TimeSpan.FromHours(1.0)
                Enabled = true
                Tags = ["availability"; "critical"]
            }
        ]
        
        for rule in defaultRules do
            alertRules.[rule.Id] <- rule
            logger.LogDebug($"Initialized alert rule: {rule.Name}")
    
    /// Initialize default notification configs
    member private this.InitializeDefaultNotificationConfigs() =
        let defaultConfigs = [
            {
                Type = EventLog
                Recipients = ["System"]
                Template = "TARS Alert: {Severity} - {Message}"
                EscalationLevels = [Level1; Level2; Level3; Level4]
                Enabled = true
            }
            {
                Type = Email
                Recipients = ["admin@tars.local"]
                Template = "TARS Alert: {Title}\n\nSeverity: {Severity}\nSource: {Source}\nMessage: {Message}\nTimestamp: {Timestamp}"
                EscalationLevels = [Level2; Level3; Level4]
                Enabled = false // Disabled by default
            }
        ]
        
        for config in defaultConfigs do
            notificationConfigs.[config.Type.ToString()] <- config
            logger.LogDebug($"Initialized notification config: {config.Type}")
    
    /// Send notification for alert
    member private this.SendNotificationAsync(alert: Alert) = task {
        try
            for kvp in notificationConfigs do
                let config = kvp.Value
                
                if config.Enabled && config.EscalationLevels |> List.contains alert.EscalationLevel then
                    do! this.SendNotificationByTypeAsync(config, alert)
                    
        with
        | ex ->
            logger.LogWarning(ex, $"Error sending notifications for alert: {alert.Id}")
    }
    
    /// Send notification by type
    member private this.SendNotificationByTypeAsync(config: NotificationConfig, alert: Alert) = task {
        try
            match config.Type with
            | EventLog ->
                // Write to Windows Event Log
                let eventLogLevel = 
                    match alert.Severity with
                    | Critical -> EventLogEntryType.Error
                    | Error -> EventLogEntryType.Error
                    | Warning -> EventLogEntryType.Warning
                    | Info -> EventLogEntryType.Information
                
                let message = config.Template
                    .Replace("{Severity}", alert.Severity.ToString())
                    .Replace("{Source}", alert.Source)
                    .Replace("{Title}", alert.Title)
                    .Replace("{Message}", alert.Message)
                    .Replace("{Timestamp}", alert.Timestamp.ToString())
                
                // In a real implementation, we would write to the actual event log
                logger.LogInformation($"EventLog notification: {message}")
            
            | Email ->
                // Send email notification
                logger.LogInformation($"Email notification would be sent to: {String.Join(", ", config.Recipients)}")
            
            | SMS ->
                // Send SMS notification
                logger.LogInformation($"SMS notification would be sent to: {String.Join(", ", config.Recipients)}")
            
            | Slack ->
                // Send Slack notification
                logger.LogInformation($"Slack notification would be sent to: {String.Join(", ", config.Recipients)}")
            
            | Teams ->
                // Send Teams notification
                logger.LogInformation($"Teams notification would be sent to: {String.Join(", ", config.Recipients)}")
            
            | Webhook ->
                // Send webhook notification
                logger.LogInformation($"Webhook notification would be sent to: {String.Join(", ", config.Recipients)}")
                
        with
        | ex ->
            logger.LogWarning(ex, $"Error sending {config.Type} notification for alert: {alert.Id}")
    }
    
    /// Alert processing loop
    member private this.AlertProcessingLoopAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogDebug("Starting alert processing loop")
            
            while not cancellationToken.IsCancellationRequested && isRunning do
                try
                    // Process auto-resolve alerts
                    this.ProcessAutoResolveAlerts()
                    
                    // Clean up old resolved alerts
                    this.CleanupOldAlerts()
                    
                    // Wait for next processing cycle
                    do! Task.Delay(TimeSpan.FromMinutes(1.0), cancellationToken)
                    
                with
                | :? OperationCanceledException ->
                    break
                | ex ->
                    logger.LogWarning(ex, "Error in alert processing loop")
                    do! Task.Delay(TimeSpan.FromMinutes(1.0), cancellationToken)
                    
        with
        | :? OperationCanceledException ->
            logger.LogDebug("Alert processing loop cancelled")
        | ex ->
            logger.LogError(ex, "Alert processing loop failed")
    }
    
    /// Escalation loop
    member private this.EscalationLoopAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogDebug("Starting alert escalation loop")
            
            while not cancellationToken.IsCancellationRequested && isRunning do
                try
                    // Process alert escalations
                    do! this.ProcessAlertEscalationsAsync()
                    
                    // Wait for next escalation cycle
                    do! Task.Delay(TimeSpan.FromMinutes(5.0), cancellationToken)
                    
                with
                | :? OperationCanceledException ->
                    break
                | ex ->
                    logger.LogWarning(ex, "Error in alert escalation loop")
                    do! Task.Delay(TimeSpan.FromMinutes(5.0), cancellationToken)
                    
        with
        | :? OperationCanceledException ->
            logger.LogDebug("Alert escalation loop cancelled")
        | ex ->
            logger.LogError(ex, "Alert escalation loop failed")
    }
    
    /// Process auto-resolve alerts
    member private this.ProcessAutoResolveAlerts() =
        let alertsToResolve = 
            activeAlerts.Values
            |> Seq.filter (fun alert ->
                match alertRules.TryGetValue(alert.Source) with
                | true, rule when rule.AutoResolve ->
                    DateTime.UtcNow - alert.LastUpdated > rule.AutoResolveTimeout
                | _ -> false)
            |> List.ofSeq
        
        for alert in alertsToResolve do
            this.ResolveAlert(alert.Id, "AutoResolve") |> ignore
            this.UpdateStatistics("AutoResolvedAlerts", 1L)
    
    /// Process alert escalations
    member private this.ProcessAlertEscalationsAsync() = task {
        let alertsToEscalate = 
            activeAlerts.Values
            |> Seq.filter (fun alert ->
                alert.Status = Active &&
                match alertRules.TryGetValue(alert.Source) with
                | true, rule ->
                    alert.EscalationCount < rule.MaxEscalations &&
                    DateTime.UtcNow - alert.LastUpdated > rule.EscalationInterval
                | _ -> false)
            |> List.ofSeq
        
        for alert in alertsToEscalate do
            let newEscalationLevel = 
                match alert.EscalationLevel with
                | Level1 -> Level2
                | Level2 -> Level3
                | Level3 -> Level4
                | Level4 -> Level4
            
            let escalatedAlert = {
                alert with
                    EscalationLevel = newEscalationLevel
                    EscalationCount = alert.EscalationCount + 1
                    LastUpdated = DateTime.UtcNow
            }
            
            activeAlerts.[alert.Id] <- escalatedAlert
            
            // Send escalation notification
            do! this.SendNotificationAsync(escalatedAlert)
            
            this.UpdateStatistics("EscalatedAlerts", 1L)
            logger.LogWarning($"Alert escalated: {alert.Id} to {newEscalationLevel}")
    }
    
    /// Clean up old alerts
    member private this.CleanupOldAlerts() =
        let cutoffTime = DateTime.UtcNow.AddDays(-30.0)
        let alertsToRemove = ResizeArray<Alert>()
        
        for alert in alertHistory do
            if alert.LastUpdated < cutoffTime then
                alertsToRemove.Add(alert)
        
        // Note: ConcurrentQueue doesn't support removal, so this is a simplified approach
        // In production, we'd use a different data structure or database
        logger.LogDebug($"Would clean up {alertsToRemove.Count} old alerts")
    
    /// Update statistics
    member private this.UpdateStatistics(key: string, increment: int64) =
        alertStatistics.AddOrUpdate(key, increment, fun _ current -> current + increment) |> ignore
    
    /// Record resolution time
    member private this.RecordResolutionTime(resolutionTime: TimeSpan) =
        // In a real implementation, we'd maintain a rolling average
        alertStatistics.AddOrUpdate("TotalResolutionTimeMinutes", int64 resolutionTime.TotalMinutes, fun _ current -> current + int64 resolutionTime.TotalMinutes) |> ignore
    
    /// Get active alerts
    member this.GetActiveAlerts() =
        activeAlerts.Values |> List.ofSeq
    
    /// Get alerts by severity
    member this.GetAlertsBySeverity(severity: AlertSeverity) =
        activeAlerts.Values 
        |> Seq.filter (fun a -> a.Severity = severity)
        |> List.ofSeq
    
    /// Get alerts by source
    member this.GetAlertsBySource(source: string) =
        activeAlerts.Values 
        |> Seq.filter (fun a -> a.Source = source)
        |> List.ofSeq
    
    /// Get alert statistics
    member this.GetAlertStatistics() =
        let totalAlerts = alertStatistics.GetOrAdd("TotalAlerts", 0L)
        let resolvedAlerts = alertStatistics.GetOrAdd("ResolvedAlerts", 0L)
        let escalatedAlerts = alertStatistics.GetOrAdd("EscalatedAlerts", 0L)
        let autoResolvedAlerts = alertStatistics.GetOrAdd("AutoResolvedAlerts", 0L)
        let totalResolutionTime = alertStatistics.GetOrAdd("TotalResolutionTimeMinutes", 0L)
        
        let averageResolutionTime = 
            if resolvedAlerts > 0L then
                float totalResolutionTime / float resolvedAlerts
            else 0.0
        
        let escalationRate = 
            if totalAlerts > 0L then
                float escalatedAlerts / float totalAlerts * 100.0
            else 0.0
        
        let autoResolveRate = 
            if resolvedAlerts > 0L then
                float autoResolvedAlerts / float resolvedAlerts * 100.0
            else 0.0
        
        let alertsBySource = 
            activeAlerts.Values
            |> Seq.groupBy (fun a -> a.Source)
            |> Seq.map (fun (source, alerts) -> (source, alerts |> Seq.length))
            |> Map.ofSeq
        
        let alertsBySeverity = 
            activeAlerts.Values
            |> Seq.groupBy (fun a -> a.Severity)
            |> Seq.map (fun (severity, alerts) -> (severity, alerts |> Seq.length))
            |> Map.ofSeq
        
        {
            TotalAlerts = totalAlerts
            ActiveAlerts = activeAlerts.Count
            ResolvedAlerts = resolvedAlerts
            AcknowledgedAlerts = activeAlerts.Values |> Seq.filter (fun a -> a.Status = Acknowledged) |> Seq.length
            SuppressedAlerts = activeAlerts.Values |> Seq.filter (fun a -> a.Status = Suppressed) |> Seq.length
            AlertsBySource = alertsBySource
            AlertsBySeverity = alertsBySeverity
            AverageResolutionTimeMinutes = averageResolutionTime
            EscalationRate = escalationRate
            AutoResolveRate = autoResolveRate
        }
