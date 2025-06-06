namespace TarsEngine.FSharp.ProjectManagement.Kanban

open System
open System.Collections.Generic
open TarsEngine.FSharp.ProjectManagement.Core.AgileFrameworks

/// Kanban workflow management and optimization
module KanbanEngine =
    
    /// Kanban flow metrics
    type FlowMetrics = {
        LeadTime: TimeSpan
        CycleTime: TimeSpan
        Throughput: float
        WipUtilization: float
        BlockedTime: TimeSpan
        FlowEfficiency: float
    }
    
    /// WIP limit violation
    type WipViolation = {
        ColumnId: Guid
        ColumnName: string
        CurrentCount: int
        WipLimit: int
        Severity: ViolationSeverity
        DetectedAt: DateTime
    }
    
    and ViolationSeverity = Warning | Critical
    
    /// Bottleneck analysis
    type Bottleneck = {
        ColumnId: Guid
        ColumnName: string
        AverageWaitTime: TimeSpan
        QueueLength: int
        ThroughputImpact: float
        Recommendations: string list
    }
    
    /// Kanban board operations
    type KanbanBoardService() =
        
        /// Create a new Kanban board
        member _.CreateBoard(name: string, teamId: Guid, columns: KanbanColumn list) =
            {
                Id = Guid.NewGuid()
                Name = name
                Description = ""
                TeamId = teamId
                Columns = columns
                WipLimits = Map.empty
                WorkItems = []
                CreatedDate = DateTime.UtcNow
                UpdatedDate = DateTime.UtcNow
            }
        
        /// Add default columns for software development
        member _.CreateDefaultSoftwareColumns() = [
            { Id = Guid.NewGuid(); Name = "Backlog"; Position = 1; WipLimit = None; Definition = "Items ready for development"; Color = "#E3F2FD" }
            { Id = Guid.NewGuid(); Name = "Analysis"; Position = 2; WipLimit = Some 3; Definition = "Requirements analysis and design"; Color = "#FFF3E0" }
            { Id = Guid.NewGuid(); Name = "Development"; Position = 3; WipLimit = Some 5; Definition = "Active development work"; Color = "#E8F5E8" }
            { Id = Guid.NewGuid(); Name = "Code Review"; Position = 4; WipLimit = Some 3; Definition = "Peer review and quality checks"; Color = "#FFF8E1" }
            { Id = Guid.NewGuid(); Name = "Testing"; Position = 5; WipLimit = Some 4; Definition = "Quality assurance testing"; Color = "#F3E5F5" }
            { Id = Guid.NewGuid(); Name = "Deployment"; Position = 6; WipLimit = Some 2; Definition = "Production deployment"; Color = "#E0F2F1" }
            { Id = Guid.NewGuid(); Name = "Done"; Position = 7; WipLimit = None; Definition = "Completed and delivered"; Color = "#E8F5E8" }
        ]
        
        /// Move work item between columns
        member this.MoveWorkItem(board: KanbanBoard, workItemId: Guid, targetColumnId: Guid) =
            let targetColumn = board.Columns |> List.find (fun c -> c.Id = targetColumnId)
            
            // Check WIP limits
            let currentItemsInColumn = this.GetWorkItemsInColumn(board, targetColumnId) |> List.length
            
            match targetColumn.WipLimit with
            | Some limit when currentItemsInColumn >= limit ->
                Error $"WIP limit exceeded for column '{targetColumn.Name}'. Current: {currentItemsInColumn}, Limit: {limit}"
            | _ ->
                // Move is allowed
                Ok { board with UpdatedDate = DateTime.UtcNow }
        
        /// Get work items in a specific column
        member _.GetWorkItemsInColumn(board: KanbanBoard, columnId: Guid) =
            // This would typically query the work items by status/column
            // For now, returning empty list as placeholder
            []
        
        /// Calculate flow metrics for the board
        member this.CalculateFlowMetrics(board: KanbanBoard, workItems: WorkItem list, period: DateRange) =
            let completedItems = 
                workItems 
                |> List.filter (fun wi -> 
                    wi.Status = Done && 
                    wi.CompletedDate.IsSome &&
                    wi.CompletedDate.Value >= period.StartDate &&
                    wi.CompletedDate.Value <= period.EndDate)
            
            let leadTimes = 
                completedItems
                |> List.choose (fun wi -> 
                    wi.CompletedDate 
                    |> Option.map (fun completed -> completed - wi.CreatedDate))
            
            let averageLeadTime = 
                if leadTimes.IsEmpty then TimeSpan.Zero
                else TimeSpan.FromTicks(leadTimes |> List.map (_.Ticks) |> List.average |> int64)
            
            let throughput = float completedItems.Length / (period.EndDate - period.StartDate).TotalDays
            
            {
                LeadTime = averageLeadTime
                CycleTime = averageLeadTime // Simplified - would calculate actual cycle time
                Throughput = throughput
                WipUtilization = 0.75 // Placeholder calculation
                BlockedTime = TimeSpan.Zero // Would calculate from blocked items
                FlowEfficiency = 0.65 // Placeholder calculation
            }
        
        /// Detect WIP limit violations
        member this.DetectWipViolations(board: KanbanBoard) =
            board.Columns
            |> List.choose (fun column ->
                match column.WipLimit with
                | Some limit ->
                    let currentCount = this.GetWorkItemsInColumn(board, column.Id) |> List.length
                    if currentCount > limit then
                        Some {
                            ColumnId = column.Id
                            ColumnName = column.Name
                            CurrentCount = currentCount
                            WipLimit = limit
                            Severity = if currentCount > limit * 2 then Critical else Warning
                            DetectedAt = DateTime.UtcNow
                        }
                    else None
                | None -> None)
        
        /// Identify bottlenecks in the workflow
        member this.IdentifyBottlenecks(board: KanbanBoard, workItems: WorkItem list) =
            board.Columns
            |> List.map (fun column ->
                let itemsInColumn = this.GetWorkItemsInColumn(board, column.Id)
                let averageWaitTime = TimeSpan.FromHours(2.5) // Placeholder calculation
                
                {
                    ColumnId = column.Id
                    ColumnName = column.Name
                    AverageWaitTime = averageWaitTime
                    QueueLength = itemsInColumn.Length
                    ThroughputImpact = 0.15 // Placeholder calculation
                    Recommendations = [
                        if itemsInColumn.Length > 5 then "Consider increasing team capacity"
                        if averageWaitTime > TimeSpan.FromDays(2) then "Review and optimize process"
                        "Add automation to reduce manual work"
                    ]
                })
            |> List.filter (fun b -> b.QueueLength > 3 || b.AverageWaitTime > TimeSpan.FromDays(1))
        
        /// Generate cumulative flow diagram data
        member _.GenerateCumulativeFlowData(board: KanbanBoard, workItems: WorkItem list, period: DateRange) =
            let dates = 
                let mutable current = period.StartDate
                [
                    while current <= period.EndDate do
                        yield current
                        current <- current.AddDays(1)
                ]
            
            dates
            |> List.map (fun date ->
                let columnCounts = 
                    board.Columns
                    |> List.map (fun column ->
                        let count = 
                            workItems
                            |> List.filter (fun wi -> 
                                wi.CreatedDate <= date && 
                                (wi.CompletedDate.IsNone || wi.CompletedDate.Value > date))
                            |> List.length
                        (column.Name, count))
                    |> Map.ofList
                
                {
                    Date = date
                    ColumnCounts = columnCounts
                })
        
        /// Optimize board configuration
        member this.OptimizeBoard(board: KanbanBoard, metrics: FlowMetrics, violations: WipViolation list) =
            let recommendations = [
                if metrics.FlowEfficiency < 0.5 then
                    "Consider reducing WIP limits to improve flow efficiency"
                
                if violations |> List.exists (fun v -> v.Severity = Critical) then
                    "Critical WIP violations detected - immediate action required"
                
                if metrics.LeadTime > TimeSpan.FromDays(14) then
                    "Lead time is high - consider breaking down work items"
                
                if metrics.Throughput < 1.0 then
                    "Low throughput - review team capacity and process efficiency"
            ]
            
            let optimizedColumns = 
                board.Columns
                |> List.map (fun column ->
                    // Adjust WIP limits based on violations and metrics
                    let adjustedWipLimit = 
                        match column.WipLimit with
                        | Some limit ->
                            let violation = violations |> List.tryFind (fun v -> v.ColumnId = column.Id)
                            match violation with
                            | Some v when v.Severity = Critical -> Some (limit - 1)
                            | _ when metrics.FlowEfficiency < 0.4 -> Some (max 1 (limit - 1))
                            | _ -> Some limit
                        | None -> None
                    
                    { column with WipLimit = adjustedWipLimit })
            
            ({ board with Columns = optimizedColumns }, recommendations)
    
    /// Kanban coaching agent
    type KanbanCoachAgent() =
        let boardService = KanbanBoardService()
        
        /// Provide daily coaching insights
        member _.ProvideDailyInsights(board: KanbanBoard, workItems: WorkItem list) =
            let violations = boardService.DetectWipViolations(board)
            let bottlenecks = boardService.IdentifyBottlenecks(board, workItems)
            
            let insights = [
                if violations.IsEmpty then
                    "✅ No WIP limit violations detected - good flow discipline!"
                else
                    $"⚠️ {violations.Length} WIP limit violations need attention"
                
                if bottlenecks.IsEmpty then
                    "✅ No significant bottlenecks identified"
                else
                    $"🚧 {bottlenecks.Length} potential bottlenecks detected"
                
                let blockedItems = workItems |> List.filter (fun wi -> wi.Status = Blocked)
                if blockedItems.IsEmpty then
                    "✅ No blocked items"
                else
                    $"🚫 {blockedItems.Length} items are blocked - review impediments"
            ]
            
            {|
                Date = DateTime.UtcNow
                Insights = insights
                Violations = violations
                Bottlenecks = bottlenecks
                Recommendations = [
                    "Focus on completing work in progress before starting new items"
                    "Address blocked items in daily standup"
                    "Review and update work item priorities"
                    "Consider pair programming for knowledge sharing"
                ]
            |}
        
        /// Weekly flow analysis
        member _.WeeklyFlowAnalysis(board: KanbanBoard, workItems: WorkItem list) =
            let lastWeek = {
                StartDate = DateTime.UtcNow.AddDays(-7)
                EndDate = DateTime.UtcNow
            }
            
            let metrics = boardService.CalculateFlowMetrics(board, workItems, lastWeek)
            let cfdData = boardService.GenerateCumulativeFlowData(board, workItems, lastWeek)
            
            {|
                Period = lastWeek
                Metrics = metrics
                CumulativeFlowData = cfdData
                Summary = $"Throughput: {metrics.Throughput:F1} items/day, Lead Time: {metrics.LeadTime.TotalDays:F1} days"
                TrendAnalysis = [
                    if metrics.Throughput > 1.5 then "📈 High throughput - team is performing well"
                    elif metrics.Throughput < 0.5 then "📉 Low throughput - investigate impediments"
                    else "📊 Stable throughput"
                    
                    if metrics.LeadTime < TimeSpan.FromDays(5) then "⚡ Fast delivery cycle"
                    elif metrics.LeadTime > TimeSpan.FromDays(15) then "🐌 Long lead times - consider process improvements"
                    else "⏱️ Reasonable lead times"
                ]
            |}
        
        /// Continuous improvement suggestions
        member _.SuggestImprovements(board: KanbanBoard, metrics: FlowMetrics, teamFeedback: string list) =
            [
                // Flow-based improvements
                if metrics.FlowEfficiency < 0.6 then
                    "Implement pull-based work assignment to improve flow"
                
                if metrics.WipUtilization > 0.9 then
                    "Reduce WIP limits to prevent overloading the team"
                
                // Process improvements
                "Add definition of done for each column"
                "Implement regular retrospectives for continuous improvement"
                "Consider adding swim lanes for different work types"
                "Set up automated notifications for blocked items"
                
                // Team improvements based on feedback
                yield! teamFeedback |> List.map (fun feedback ->
                    $"Address team concern: {feedback}")
            ]
