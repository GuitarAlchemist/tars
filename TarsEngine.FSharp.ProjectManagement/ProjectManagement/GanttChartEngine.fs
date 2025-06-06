namespace TarsEngine.FSharp.ProjectManagement.ProjectManagement

open System
open System.Collections.Generic
open TarsEngine.FSharp.ProjectManagement.Core.AgileFrameworks

/// Gantt chart generation and project timeline management
module GanttChartEngine =
    
    /// Critical path analysis result
    type CriticalPathAnalysis = {
        CriticalPath: Guid list
        TotalDuration: TimeSpan
        FloatTimes: Map<Guid, TimeSpan>
        CriticalTasks: GanttTask list
        ProjectEndDate: DateTime
    }
    
    /// Resource allocation data
    type ResourceAllocation = {
        ResourceId: Guid
        TaskId: Guid
        AllocationPercentage: float
        StartDate: DateTime
        EndDate: DateTime
        IsOverallocated: bool
    }
    
    /// Project timeline service
    type GanttChartService() =
        
        /// Create Gantt chart from work items
        member _.CreateGanttChart(project: Project, workItems: WorkItem list, resources: TeamMember list) =
            let ganttTasks = 
                workItems
                |> List.map (fun wi ->
                    let duration = 
                        match wi.EstimatedHours with
                        | Some hours -> TimeSpan.FromHours(hours)
                        | None -> TimeSpan.FromDays(3) // Default 3 days
                    
                    let startDate = 
                        match wi.CreatedDate with
                        | date when date > project.StartDate -> date
                        | _ -> project.StartDate
                    
                    {
                        Id = wi.Id
                        Name = wi.Title
                        StartDate = startDate
                        EndDate = startDate.Add(duration)
                        Duration = duration
                        Progress = if wi.Status = Done then 1.0 else 0.0
                        Dependencies = wi.Dependencies
                        AssignedResources = wi.Assignee |> Option.map List.singleton |> Option.defaultValue []
                        IsMilestone = wi.Type = Epic
                        IsCriticalPath = false // Will be calculated
                        ParentTask = wi.EpicId
                        SubTasks = []
                    })
            
            {
                Id = Guid.NewGuid()
                ProjectId = project.Id
                Name = $"{project.Name} - Gantt Chart"
                Tasks = ganttTasks
                CriticalPath = [] // Will be calculated
                Baseline = None
                CreatedDate = DateTime.UtcNow
                UpdatedDate = DateTime.UtcNow
            }
        
        /// Calculate critical path using CPM (Critical Path Method)
        member _.CalculateCriticalPath(ganttChart: GanttChart) =
            let tasks = ganttChart.Tasks |> List.map (fun t -> (t.Id, t)) |> Map.ofList
            
            // Forward pass - calculate earliest start and finish times
            let mutable earliestTimes = Map.empty<Guid, DateTime * DateTime>
            
            let rec calculateEarliest (taskId: Guid) =
                if earliestTimes.ContainsKey(taskId) then
                    earliestTimes.[taskId]
                else
                    let task = tasks.[taskId]
                    let dependencyFinishTimes = 
                        task.Dependencies
                        |> List.map (fun depId -> 
                            let (_, finishTime) = calculateEarliest depId
                            finishTime)
                    
                    let earliestStart = 
                        if dependencyFinishTimes.IsEmpty then task.StartDate
                        else dependencyFinishTimes |> List.max
                    
                    let earliestFinish = earliestStart.Add(task.Duration)
                    earliestTimes <- earliestTimes.Add(taskId, (earliestStart, earliestFinish))
                    (earliestStart, earliestFinish)
            
            // Calculate for all tasks
            ganttChart.Tasks |> List.iter (fun t -> calculateEarliest t.Id |> ignore)
            
            // Backward pass - calculate latest start and finish times
            let projectEndDate = 
                earliestTimes.Values 
                |> Seq.map snd 
                |> Seq.max
            
            let mutable latestTimes = Map.empty<Guid, DateTime * DateTime>
            
            let rec calculateLatest (taskId: Guid) =
                if latestTimes.ContainsKey(taskId) then
                    latestTimes.[taskId]
                else
                    let task = tasks.[taskId]
                    let successors = 
                        ganttChart.Tasks 
                        |> List.filter (fun t -> t.Dependencies.Contains(taskId))
                    
                    let latestFinish = 
                        if successors.IsEmpty then projectEndDate
                        else 
                            successors
                            |> List.map (fun s -> 
                                let (startTime, _) = calculateLatest s.Id
                                startTime)
                            |> List.min
                    
                    let latestStart = latestFinish.Subtract(task.Duration)
                    latestTimes <- latestTimes.Add(taskId, (latestStart, latestFinish))
                    (latestStart, latestFinish)
            
            // Calculate for all tasks (reverse order)
            ganttChart.Tasks |> List.rev |> List.iter (fun t -> calculateLatest t.Id |> ignore)
            
            // Identify critical path (tasks with zero float)
            let floatTimes = 
                ganttChart.Tasks
                |> List.map (fun task ->
                    let (earliestStart, _) = earliestTimes.[task.Id]
                    let (latestStart, _) = latestTimes.[task.Id]
                    let float = latestStart - earliestStart
                    (task.Id, float))
                |> Map.ofList
            
            let criticalTaskIds = 
                floatTimes
                |> Map.filter (fun _ float -> float = TimeSpan.Zero)
                |> Map.keys
                |> List.ofSeq
            
            let criticalTasks = 
                ganttChart.Tasks
                |> List.filter (fun t -> criticalTaskIds.Contains(t.Id))
            
            {
                CriticalPath = criticalTaskIds
                TotalDuration = projectEndDate - ganttChart.Tasks.[0].StartDate
                FloatTimes = floatTimes
                CriticalTasks = criticalTasks
                ProjectEndDate = projectEndDate
            }
        
        /// Analyze resource allocation and conflicts
        member _.AnalyzeResourceAllocation(ganttChart: GanttChart, resources: TeamMember list) =
            let allocations = 
                ganttChart.Tasks
                |> List.collect (fun task ->
                    task.AssignedResources
                    |> List.map (fun resourceId ->
                        {
                            ResourceId = resourceId
                            TaskId = task.Id
                            AllocationPercentage = 1.0 / float task.AssignedResources.Length
                            StartDate = task.StartDate
                            EndDate = task.EndDate
                            IsOverallocated = false // Will be calculated
                        }))
            
            // Check for overallocation
            let overallocatedResources = 
                resources
                |> List.choose (fun resource ->
                    let resourceAllocations = 
                        allocations 
                        |> List.filter (fun a -> a.ResourceId = resource.Id)
                    
                    // Check for overlapping time periods
                    let overlaps = 
                        resourceAllocations
                        |> List.collect (fun a1 ->
                            resourceAllocations
                            |> List.filter (fun a2 -> 
                                a1.TaskId <> a2.TaskId &&
                                a1.StartDate < a2.EndDate &&
                                a2.StartDate < a1.EndDate)
                            |> List.map (fun a2 -> (a1, a2)))
                    
                    if overlaps.IsEmpty then None
                    else Some (resource, overlaps))
            
            {|
                Allocations = allocations
                OverallocatedResources = overallocatedResources
                ResourceUtilization = 
                    resources
                    |> List.map (fun r ->
                        let totalHours = 
                            allocations
                            |> List.filter (fun a -> a.ResourceId = r.Id)
                            |> List.sumBy (fun a -> (a.EndDate - a.StartDate).TotalHours * a.AllocationPercentage)
                        (r.Id, totalHours / (r.Capacity * 40.0))) // Assuming 40-hour work week
                    |> Map.ofList
            |}
        
        /// Generate project timeline with milestones
        member _.GenerateProjectTimeline(project: Project, ganttChart: GanttChart) =
            let milestones = 
                project.Milestones
                |> List.map (fun m ->
                    {|
                        Id = m.Id
                        Name = m.Name
                        Date = m.DueDate
                        Status = m.Status
                        Dependencies = m.Dependencies
                        IsOnTrack = m.DueDate >= DateTime.UtcNow || m.Status = Completed
                    |})
            
            let phases = 
                ganttChart.Tasks
                |> List.filter (fun t -> t.ParentTask.IsNone)
                |> List.groupBy (fun t -> t.StartDate.ToString("yyyy-MM"))
                |> List.map (fun (month, tasks) ->
                    {|
                        Phase = month
                        Tasks = tasks
                        StartDate = tasks |> List.map (_.StartDate) |> List.min
                        EndDate = tasks |> List.map (_.EndDate) |> List.max
                        Progress = tasks |> List.averageBy (_.Progress)
                    |})
            
            {|
                ProjectId = project.Id
                ProjectName = project.Name
                StartDate = project.StartDate
                EndDate = project.EndDate
                Milestones = milestones
                Phases = phases
                OverallProgress = ganttChart.Tasks |> List.averageBy (_.Progress)
                Status = project.Status
                RiskLevel = 
                    let overdueMilestones = milestones |> List.filter (fun m -> not m.IsOnTrack)
                    if overdueMilestones.Length > 2 then "High"
                    elif overdueMilestones.Length > 0 then "Medium"
                    else "Low"
            |}
        
        /// Create baseline for comparison
        member _.CreateBaseline(ganttChart: GanttChart, name: string) =
            let baseline = {
                Name = name
                CreatedDate = DateTime.UtcNow
                Tasks = ganttChart.Tasks |> List.map (fun t -> (t.Id, t)) |> Map.ofMap
            }
            
            { ganttChart with Baseline = Some baseline }
        
        /// Compare current progress with baseline
        member _.CompareWithBaseline(ganttChart: GanttChart) =
            match ganttChart.Baseline with
            | None -> None
            | Some baseline ->
                let variances = 
                    ganttChart.Tasks
                    |> List.choose (fun currentTask ->
                        baseline.Tasks.TryFind(currentTask.Id)
                        |> Option.map (fun baselineTask ->
                            {|
                                TaskId = currentTask.Id
                                TaskName = currentTask.Name
                                ScheduleVariance = (currentTask.EndDate - baselineTask.EndDate).TotalDays
                                ProgressVariance = currentTask.Progress - baselineTask.Progress
                                IsDelayed = currentTask.EndDate > baselineTask.EndDate
                                IsCritical = currentTask.IsCriticalPath
                            |}))
                
                Some {|
                    BaselineName = baseline.Name
                    BaselineDate = baseline.CreatedDate
                    TaskVariances = variances
                    OverallScheduleVariance = variances |> List.averageBy (_.ScheduleVariance)
                    DelayedTasks = variances |> List.filter (_.IsDelayed)
                    CriticalDelayedTasks = variances |> List.filter (fun v -> v.IsDelayed && v.IsCritical)
                |}
        
        /// Generate what-if scenarios
        member _.GenerateWhatIfScenarios(ganttChart: GanttChart, scenarios: Map<string, float>) =
            scenarios
            |> Map.map (fun scenarioName multiplier ->
                let adjustedTasks = 
                    ganttChart.Tasks
                    |> List.map (fun task ->
                        let newDuration = TimeSpan.FromTicks(int64 (float task.Duration.Ticks * multiplier))
                        { task with 
                            Duration = newDuration
                            EndDate = task.StartDate.Add(newDuration) })
                
                let adjustedChart = { ganttChart with Tasks = adjustedTasks }
                let criticalPath = this.CalculateCriticalPath(adjustedChart)
                
                {|
                    ScenarioName = scenarioName
                    Multiplier = multiplier
                    NewProjectEndDate = criticalPath.ProjectEndDate
                    DelayDays = (criticalPath.ProjectEndDate - ganttChart.Tasks |> List.map (_.EndDate) |> List.max).TotalDays
                    Impact = 
                        if multiplier > 1.2 then "High Risk"
                        elif multiplier > 1.1 then "Medium Risk"
                        else "Low Risk"
                |})
    
    /// Project Manager agent for timeline management
    type ProjectManagerAgent() =
        let ganttService = GanttChartService()
        
        /// Generate executive dashboard
        member _.GenerateExecutiveDashboard(projects: Project list, ganttCharts: GanttChart list) =
            let totalProjects = projects.Length
            let activeProjects = projects |> List.filter (fun p -> p.Status = Execution) |> List.length
            let completedProjects = projects |> List.filter (fun p -> p.Status = Closure) |> List.length
            let delayedProjects = 
                projects 
                |> List.filter (fun p -> 
                    p.EndDate.IsSome && p.EndDate.Value < DateTime.UtcNow && p.Status <> Closure)
                |> List.length
            
            {|
                Summary = {|
                    TotalProjects = totalProjects
                    ActiveProjects = activeProjects
                    CompletedProjects = completedProjects
                    DelayedProjects = delayedProjects
                    OnTimeDeliveryRate = float completedProjects / float totalProjects
                |}
                ProjectHealth = 
                    projects
                    |> List.map (fun p ->
                        let health = 
                            if p.Status = Closure then "Completed"
                            elif p.EndDate.IsSome && p.EndDate.Value < DateTime.UtcNow then "Delayed"
                            elif p.Status = OnHold then "On Hold"
                            else "On Track"
                        
                        {|
                            ProjectId = p.Id
                            ProjectName = p.Name
                            Health = health
                            Progress = 0.75 // Would calculate actual progress
                            Budget = p.Budget
                            EndDate = p.EndDate
                        |})
                Recommendations = [
                    if delayedProjects > 0 then
                        $"Review {delayedProjects} delayed projects for recovery actions"
                    if float activeProjects / float totalProjects > 0.8 then
                        "High project load - consider resource allocation"
                    "Conduct monthly project health reviews"
                    "Implement risk mitigation strategies"
                ]
            |}
