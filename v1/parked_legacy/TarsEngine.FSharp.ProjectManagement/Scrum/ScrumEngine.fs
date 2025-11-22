namespace TarsEngine.FSharp.ProjectManagement.Scrum

open System
open System.Collections.Generic
open TarsEngine.FSharp.ProjectManagement.Core.AgileFrameworks

/// Scrum framework implementation with ceremonies and metrics
module ScrumEngine =
    
    /// Sprint planning result
    type SprintPlanningResult = {
        Sprint: Sprint
        CommittedStoryPoints: int
        TeamCapacity: float
        CapacityUtilization: float
        RiskFactors: string list
        Confidence: int // 1-5 scale
    }
    
    /// Daily standup data
    type DailyStandupData = {
        Date: DateTime
        TeamMember: Guid
        Yesterday: string list
        Today: string list
        Impediments: string list
        Confidence: int // 1-5 scale for sprint goal
    }
    
    /// Sprint review metrics
    type SprintReviewMetrics = {
        SprintId: Guid
        PlannedStoryPoints: int
        CompletedStoryPoints: int
        SprintGoalAchieved: bool
        Velocity: float
        BurndownData: BurndownPoint list
        QualityMetrics: QualityMetrics
        TeamSatisfaction: float
    }
    
    and QualityMetrics = {
        DefectsFound: int
        DefectsFixed: int
        CodeCoverage: float
        TechnicalDebtAdded: float
        TechnicalDebtReduced: float
    }
    
    /// Scrum Master agent for facilitating ceremonies
    type ScrumMasterAgent() =
        
        /// Facilitate sprint planning
        member _.FacilitateSprintPlanning(team: AgileTeam, backlog: WorkItem list, sprintLength: int) =
            let sprintCapacity = team.Capacity * float sprintLength
            let averageVelocity = 
                if team.Velocity.IsEmpty then 20.0
                else team.Velocity |> List.average
            
            // Select items for sprint based on priority and capacity
            let mutable remainingCapacity = sprintCapacity
            let mutable selectedItems = []
            let mutable totalStoryPoints = 0
            
            for item in backlog |> List.sortBy (fun i -> int i.Priority) do
                let storyPoints = 
                    match item.StoryPoints with
                    | Some sp -> int sp
                    | None -> 3 // Default estimation
                
                let estimatedHours = float storyPoints * 6.0 // Rough conversion
                
                if remainingCapacity >= estimatedHours then
                    selectedItems <- item :: selectedItems
                    totalStoryPoints <- totalStoryPoints + storyPoints
                    remainingCapacity <- remainingCapacity - estimatedHours
            
            let sprint = {
                Id = Guid.NewGuid()
                Name = $"Sprint {DateTime.UtcNow:yyyy-MM-dd}"
                Goal = "Deliver committed user stories with high quality"
                StartDate = DateTime.UtcNow
                EndDate = DateTime.UtcNow.AddDays(float sprintLength)
                TeamId = team.Id
                Capacity = sprintCapacity
                CommittedStoryPoints = totalStoryPoints
                CompletedStoryPoints = 0
                WorkItems = selectedItems |> List.map (_.Id)
                Status = Planning
                Retrospective = None
            }
            
            let riskFactors = [
                if totalStoryPoints > int (averageVelocity * 1.2) then
                    "Committed story points exceed historical velocity"
                if remainingCapacity < sprintCapacity * 0.1 then
                    "Very high capacity utilization - little buffer for unexpected work"
                if selectedItems |> List.exists (fun i -> i.Dependencies.Length > 0) then
                    "Sprint contains items with external dependencies"
            ]
            
            {
                Sprint = sprint
                CommittedStoryPoints = totalStoryPoints
                TeamCapacity = sprintCapacity
                CapacityUtilization = (sprintCapacity - remainingCapacity) / sprintCapacity
                RiskFactors = riskFactors
                Confidence = if riskFactors.Length <= 1 then 5 else 3
            }
        
        /// Generate daily standup questions
        member _.GenerateStandupQuestions(teamMember: TeamMember, sprint: Sprint) =
            [
                "What did you accomplish yesterday?"
                "What will you work on today?"
                "Are there any impediments blocking your progress?"
                $"How confident are you (1-5) that we'll achieve the sprint goal: '{sprint.Goal}'?"
                "Do you need help from anyone on the team?"
            ]
        
        /// Analyze daily standup responses
        member _.AnalyzeStandupResponses(responses: DailyStandupData list) =
            let impediments = 
                responses 
                |> List.collect (_.Impediments)
                |> List.distinct
            
            let averageConfidence = 
                responses 
                |> List.map (_.Confidence)
                |> List.average
            
            let riskIndicators = [
                if averageConfidence < 3.0 then
                    "Low team confidence in sprint goal achievement"
                if impediments.Length > 3 then
                    "High number of impediments reported"
                if responses |> List.exists (fun r -> r.Today.IsEmpty) then
                    "Some team members have no planned work"
            ]
            
            {|
                Date = DateTime.UtcNow
                TotalImpediments = impediments.Length
                AverageConfidence = averageConfidence
                RiskIndicators = riskIndicators
                ActionItems = [
                    yield! impediments |> List.map (fun imp -> $"Resolve impediment: {imp}")
                    if averageConfidence < 3.0 then
                        "Review sprint scope and adjust if necessary"
                ]
            |}
        
        /// Facilitate sprint review
        member _.FacilitateSprintReview(sprint: Sprint, completedItems: WorkItem list) =
            let completedStoryPoints = 
                completedItems
                |> List.sumBy (fun item -> 
                    match item.StoryPoints with
                    | Some sp -> int sp
                    | None -> 0)
            
            let velocity = float completedStoryPoints
            let goalAchieved = completedStoryPoints >= int (float sprint.CommittedStoryPoints * 0.8)
            
            // Generate burndown data (simplified)
            let burndownData = 
                [0..14] // Assuming 2-week sprint
                |> List.map (fun day ->
                    let date = sprint.StartDate.AddDays(float day)
                    let idealRemaining = float sprint.CommittedStoryPoints * (1.0 - float day / 14.0)
                    let actualRemaining = 
                        if day <= 10 then idealRemaining * 1.1 // Slightly behind
                        else float (sprint.CommittedStoryPoints - completedStoryPoints)
                    
                    {
                        Date = date
                        RemainingWork = actualRemaining
                        IdealRemaining = idealRemaining
                    })
            
            {
                SprintId = sprint.Id
                PlannedStoryPoints = sprint.CommittedStoryPoints
                CompletedStoryPoints = completedStoryPoints
                SprintGoalAchieved = goalAchieved
                Velocity = velocity
                BurndownData = burndownData
                QualityMetrics = {
                    DefectsFound = 2
                    DefectsFixed = 2
                    CodeCoverage = 0.85
                    TechnicalDebtAdded = 0.5
                    TechnicalDebtReduced = 1.2
                }
                TeamSatisfaction = 4.2
            }
        
        /// Facilitate sprint retrospective
        member _.FacilitateRetrospective(team: AgileTeam, sprintMetrics: SprintReviewMetrics) =
            let wentWell = [
                if sprintMetrics.SprintGoalAchieved then
                    "Successfully achieved sprint goal"
                if sprintMetrics.Velocity >= 20.0 then
                    "High team velocity"
                if sprintMetrics.QualityMetrics.CodeCoverage > 0.8 then
                    "Maintained good code coverage"
                if sprintMetrics.TeamSatisfaction > 4.0 then
                    "High team satisfaction"
            ]
            
            let needsImprovement = [
                if not sprintMetrics.SprintGoalAchieved then
                    "Sprint goal not fully achieved"
                if sprintMetrics.QualityMetrics.DefectsFound > 3 then
                    "Higher than expected defect count"
                if sprintMetrics.QualityMetrics.TechnicalDebtAdded > 1.0 then
                    "Technical debt increased significantly"
                if sprintMetrics.TeamSatisfaction < 3.5 then
                    "Team satisfaction below target"
            ]
            
            let actionItems = [
                if not sprintMetrics.SprintGoalAchieved then
                    {
                        Id = Guid.NewGuid()
                        Description = "Review sprint planning process and capacity estimation"
                        AssigneeId = None
                        DueDate = Some (DateTime.UtcNow.AddDays(7))
                        Status = Open
                    }
                if sprintMetrics.QualityMetrics.DefectsFound > 3 then
                    {
                        Id = Guid.NewGuid()
                        Description = "Implement additional code review practices"
                        AssigneeId = None
                        DueDate = Some (DateTime.UtcNow.AddDays(14))
                        Status = Open
                    }
            ]
            
            {
                WentWell = wentWell
                NeedsImprovement = needsImprovement
                ActionItems = actionItems
                TeamMood = int sprintMetrics.TeamSatisfaction
                Date = DateTime.UtcNow
            }
        
        /// Generate sprint health report
        member _.GenerateSprintHealthReport(sprint: Sprint, currentMetrics: SprintReviewMetrics) =
            let daysIntoSprint = (DateTime.UtcNow - sprint.StartDate).TotalDays
            let sprintProgress = daysIntoSprint / (sprint.EndDate - sprint.StartDate).TotalDays
            
            let expectedCompletion = float sprint.CommittedStoryPoints * sprintProgress
            let actualCompletion = float currentMetrics.CompletedStoryPoints
            
            let healthStatus = 
                if actualCompletion >= expectedCompletion * 0.9 then "Green"
                elif actualCompletion >= expectedCompletion * 0.7 then "Yellow"
                else "Red"
            
            {|
                SprintId = sprint.Id
                HealthStatus = healthStatus
                Progress = sprintProgress
                ExpectedCompletion = expectedCompletion
                ActualCompletion = actualCompletion
                RemainingDays = (sprint.EndDate - DateTime.UtcNow).TotalDays
                Recommendations = [
                    if healthStatus = "Red" then
                        "Consider descoping some work items"
                        "Identify and remove impediments"
                        "Increase team focus on sprint goal"
                    elif healthStatus = "Yellow" then
                        "Monitor progress closely"
                        "Address any emerging impediments quickly"
                    else
                        "Sprint is on track - maintain current pace"
                ]
            |}
    
    /// Product Owner agent for backlog management
    type ProductOwnerAgent() =
        
        /// Prioritize backlog items
        member _.PrioritizeBacklog(backlog: WorkItem list, businessValue: Map<Guid, int>) =
            backlog
            |> List.sortByDescending (fun item ->
                let value = businessValue.TryFind(item.Id) |> Option.defaultValue 5
                let urgency = 
                    match item.Priority with
                    | Priority.Critical -> 10
                    | Priority.High -> 8
                    | Priority.Medium -> 5
                    | Priority.Low -> 3
                    | Priority.Backlog -> 1
                
                let effort = 
                    match item.StoryPoints with
                    | Some sp -> 10 - int sp // Lower effort = higher priority
                    | None -> 5
                
                // Weighted score: 50% value, 30% urgency, 20% effort
                float value * 0.5 + float urgency * 0.3 + float effort * 0.2)
        
        /// Generate acceptance criteria
        member _.GenerateAcceptanceCriteria(workItem: WorkItem) =
            match workItem.Type with
            | UserStory ->
                [
                    "Given the user is authenticated"
                    "When they perform the specified action"
                    "Then the expected outcome is achieved"
                    "And the system maintains data integrity"
                    "And appropriate logging is performed"
                ]
            | Bug ->
                [
                    "Given the bug reproduction steps"
                    "When the fix is applied"
                    "Then the issue no longer occurs"
                    "And no regression is introduced"
                    "And the fix is covered by tests"
                ]
            | _ ->
                [
                    "Clear definition of done is established"
                    "Quality criteria are met"
                    "Documentation is updated if needed"
                ]
