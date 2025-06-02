namespace TarsEngine.FSharp.Requirements.Repository

open System
open System.Threading.Tasks
open System.Collections.Concurrent
open TarsEngine.FSharp.Requirements.Models

/// <summary>
/// In-memory implementation of the requirement repository
/// Real implementation for testing and development
/// </summary>
type InMemoryRequirementRepository() =
    
    let requirements = ConcurrentDictionary<string, Requirement>()
    let testCases = ConcurrentDictionary<string, TestCase>()
    let traceabilityLinks = ConcurrentDictionary<string, TraceabilityLink>()
    let testExecutionResults = ConcurrentDictionary<string, TestExecutionResult list>()
    
    /// <summary>
    /// Helper to convert sync operations to async
    /// </summary>
    let asyncResult<'T> (result: 'T) = Task.FromResult(result)
    
    /// <summary>
    /// Helper to filter requirements by predicate
    /// </summary>
    let filterRequirements (predicate: Requirement -> bool) =
        requirements.Values
        |> Seq.filter predicate
        |> Seq.sortByDescending (fun r -> r.CreatedAt)
        |> List.ofSeq
    
    /// <summary>
    /// Helper to search requirements by text
    /// </summary>
    let searchRequirements (searchText: string) =
        let lowerSearchText = searchText.ToLowerInvariant()
        filterRequirements (fun r ->
            r.Title.ToLowerInvariant().Contains(lowerSearchText) ||
            r.Description.ToLowerInvariant().Contains(lowerSearchText) ||
            r.Tags |> List.exists (fun tag -> tag.ToLowerInvariant().Contains(lowerSearchText))
        )
    
    interface IRequirementRepository with
        
        member this.InitializeDatabaseAsync() = task {
            // No initialization needed for in-memory
            return Ok ()
        }
        
        member this.CreateRequirementAsync(requirement: Requirement) = task {
            try
                if requirements.TryAdd(requirement.Id, requirement) then
                    return Ok requirement.Id
                else
                    return Error $"Requirement with ID {requirement.Id} already exists"
            with
            | ex -> return Error $"Failed to create requirement: {ex.Message}"
        }
        
        member this.GetRequirementAsync(id: string) = task {
            try
                let found = requirements.TryGetValue(id)
                return Ok (if fst found then Some (snd found) else None)
            with
            | ex -> return Error $"Failed to get requirement: {ex.Message}"
        }
        
        member this.UpdateRequirementAsync(requirement: Requirement) = task {
            try
                if requirements.ContainsKey(requirement.Id) then
                    requirements.[requirement.Id] <- requirement
                    return Ok ()
                else
                    return Error $"Requirement with ID {requirement.Id} not found"
            with
            | ex -> return Error $"Failed to update requirement: {ex.Message}"
        }
        
        member this.DeleteRequirementAsync(id: string) = task {
            try
                let removed = requirements.TryRemove(id)
                if fst removed then
                    return Ok ()
                else
                    return Error $"Requirement with ID {id} not found"
            with
            | ex -> return Error $"Failed to delete requirement: {ex.Message}"
        }
        
        member this.ListRequirementsAsync() = task {
            try
                let result = filterRequirements (fun _ -> true)
                return Ok result
            with
            | ex -> return Error $"Failed to list requirements: {ex.Message}"
        }
        
        member this.GetRequirementsByTypeAsync(reqType: RequirementType) = task {
            try
                let result = filterRequirements (fun r -> r.Type = reqType)
                return Ok result
            with
            | ex -> return Error $"Failed to get requirements by type: {ex.Message}"
        }
        
        member this.GetRequirementsByStatusAsync(status: RequirementStatus) = task {
            try
                let result = filterRequirements (fun r -> r.Status = status)
                return Ok result
            with
            | ex -> return Error $"Failed to get requirements by status: {ex.Message}"
        }
        
        member this.GetRequirementsByPriorityAsync(priority: RequirementPriority) = task {
            try
                let result = filterRequirements (fun r -> r.Priority = priority)
                return Ok result
            with
            | ex -> return Error $"Failed to get requirements by priority: {ex.Message}"
        }
        
        member this.GetRequirementsByAssigneeAsync(assignee: string) = task {
            try
                let result = filterRequirements (fun r -> r.Assignee = Some assignee)
                return Ok result
            with
            | ex -> return Error $"Failed to get requirements by assignee: {ex.Message}"
        }
        
        member this.SearchRequirementsAsync(searchText: string) = task {
            try
                let result = searchRequirements searchText
                return Ok result
            with
            | ex -> return Error $"Failed to search requirements: {ex.Message}"
        }
        
        member this.GetRequirementsByTagAsync(tag: string) = task {
            try
                let result = filterRequirements (fun r -> r.Tags |> List.contains tag)
                return Ok result
            with
            | ex -> return Error $"Failed to get requirements by tag: {ex.Message}"
        }
        
        member this.GetOverdueRequirementsAsync() = task {
            try
                let now = DateTime.UtcNow
                let result = filterRequirements (fun r ->
                    match r.TargetDate with
                    | Some targetDate -> 
                        targetDate < now && 
                        r.Status <> RequirementStatus.Verified &&
                        r.Status <> RequirementStatus.Rejected &&
                        r.Status <> RequirementStatus.Obsolete
                    | None -> false
                )
                return Ok result
            with
            | ex -> return Error $"Failed to get overdue requirements: {ex.Message}"
        }
        
        // Test Case methods - basic implementations
        member this.CreateTestCaseAsync(testCase: TestCase) = task {
            try
                if testCases.TryAdd(testCase.Id, testCase) then
                    return Ok testCase.Id
                else
                    return Error $"Test case with ID {testCase.Id} already exists"
            with
            | ex -> return Error $"Failed to create test case: {ex.Message}"
        }
        
        member this.GetTestCaseAsync(id: string) = task {
            try
                let found = testCases.TryGetValue(id)
                return Ok (if fst found then Some (snd found) else None)
            with
            | ex -> return Error $"Failed to get test case: {ex.Message}"
        }
        
        member this.UpdateTestCaseAsync(testCase: TestCase) = task {
            try
                if testCases.ContainsKey(testCase.Id) then
                    testCases.[testCase.Id] <- testCase
                    return Ok ()
                else
                    return Error $"Test case with ID {testCase.Id} not found"
            with
            | ex -> return Error $"Failed to update test case: {ex.Message}"
        }
        
        member this.DeleteTestCaseAsync(id: string) = task {
            try
                let removed = testCases.TryRemove(id)
                if fst removed then
                    return Ok ()
                else
                    return Error $"Test case with ID {id} not found"
            with
            | ex -> return Error $"Failed to delete test case: {ex.Message}"
        }
        
        member this.ListTestCasesAsync() = task {
            try
                let result = 
                    testCases.Values
                    |> Seq.sortByDescending (fun tc -> tc.CreatedAt)
                    |> List.ofSeq
                return Ok result
            with
            | ex -> return Error $"Failed to list test cases: {ex.Message}"
        }
        
        member this.GetTestCasesByRequirementAsync(requirementId: string) = task {
            try
                let result = 
                    testCases.Values
                    |> Seq.filter (fun tc -> tc.RequirementId = requirementId)
                    |> Seq.sortByDescending (fun tc -> tc.CreatedAt)
                    |> List.ofSeq
                return Ok result
            with
            | ex -> return Error $"Failed to get test cases by requirement: {ex.Message}"
        }
        
        // Test execution methods
        member this.SaveTestExecutionResultAsync(result: TestExecutionResult) = task {
            try
                let existingResults = 
                    match testExecutionResults.TryGetValue(result.TestCaseId) with
                    | true, results -> results
                    | false, _ -> []
                
                let updatedResults = result :: existingResults
                testExecutionResults.[result.TestCaseId] <- updatedResults
                return Ok ()
            with
            | ex -> return Error $"Failed to save test execution result: {ex.Message}"
        }
        
        member this.GetTestExecutionHistoryAsync(testCaseId: string) = task {
            try
                let results = 
                    match testExecutionResults.TryGetValue(testCaseId) with
                    | true, results -> results |> List.sortByDescending (fun r -> r.StartTime)
                    | false, _ -> []
                return Ok results
            with
            | ex -> return Error $"Failed to get test execution history: {ex.Message}"
        }
        
        member this.GetLatestTestResultAsync(testCaseId: string) = task {
            try
                let result = 
                    match testExecutionResults.TryGetValue(testCaseId) with
                    | true, results -> results |> List.sortByDescending (fun r -> r.StartTime) |> List.tryHead
                    | false, _ -> None
                return Ok result
            with
            | ex -> return Error $"Failed to get latest test result: {ex.Message}"
        }

        // Traceability methods
        member this.CreateTraceabilityLinkAsync(link: TraceabilityLink) = task {
            try
                if traceabilityLinks.TryAdd(link.Id, link) then
                    return Ok link.Id
                else
                    return Error $"Traceability link with ID {link.Id} already exists"
            with
            | ex -> return Error $"Failed to create traceability link: {ex.Message}"
        }

        member this.GetTraceabilityLinkAsync(id: string) = task {
            try
                let found = traceabilityLinks.TryGetValue(id)
                return Ok (if fst found then Some (snd found) else None)
            with
            | ex -> return Error $"Failed to get traceability link: {ex.Message}"
        }

        member this.UpdateTraceabilityLinkAsync(link: TraceabilityLink) = task {
            try
                if traceabilityLinks.ContainsKey(link.Id) then
                    traceabilityLinks.[link.Id] <- link
                    return Ok ()
                else
                    return Error $"Traceability link with ID {link.Id} not found"
            with
            | ex -> return Error $"Failed to update traceability link: {ex.Message}"
        }

        member this.DeleteTraceabilityLinkAsync(id: string) = task {
            try
                let removed = traceabilityLinks.TryRemove(id)
                if fst removed then
                    return Ok ()
                else
                    return Error $"Traceability link with ID {id} not found"
            with
            | ex -> return Error $"Failed to delete traceability link: {ex.Message}"
        }

        member this.ListTraceabilityLinksAsync() = task {
            try
                let result =
                    traceabilityLinks.Values
                    |> Seq.sortByDescending (fun tl -> tl.CreatedAt)
                    |> List.ofSeq
                return Ok result
            with
            | ex -> return Error $"Failed to list traceability links: {ex.Message}"
        }

        member this.GetTraceabilityLinksByRequirementAsync(requirementId: string) = task {
            try
                let result =
                    traceabilityLinks.Values
                    |> Seq.filter (fun tl -> tl.RequirementId = requirementId)
                    |> Seq.sortByDescending (fun tl -> tl.CreatedAt)
                    |> List.ofSeq
                return Ok result
            with
            | ex -> return Error $"Failed to get traceability links by requirement: {ex.Message}"
        }

        member this.GetTraceabilityLinksByFileAsync(filePath: string) = task {
            try
                let result =
                    traceabilityLinks.Values
                    |> Seq.filter (fun tl -> tl.SourceFile = filePath)
                    |> Seq.sortByDescending (fun tl -> tl.CreatedAt)
                    |> List.ofSeq
                return Ok result
            with
            | ex -> return Error $"Failed to get traceability links by file: {ex.Message}"
        }

        // Analytics and reporting methods
        member this.GetRequirementStatisticsAsync() = task {
            try
                let allRequirements = requirements.Values |> List.ofSeq
                let total = allRequirements.Length

                let typeMap =
                    allRequirements
                    |> List.groupBy (fun r -> r.Type)
                    |> List.map (fun (t, reqs) -> (t, reqs.Length))
                    |> Map.ofList

                let statusMap =
                    allRequirements
                    |> List.groupBy (fun r -> r.Status)
                    |> List.map (fun (s, reqs) -> (s, reqs.Length))
                    |> Map.ofList

                let priorityMap =
                    allRequirements
                    |> List.groupBy (fun r -> r.Priority)
                    |> List.map (fun (p, reqs) -> (p, reqs.Length))
                    |> Map.ofList

                let completed = statusMap |> Map.tryFind RequirementStatus.Verified |> Option.defaultValue 0
                let completionRate = if total > 0 then (float completed / float total) * 100.0 else 0.0

                let now = DateTime.UtcNow
                let overdueCount =
                    allRequirements
                    |> List.filter (fun r ->
                        match r.TargetDate with
                        | Some targetDate ->
                            targetDate < now &&
                            r.Status <> RequirementStatus.Verified &&
                            r.Status <> RequirementStatus.Rejected &&
                            r.Status <> RequirementStatus.Obsolete
                        | None -> false)
                    |> List.length

                let startOfMonth = DateTime(now.Year, now.Month, 1)
                let createdThisMonth =
                    allRequirements
                    |> List.filter (fun r -> r.CreatedAt >= startOfMonth)
                    |> List.length

                let completedThisMonth =
                    allRequirements
                    |> List.filter (fun r -> r.Status = RequirementStatus.Verified && r.UpdatedAt >= startOfMonth)
                    |> List.length

                let statistics = {
                    TotalRequirements = total
                    RequirementsByType = typeMap
                    RequirementsByStatus = statusMap
                    RequirementsByPriority = priorityMap
                    AverageImplementationTime = None // TODO: Calculate from actual data
                    CompletionRate = completionRate
                    OverdueCount = overdueCount
                    CreatedThisMonth = createdThisMonth
                    CompletedThisMonth = completedThisMonth
                    GeneratedAt = DateTime.UtcNow
                }

                return Ok statistics
            with
            | ex -> return Error $"Failed to get requirement statistics: {ex.Message}"
        }

        member this.GetTestCoverageAsync(requirementId: string) = task {
            try
                let reqTestCases =
                    testCases.Values
                    |> Seq.filter (fun tc -> tc.RequirementId = requirementId)
                    |> List.ofSeq

                let total = reqTestCases.Length
                let passing = reqTestCases |> List.filter (fun tc -> tc.Status = TestStatus.Passed) |> List.length
                let failing = reqTestCases |> List.filter (fun tc -> tc.Status = TestStatus.Failed) |> List.length
                let notRun = reqTestCases |> List.filter (fun tc -> tc.Status = TestStatus.NotRun) |> List.length

                let coverage = if total > 0 then (float passing / float total) * 100.0 else 0.0

                let lastTestRun =
                    reqTestCases
                    |> List.choose (fun tc -> tc.LastExecuted)
                    |> List.sortDescending
                    |> List.tryHead

                let testTypes =
                    reqTestCases
                    |> List.groupBy (fun tc -> tc.TestType)
                    |> List.map (fun (tt, tcs) -> (tt, tcs.Length))
                    |> Map.ofList

                let testCoverage = {
                    RequirementId = requirementId
                    TotalTestCases = total
                    PassingTests = passing
                    FailingTests = failing
                    NotRunTests = notRun
                    CoveragePercentage = coverage
                    LastTestRun = lastTestRun
                    TestTypes = testTypes
                    GeneratedAt = DateTime.UtcNow
                }

                return Ok testCoverage
            with
            | ex -> return Error $"Failed to get test coverage: {ex.Message}"
        }

        member this.GetTraceabilityAnalysisAsync(requirementId: string) = task {
            try
                let links =
                    traceabilityLinks.Values
                    |> Seq.filter (fun tl -> tl.RequirementId = requirementId && tl.IsValid)
                    |> List.ofSeq

                let analysis = TraceabilityLinkHelpers.analyzeTraceability requirementId links
                return Ok analysis
            with
            | ex -> return Error $"Failed to get traceability analysis: {ex.Message}"
        }

        // Bulk operations
        member this.BulkCreateRequirementsAsync(requirementList: Requirement list) = task {
            try
                let ids = ResizeArray<string>()
                for requirement in requirementList do
                    if requirements.TryAdd(requirement.Id, requirement) then
                        ids.Add(requirement.Id)
                    else
                        return Error $"Requirement with ID {requirement.Id} already exists"

                return Ok (ids |> List.ofSeq)
            with
            | ex -> return Error $"Failed to bulk create requirements: {ex.Message}"
        }

        member this.BulkUpdateRequirementsAsync(requirementList: Requirement list) = task {
            try
                for requirement in requirementList do
                    if requirements.ContainsKey(requirement.Id) then
                        requirements.[requirement.Id] <- requirement
                    else
                        return Error $"Requirement with ID {requirement.Id} not found"

                return Ok ()
            with
            | ex -> return Error $"Failed to bulk update requirements: {ex.Message}"
        }

        member this.BulkDeleteRequirementsAsync(ids: string list) = task {
            try
                for id in ids do
                    let removed = requirements.TryRemove(id)
                    if not (fst removed) then
                        return Error $"Requirement with ID {id} not found"

                return Ok ()
            with
            | ex -> return Error $"Failed to bulk delete requirements: {ex.Message}"
        }

        // Database management (no-op for in-memory)
        member this.BackupDatabaseAsync(backupPath: string) = task {
            return Error "Backup not supported for in-memory repository"
        }

        member this.RestoreDatabaseAsync(backupPath: string) = task {
            return Error "Restore not supported for in-memory repository"
        }
