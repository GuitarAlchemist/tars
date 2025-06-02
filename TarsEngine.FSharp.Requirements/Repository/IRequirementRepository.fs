namespace TarsEngine.FSharp.Requirements.Repository

open System
open System.Threading.Tasks
open TarsEngine.FSharp.Requirements.Models

/// <summary>
/// Repository interface for requirement management
/// Real implementation - no fake or placeholder methods
/// </summary>
type IRequirementRepository =
    
    // Requirement CRUD operations
    abstract member CreateRequirementAsync: Requirement -> Task<Result<string, string>>
    abstract member GetRequirementAsync: string -> Task<Result<Requirement option, string>>
    abstract member UpdateRequirementAsync: Requirement -> Task<Result<unit, string>>
    abstract member DeleteRequirementAsync: string -> Task<Result<unit, string>>
    abstract member ListRequirementsAsync: unit -> Task<Result<Requirement list, string>>
    
    // Requirement queries
    abstract member GetRequirementsByTypeAsync: RequirementType -> Task<Result<Requirement list, string>>
    abstract member GetRequirementsByStatusAsync: RequirementStatus -> Task<Result<Requirement list, string>>
    abstract member GetRequirementsByPriorityAsync: RequirementPriority -> Task<Result<Requirement list, string>>
    abstract member GetRequirementsByAssigneeAsync: string -> Task<Result<Requirement list, string>>
    abstract member SearchRequirementsAsync: string -> Task<Result<Requirement list, string>>
    abstract member GetRequirementsByTagAsync: string -> Task<Result<Requirement list, string>>
    abstract member GetOverdueRequirementsAsync: unit -> Task<Result<Requirement list, string>>
    
    // Test case CRUD operations
    abstract member CreateTestCaseAsync: TestCase -> Task<Result<string, string>>
    abstract member GetTestCaseAsync: string -> Task<Result<TestCase option, string>>
    abstract member UpdateTestCaseAsync: TestCase -> Task<Result<unit, string>>
    abstract member DeleteTestCaseAsync: string -> Task<Result<unit, string>>
    abstract member ListTestCasesAsync: unit -> Task<Result<TestCase list, string>>
    abstract member GetTestCasesByRequirementAsync: string -> Task<Result<TestCase list, string>>
    
    // Test execution
    abstract member SaveTestExecutionResultAsync: TestExecutionResult -> Task<Result<unit, string>>
    abstract member GetTestExecutionHistoryAsync: string -> Task<Result<TestExecutionResult list, string>>
    abstract member GetLatestTestResultAsync: string -> Task<Result<TestExecutionResult option, string>>
    
    // Traceability link CRUD operations
    abstract member CreateTraceabilityLinkAsync: TraceabilityLink -> Task<Result<string, string>>
    abstract member GetTraceabilityLinkAsync: string -> Task<Result<TraceabilityLink option, string>>
    abstract member UpdateTraceabilityLinkAsync: TraceabilityLink -> Task<Result<unit, string>>
    abstract member DeleteTraceabilityLinkAsync: string -> Task<Result<unit, string>>
    abstract member ListTraceabilityLinksAsync: unit -> Task<Result<TraceabilityLink list, string>>
    abstract member GetTraceabilityLinksByRequirementAsync: string -> Task<Result<TraceabilityLink list, string>>
    abstract member GetTraceabilityLinksByFileAsync: string -> Task<Result<TraceabilityLink list, string>>
    
    // Analytics and reporting
    abstract member GetRequirementStatisticsAsync: unit -> Task<Result<RequirementStatistics, string>>
    abstract member GetTestCoverageAsync: string -> Task<Result<TestCoverage, string>>
    abstract member GetTraceabilityAnalysisAsync: string -> Task<Result<TraceabilityAnalysis, string>>
    
    // Bulk operations
    abstract member BulkCreateRequirementsAsync: Requirement list -> Task<Result<string list, string>>
    abstract member BulkUpdateRequirementsAsync: Requirement list -> Task<Result<unit, string>>
    abstract member BulkDeleteRequirementsAsync: string list -> Task<Result<unit, string>>
    
    // Database management
    abstract member InitializeDatabaseAsync: unit -> Task<Result<unit, string>>
    abstract member BackupDatabaseAsync: string -> Task<Result<unit, string>>
    abstract member RestoreDatabaseAsync: string -> Task<Result<unit, string>>

/// <summary>
/// Requirement statistics for reporting
/// </summary>
and [<CLIMutable>] RequirementStatistics = {
    TotalRequirements: int
    RequirementsByType: Map<RequirementType, int>
    RequirementsByStatus: Map<RequirementStatus, int>
    RequirementsByPriority: Map<RequirementPriority, int>
    AverageImplementationTime: float option
    CompletionRate: float
    OverdueCount: int
    CreatedThisMonth: int
    CompletedThisMonth: int
    GeneratedAt: DateTime
}

/// <summary>
/// Test coverage information
/// </summary>
and [<CLIMutable>] TestCoverage = {
    RequirementId: string
    TotalTestCases: int
    PassingTests: int
    FailingTests: int
    NotRunTests: int
    CoveragePercentage: float
    LastTestRun: DateTime option
    TestTypes: Map<string, int>
    GeneratedAt: DateTime
}

/// <summary>
/// Query parameters for requirement searches
/// </summary>
[<CLIMutable>]
type RequirementQuery = {
    Types: RequirementType list option
    Statuses: RequirementStatus list option
    Priorities: RequirementPriority list option
    Tags: string list option
    Assignees: string list option
    CreatedAfter: DateTime option
    CreatedBefore: DateTime option
    UpdatedAfter: DateTime option
    UpdatedBefore: DateTime option
    SearchText: string option
    IncludeObsolete: bool
    SortBy: string option
    SortDirection: string option
    Skip: int option
    Take: int option
}

/// <summary>
/// Paged result for large queries
/// </summary>
[<CLIMutable>]
type PagedResult<'T> = {
    Items: 'T list
    TotalCount: int
    PageSize: int
    PageNumber: int
    HasNextPage: bool
    HasPreviousPage: bool
}

module RequirementQueryHelpers =
    
    /// <summary>
    /// Create empty query
    /// </summary>
    let empty = {
        Types = None
        Statuses = None
        Priorities = None
        Tags = None
        Assignees = None
        CreatedAfter = None
        CreatedBefore = None
        UpdatedAfter = None
        UpdatedBefore = None
        SearchText = None
        IncludeObsolete = false
        SortBy = None
        SortDirection = None
        Skip = None
        Take = None
    }
    
    /// <summary>
    /// Create query for specific type
    /// </summary>
    let forType (reqType: RequirementType) =
        { empty with Types = Some [reqType] }
    
    /// <summary>
    /// Create query for specific status
    /// </summary>
    let forStatus (status: RequirementStatus) =
        { empty with Statuses = Some [status] }
    
    /// <summary>
    /// Create query for specific assignee
    /// </summary>
    let forAssignee (assignee: string) =
        { empty with Assignees = Some [assignee] }
    
    /// <summary>
    /// Create query for text search
    /// </summary>
    let forText (searchText: string) =
        { empty with SearchText = Some searchText }
    
    /// <summary>
    /// Add pagination to query
    /// </summary>
    let withPagination (pageNumber: int) (pageSize: int) (query: RequirementQuery) =
        { query with 
            Skip = Some ((pageNumber - 1) * pageSize)
            Take = Some pageSize }
    
    /// <summary>
    /// Add sorting to query
    /// </summary>
    let withSorting (sortBy: string) (direction: string) (query: RequirementQuery) =
        { query with 
            SortBy = Some sortBy
            SortDirection = Some direction }

module PagedResultHelpers =
    
    /// <summary>
    /// Create paged result
    /// </summary>
    let create<'T> (items: 'T list) (totalCount: int) (pageNumber: int) (pageSize: int) =
        {
            Items = items
            TotalCount = totalCount
            PageSize = pageSize
            PageNumber = pageNumber
            HasNextPage = (pageNumber * pageSize) < totalCount
            HasPreviousPage = pageNumber > 1
        }
    
    /// <summary>
    /// Create empty paged result
    /// </summary>
    let empty<'T> = {
        Items = []
        TotalCount = 0
        PageSize = 0
        PageNumber = 1
        HasNextPage = false
        HasPreviousPage = false
    }
