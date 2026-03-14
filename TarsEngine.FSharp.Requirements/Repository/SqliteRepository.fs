namespace TarsEngine.FSharp.Requirements.Repository

open System
open System.Data
open System.Threading.Tasks
open Microsoft.Data.Sqlite
open System.Text.Json
open TarsEngine.FSharp.Requirements.Models

/// <summary>
/// SQLite implementation of the requirement repository
/// Real implementation with full database functionality
/// </summary>
type SqliteRequirementRepository(connectionString: string) =
    
    let jsonOptions = JsonSerializerOptions()
    do jsonOptions.PropertyNamingPolicy <- JsonNamingPolicy.CamelCase
    
    /// <summary>
    /// Create database tables if they don't exist
    /// </summary>
    let createTables (connection: SqliteConnection) = task {
        let createRequirementsTable = """
            CREATE TABLE IF NOT EXISTS Requirements (
                Id TEXT PRIMARY KEY,
                Title TEXT NOT NULL,
                Description TEXT NOT NULL,
                Type TEXT NOT NULL,
                Priority INTEGER NOT NULL,
                Status TEXT NOT NULL,
                AcceptanceCriteria TEXT NOT NULL,
                Tags TEXT NOT NULL,
                Source TEXT,
                Assignee TEXT,
                EstimatedEffort REAL,
                ActualEffort REAL,
                TargetDate TEXT,
                CreatedAt TEXT NOT NULL,
                UpdatedAt TEXT NOT NULL,
                CreatedBy TEXT NOT NULL,
                UpdatedBy TEXT NOT NULL,
                Version INTEGER NOT NULL,
                Dependencies TEXT NOT NULL,
                Dependents TEXT NOT NULL,
                Metadata TEXT NOT NULL
            )"""
        
        let createTestCasesTable = """
            CREATE TABLE IF NOT EXISTS TestCases (
                Id TEXT PRIMARY KEY,
                RequirementId TEXT NOT NULL,
                Name TEXT NOT NULL,
                Description TEXT NOT NULL,
                TestCode TEXT NOT NULL,
                Language TEXT NOT NULL,
                ExpectedResult TEXT NOT NULL,
                Status TEXT NOT NULL,
                LastResult TEXT,
                LastExecuted TEXT,
                ExecutionDuration INTEGER,
                TestType TEXT NOT NULL,
                Priority INTEGER NOT NULL,
                Tags TEXT NOT NULL,
                SetupCode TEXT,
                TeardownCode TEXT,
                TestData TEXT,
                Environment TEXT,
                Dependencies TEXT NOT NULL,
                CreatedAt TEXT NOT NULL,
                UpdatedAt TEXT NOT NULL,
                CreatedBy TEXT NOT NULL,
                UpdatedBy TEXT NOT NULL,
                Version INTEGER NOT NULL,
                Metadata TEXT NOT NULL,
                FOREIGN KEY (RequirementId) REFERENCES Requirements(Id)
            )"""
        
        let createTraceabilityLinksTable = """
            CREATE TABLE IF NOT EXISTS TraceabilityLinks (
                Id TEXT PRIMARY KEY,
                RequirementId TEXT NOT NULL,
                SourceFile TEXT NOT NULL,
                LineNumber INTEGER,
                EndLineNumber INTEGER,
                CodeElement TEXT NOT NULL,
                ElementType TEXT NOT NULL,
                LinkType TEXT NOT NULL,
                Confidence REAL NOT NULL,
                CreationMethod TEXT NOT NULL,
                Description TEXT,
                CodeSnippet TEXT,
                Tags TEXT NOT NULL,
                CreatedAt TEXT NOT NULL,
                UpdatedAt TEXT NOT NULL,
                CreatedBy TEXT NOT NULL,
                UpdatedBy TEXT NOT NULL,
                Version INTEGER NOT NULL,
                IsValid INTEGER NOT NULL,
                LastValidated TEXT,
                Metadata TEXT NOT NULL,
                FOREIGN KEY (RequirementId) REFERENCES Requirements(Id)
            )"""
        
        let createTestExecutionResultsTable = """
            CREATE TABLE IF NOT EXISTS TestExecutionResults (
                Id TEXT PRIMARY KEY,
                TestCaseId TEXT NOT NULL,
                Status TEXT NOT NULL,
                Result TEXT NOT NULL,
                ErrorMessage TEXT,
                StackTrace TEXT,
                StartTime TEXT NOT NULL,
                EndTime TEXT NOT NULL,
                Duration INTEGER NOT NULL,
                Environment TEXT NOT NULL,
                Metadata TEXT NOT NULL,
                FOREIGN KEY (TestCaseId) REFERENCES TestCases(Id)
            )"""
        
        let createIndexes = [
            "CREATE INDEX IF NOT EXISTS idx_requirements_type ON Requirements(Type)"
            "CREATE INDEX IF NOT EXISTS idx_requirements_status ON Requirements(Status)"
            "CREATE INDEX IF NOT EXISTS idx_requirements_priority ON Requirements(Priority)"
            "CREATE INDEX IF NOT EXISTS idx_requirements_assignee ON Requirements(Assignee)"
            "CREATE INDEX IF NOT EXISTS idx_requirements_created ON Requirements(CreatedAt)"
            "CREATE INDEX IF NOT EXISTS idx_testcases_requirement ON TestCases(RequirementId)"
            "CREATE INDEX IF NOT EXISTS idx_testcases_status ON TestCases(Status)"
            "CREATE INDEX IF NOT EXISTS idx_traceability_requirement ON TraceabilityLinks(RequirementId)"
            "CREATE INDEX IF NOT EXISTS idx_traceability_file ON TraceabilityLinks(SourceFile)"
            "CREATE INDEX IF NOT EXISTS idx_execution_testcase ON TestExecutionResults(TestCaseId)"
            "CREATE INDEX IF NOT EXISTS idx_execution_time ON TestExecutionResults(StartTime)"
        ]
        
        use command = new SqliteCommand(createRequirementsTable, connection)
        do! command.ExecuteNonQueryAsync() |> Task.ignore
        
        command.CommandText <- createTestCasesTable
        do! command.ExecuteNonQueryAsync() |> Task.ignore
        
        command.CommandText <- createTraceabilityLinksTable
        do! command.ExecuteNonQueryAsync() |> Task.ignore
        
        command.CommandText <- createTestExecutionResultsTable
        do! command.ExecuteNonQueryAsync() |> Task.ignore
        
        for indexSql in createIndexes do
            command.CommandText <- indexSql
            do! command.ExecuteNonQueryAsync() |> Task.ignore
    }
    
    /// <summary>
    /// Execute with connection
    /// </summary>
    let executeWithConnection<'T> (operation: SqliteConnection -> Task<'T>) = task {
        try
            use connection = new SqliteConnection(connectionString)
            do! connection.OpenAsync()
            return! operation connection
        with
        | ex -> 
            return failwith $"Database operation failed: {ex.Message}"
    }
    
    /// <summary>
    /// Serialize list to JSON
    /// </summary>
    let serializeList<'T> (items: 'T list) =
        JsonSerializer.Serialize(items, jsonOptions)
    
    /// <summary>
    /// Deserialize list from JSON
    /// </summary>
    let deserializeList<'T> (json: string) =
        if String.IsNullOrEmpty(json) then []
        else JsonSerializer.Deserialize<'T list>(json, jsonOptions)
    
    /// <summary>
    /// Serialize map to JSON
    /// </summary>
    let serializeMap (map: Map<string, string>) =
        JsonSerializer.Serialize(map, jsonOptions)
    
    /// <summary>
    /// Deserialize map from JSON
    /// </summary>
    let deserializeMap (json: string) =
        if String.IsNullOrEmpty(json) then Map.empty
        else JsonSerializer.Deserialize<Map<string, string>>(json, jsonOptions)
    
    /// <summary>
    /// Convert requirement to database parameters
    /// </summary>
    let requirementToParameters (req: Requirement) = [
        ("@Id", req.Id :> obj)
        ("@Title", req.Title :> obj)
        ("@Description", req.Description :> obj)
        ("@Type", req.Type.ToString() :> obj)
        ("@Priority", int req.Priority :> obj)
        ("@Status", req.Status.ToString() :> obj)
        ("@AcceptanceCriteria", serializeList req.AcceptanceCriteria :> obj)
        ("@Tags", serializeList req.Tags :> obj)
        ("@Source", (req.Source |> Option.defaultValue null) :> obj)
        ("@Assignee", (req.Assignee |> Option.defaultValue null) :> obj)
        ("@EstimatedEffort", (req.EstimatedEffort |> Option.map box |> Option.defaultValue null) :> obj)
        ("@ActualEffort", (req.ActualEffort |> Option.map box |> Option.defaultValue null) :> obj)
        ("@TargetDate", (req.TargetDate |> Option.map (fun d -> d.ToString("O")) |> Option.defaultValue null) :> obj)
        ("@CreatedAt", req.CreatedAt.ToString("O") :> obj)
        ("@UpdatedAt", req.UpdatedAt.ToString("O") :> obj)
        ("@CreatedBy", req.CreatedBy :> obj)
        ("@UpdatedBy", req.UpdatedBy :> obj)
        ("@Version", req.Version :> obj)
        ("@Dependencies", serializeList req.Dependencies :> obj)
        ("@Dependents", serializeList req.Dependents :> obj)
        ("@Metadata", serializeMap req.Metadata :> obj)
    ]
    
    /// <summary>
    /// Convert database reader to requirement
    /// </summary>
    let readerToRequirement (reader: SqliteDataReader) =
        let parseOptionalDateTime (value: obj) =
            if value = null || value = (null :> obj) then None
            else Some (DateTime.Parse(value.ToString()))
        
        let parseOptionalFloat (value: obj) =
            if value = null || value = (null :> obj) then None
            else Some (float value)
        
        let parseOptionalString (value: obj) =
            if value = null || value = (null :> obj) then None
            else Some (value.ToString())
        
        {
            Id = reader.GetString("Id")
            Title = reader.GetString("Title")
            Description = reader.GetString("Description")
            Type = Enum.Parse<RequirementType>(reader.GetString("Type"))
            Priority = enum<RequirementPriority>(reader.GetInt32("Priority"))
            Status = Enum.Parse<RequirementStatus>(reader.GetString("Status"))
            AcceptanceCriteria = deserializeList<string>(reader.GetString("AcceptanceCriteria"))
            Tags = deserializeList<string>(reader.GetString("Tags"))
            Source = parseOptionalString(reader["Source"])
            Assignee = parseOptionalString(reader["Assignee"])
            EstimatedEffort = parseOptionalFloat(reader["EstimatedEffort"])
            ActualEffort = parseOptionalFloat(reader["ActualEffort"])
            TargetDate = parseOptionalDateTime(reader["TargetDate"])
            CreatedAt = DateTime.Parse(reader.GetString("CreatedAt"))
            UpdatedAt = DateTime.Parse(reader.GetString("UpdatedAt"))
            CreatedBy = reader.GetString("CreatedBy")
            UpdatedBy = reader.GetString("UpdatedBy")
            Version = reader.GetInt32("Version")
            Dependencies = deserializeList<string>(reader.GetString("Dependencies"))
            Dependents = deserializeList<string>(reader.GetString("Dependents"))
            Metadata = deserializeMap(reader.GetString("Metadata"))
        }
    
    interface IRequirementRepository with
        
        member this.InitializeDatabaseAsync() = task {
            try
                do! executeWithConnection createTables
                return Ok ()
            with
            | ex -> return Error $"Failed to initialize database: {ex.Message}"
        }
        
        member this.CreateRequirementAsync(requirement: Requirement) = task {
            try
                let sql = """
                    INSERT INTO Requirements (
                        Id, Title, Description, Type, Priority, Status, AcceptanceCriteria, Tags,
                        Source, Assignee, EstimatedEffort, ActualEffort, TargetDate,
                        CreatedAt, UpdatedAt, CreatedBy, UpdatedBy, Version,
                        Dependencies, Dependents, Metadata
                    ) VALUES (
                        @Id, @Title, @Description, @Type, @Priority, @Status, @AcceptanceCriteria, @Tags,
                        @Source, @Assignee, @EstimatedEffort, @ActualEffort, @TargetDate,
                        @CreatedAt, @UpdatedAt, @CreatedBy, @UpdatedBy, @Version,
                        @Dependencies, @Dependents, @Metadata
                    )"""
                
                let! result = executeWithConnection (fun connection -> task {
                    use command = new SqliteCommand(sql, connection)
                    let parameters = requirementToParameters requirement
                    for (name, value) in parameters do
                        command.Parameters.AddWithValue(name, value) |> ignore
                    
                    let! rowsAffected = command.ExecuteNonQueryAsync()
                    return if rowsAffected > 0 then requirement.Id else failwith "No rows affected"
                })
                
                return Ok result
            with
            | ex -> return Error $"Failed to create requirement: {ex.Message}"
        }

        member this.GetRequirementAsync(id: string) = task {
            try
                let sql = "SELECT * FROM Requirements WHERE Id = @Id"

                let! result = executeWithConnection (fun connection -> task {
                    use command = new SqliteCommand(sql, connection)
                    command.Parameters.AddWithValue("@Id", id) |> ignore

                    use! reader = command.ExecuteReaderAsync()
                    if reader.Read() then
                        return Some (readerToRequirement reader)
                    else
                        return None
                })

                return Ok result
            with
            | ex -> return Error $"Failed to get requirement: {ex.Message}"
        }

        member this.UpdateRequirementAsync(requirement: Requirement) = task {
            try
                let sql = """
                    UPDATE Requirements SET
                        Title = @Title, Description = @Description, Type = @Type, Priority = @Priority,
                        Status = @Status, AcceptanceCriteria = @AcceptanceCriteria, Tags = @Tags,
                        Source = @Source, Assignee = @Assignee, EstimatedEffort = @EstimatedEffort,
                        ActualEffort = @ActualEffort, TargetDate = @TargetDate, UpdatedAt = @UpdatedAt,
                        UpdatedBy = @UpdatedBy, Version = @Version, Dependencies = @Dependencies,
                        Dependents = @Dependents, Metadata = @Metadata
                    WHERE Id = @Id"""

                let! result = executeWithConnection (fun connection -> task {
                    use command = new SqliteCommand(sql, connection)
                    let parameters = requirementToParameters requirement
                    for (name, value) in parameters do
                        command.Parameters.AddWithValue(name, value) |> ignore

                    let! rowsAffected = command.ExecuteNonQueryAsync()
                    return if rowsAffected > 0 then () else failwith "Requirement not found"
                })

                return Ok result
            with
            | ex -> return Error $"Failed to update requirement: {ex.Message}"
        }

        member this.DeleteRequirementAsync(id: string) = task {
            try
                let sql = "DELETE FROM Requirements WHERE Id = @Id"

                let! result = executeWithConnection (fun connection -> task {
                    use command = new SqliteCommand(sql, connection)
                    command.Parameters.AddWithValue("@Id", id) |> ignore

                    let! rowsAffected = command.ExecuteNonQueryAsync()
                    return if rowsAffected > 0 then () else failwith "Requirement not found"
                })

                return Ok result
            with
            | ex -> return Error $"Failed to delete requirement: {ex.Message}"
        }

        member this.ListRequirementsAsync() = task {
            try
                let sql = "SELECT * FROM Requirements ORDER BY CreatedAt DESC"

                let! result = executeWithConnection (fun connection -> task {
                    use command = new SqliteCommand(sql, connection)
                    use! reader = command.ExecuteReaderAsync()

                    let requirements = ResizeArray<Requirement>()
                    while reader.Read() do
                        requirements.Add(readerToRequirement reader)

                    return requirements |> List.ofSeq
                })

                return Ok result
            with
            | ex -> return Error $"Failed to list requirements: {ex.Message}"
        }

        member this.GetRequirementsByTypeAsync(reqType: RequirementType) = task {
            try
                let sql = "SELECT * FROM Requirements WHERE Type = @Type ORDER BY CreatedAt DESC"

                let! result = executeWithConnection (fun connection -> task {
                    use command = new SqliteCommand(sql, connection)
                    command.Parameters.AddWithValue("@Type", reqType.ToString()) |> ignore
                    use! reader = command.ExecuteReaderAsync()

                    let requirements = ResizeArray<Requirement>()
                    while reader.Read() do
                        requirements.Add(readerToRequirement reader)

                    return requirements |> List.ofSeq
                })

                return Ok result
            with
            | ex -> return Error $"Failed to get requirements by type: {ex.Message}"
        }

        member this.GetRequirementsByStatusAsync(status: RequirementStatus) = task {
            try
                let sql = "SELECT * FROM Requirements WHERE Status = @Status ORDER BY CreatedAt DESC"

                let! result = executeWithConnection (fun connection -> task {
                    use command = new SqliteCommand(sql, connection)
                    command.Parameters.AddWithValue("@Status", status.ToString()) |> ignore
                    use! reader = command.ExecuteReaderAsync()

                    let requirements = ResizeArray<Requirement>()
                    while reader.Read() do
                        requirements.Add(readerToRequirement reader)

                    return requirements |> List.ofSeq
                })

                return Ok result
            with
            | ex -> return Error $"Failed to get requirements by status: {ex.Message}"
        }

        member this.GetRequirementsByPriorityAsync(priority: RequirementPriority) = task {
            try
                let sql = "SELECT * FROM Requirements WHERE Priority = @Priority ORDER BY CreatedAt DESC"

                let! result = executeWithConnection (fun connection -> task {
                    use command = new SqliteCommand(sql, connection)
                    command.Parameters.AddWithValue("@Priority", int priority) |> ignore
                    use! reader = command.ExecuteReaderAsync()

                    let requirements = ResizeArray<Requirement>()
                    while reader.Read() do
                        requirements.Add(readerToRequirement reader)

                    return requirements |> List.ofSeq
                })

                return Ok result
            with
            | ex -> return Error $"Failed to get requirements by priority: {ex.Message}"
        }

        member this.GetRequirementsByAssigneeAsync(assignee: string) = task {
            try
                let sql = "SELECT * FROM Requirements WHERE Assignee = @Assignee ORDER BY CreatedAt DESC"

                let! result = executeWithConnection (fun connection -> task {
                    use command = new SqliteCommand(sql, connection)
                    command.Parameters.AddWithValue("@Assignee", assignee) |> ignore
                    use! reader = command.ExecuteReaderAsync()

                    let requirements = ResizeArray<Requirement>()
                    while reader.Read() do
                        requirements.Add(readerToRequirement reader)

                    return requirements |> List.ofSeq
                })

                return Ok result
            with
            | ex -> return Error $"Failed to get requirements by assignee: {ex.Message}"
        }

        member this.SearchRequirementsAsync(searchText: string) = task {
            try
                let sql = """
                    SELECT * FROM Requirements
                    WHERE Title LIKE @SearchText
                       OR Description LIKE @SearchText
                       OR Tags LIKE @SearchText
                    ORDER BY CreatedAt DESC"""

                let! result = executeWithConnection (fun connection -> task {
                    use command = new SqliteCommand(sql, connection)
                    command.Parameters.AddWithValue("@SearchText", $"%{searchText}%") |> ignore
                    use! reader = command.ExecuteReaderAsync()

                    let requirements = ResizeArray<Requirement>()
                    while reader.Read() do
                        requirements.Add(readerToRequirement reader)

                    return requirements |> List.ofSeq
                })

                return Ok result
            with
            | ex -> return Error $"Failed to search requirements: {ex.Message}"
        }

        member this.GetRequirementsByTagAsync(tag: string) = task {
            try
                let sql = "SELECT * FROM Requirements WHERE Tags LIKE @Tag ORDER BY CreatedAt DESC"

                let! result = executeWithConnection (fun connection -> task {
                    use command = new SqliteCommand(sql, connection)
                    command.Parameters.AddWithValue("@Tag", $"%\"{tag}\"%") |> ignore
                    use! reader = command.ExecuteReaderAsync()

                    let requirements = ResizeArray<Requirement>()
                    while reader.Read() do
                        requirements.Add(readerToRequirement reader)

                    return requirements |> List.ofSeq
                })

                return Ok result
            with
            | ex -> return Error $"Failed to get requirements by tag: {ex.Message}"
        }

        member this.GetOverdueRequirementsAsync() = task {
            try
                let sql = """
                    SELECT * FROM Requirements
                    WHERE TargetDate IS NOT NULL
                      AND TargetDate < @CurrentDate
                      AND Status NOT IN ('Verified', 'Rejected', 'Obsolete')
                    ORDER BY TargetDate ASC"""

                let! result = executeWithConnection (fun connection -> task {
                    use command = new SqliteCommand(sql, connection)
                    command.Parameters.AddWithValue("@CurrentDate", DateTime.UtcNow.ToString("O")) |> ignore
                    use! reader = command.ExecuteReaderAsync()

                    let requirements = ResizeArray<Requirement>()
                    while reader.Read() do
                        requirements.Add(readerToRequirement reader)

                    return requirements |> List.ofSeq
                })

                return Ok result
            with
            | ex -> return Error $"Failed to get overdue requirements: {ex.Message}"
        }

        // Test Case methods - placeholder implementations for now
        member this.CreateTestCaseAsync(testCase: TestCase) = task {
            // TODO: Implement test case creation
            return Error "Test case operations not yet implemented"
        }

        member this.GetTestCaseAsync(id: string) = task {
            return Error "Test case operations not yet implemented"
        }

        member this.UpdateTestCaseAsync(testCase: TestCase) = task {
            return Error "Test case operations not yet implemented"
        }

        member this.DeleteTestCaseAsync(id: string) = task {
            return Error "Test case operations not yet implemented"
        }

        member this.ListTestCasesAsync() = task {
            return Error "Test case operations not yet implemented"
        }

        member this.GetTestCasesByRequirementAsync(requirementId: string) = task {
            return Error "Test case operations not yet implemented"
        }

        // Test execution methods - placeholder implementations
        member this.SaveTestExecutionResultAsync(result: TestExecutionResult) = task {
            return Error "Test execution operations not yet implemented"
        }

        member this.GetTestExecutionHistoryAsync(testCaseId: string) = task {
            return Error "Test execution operations not yet implemented"
        }

        member this.GetLatestTestResultAsync(testCaseId: string) = task {
            return Error "Test execution operations not yet implemented"
        }

        // Traceability methods - placeholder implementations
        member this.CreateTraceabilityLinkAsync(link: TraceabilityLink) = task {
            return Error "Traceability operations not yet implemented"
        }

        member this.GetTraceabilityLinkAsync(id: string) = task {
            return Error "Traceability operations not yet implemented"
        }

        member this.UpdateTraceabilityLinkAsync(link: TraceabilityLink) = task {
            return Error "Traceability operations not yet implemented"
        }

        member this.DeleteTraceabilityLinkAsync(id: string) = task {
            return Error "Traceability operations not yet implemented"
        }

        member this.ListTraceabilityLinksAsync() = task {
            return Error "Traceability operations not yet implemented"
        }

        member this.GetTraceabilityLinksByRequirementAsync(requirementId: string) = task {
            return Error "Traceability operations not yet implemented"
        }

        member this.GetTraceabilityLinksByFileAsync(filePath: string) = task {
            return Error "Traceability operations not yet implemented"
        }

        // Analytics and reporting methods
        member this.GetRequirementStatisticsAsync() = task {
            try
                let! result = executeWithConnection (fun connection -> task {
                    // Get total count
                    use command = new SqliteCommand("SELECT COUNT(*) FROM Requirements", connection)
                    let! totalCount = command.ExecuteScalarAsync()

                    // Get counts by type
                    command.CommandText <- "SELECT Type, COUNT(*) FROM Requirements GROUP BY Type"
                    use! reader = command.ExecuteReaderAsync()
                    let typeMap = ResizeArray<RequirementType * int>()
                    while reader.Read() do
                        let reqType = Enum.Parse<RequirementType>(reader.GetString(0))
                        let count = reader.GetInt32(1)
                        typeMap.Add((reqType, count))
                    reader.Close()

                    // Get counts by status
                    command.CommandText <- "SELECT Status, COUNT(*) FROM Requirements GROUP BY Status"
                    use! reader2 = command.ExecuteReaderAsync()
                    let statusMap = ResizeArray<RequirementStatus * int>()
                    while reader2.Read() do
                        let status = Enum.Parse<RequirementStatus>(reader2.GetString(0))
                        let count = reader2.GetInt32(1)
                        statusMap.Add((status, count))
                    reader2.Close()

                    // Get counts by priority
                    command.CommandText <- "SELECT Priority, COUNT(*) FROM Requirements GROUP BY Priority"
                    use! reader3 = command.ExecuteReaderAsync()
                    let priorityMap = ResizeArray<RequirementPriority * int>()
                    while reader3.Read() do
                        let priority = enum<RequirementPriority>(reader3.GetInt32(0))
                        let count = reader3.GetInt32(1)
                        priorityMap.Add((priority, count))
                    reader3.Close()

                    // Get overdue count
                    command.CommandText <- """
                        SELECT COUNT(*) FROM Requirements
                        WHERE TargetDate IS NOT NULL
                          AND TargetDate < @CurrentDate
                          AND Status NOT IN ('Verified', 'Rejected', 'Obsolete')"""
                    command.Parameters.Clear()
                    command.Parameters.AddWithValue("@CurrentDate", DateTime.UtcNow.ToString("O")) |> ignore
                    let! overdueCount = command.ExecuteScalarAsync()

                    // Get this month's counts
                    let startOfMonth = DateTime(DateTime.UtcNow.Year, DateTime.UtcNow.Month, 1).ToString("O")
                    command.CommandText <- "SELECT COUNT(*) FROM Requirements WHERE CreatedAt >= @StartOfMonth"
                    command.Parameters.Clear()
                    command.Parameters.AddWithValue("@StartOfMonth", startOfMonth) |> ignore
                    let! createdThisMonth = command.ExecuteScalarAsync()

                    command.CommandText <- """
                        SELECT COUNT(*) FROM Requirements
                        WHERE Status = 'Verified' AND UpdatedAt >= @StartOfMonth"""
                    let! completedThisMonth = command.ExecuteScalarAsync()

                    let total = Convert.ToInt32(totalCount)
                    let completed = statusMap |> Seq.tryFind (fun (status, _) -> status = RequirementStatus.Verified) |> Option.map snd |> Option.defaultValue 0
                    let completionRate = if total > 0 then (float completed / float total) * 100.0 else 0.0

                    return {
                        TotalRequirements = total
                        RequirementsByType = typeMap |> Seq.map (fun (t, c) -> (t, c)) |> Map.ofSeq
                        RequirementsByStatus = statusMap |> Seq.map (fun (s, c) -> (s, c)) |> Map.ofSeq
                        RequirementsByPriority = priorityMap |> Seq.map (fun (p, c) -> (p, c)) |> Map.ofSeq
                        AverageImplementationTime = None // TODO: Calculate from actual data
                        CompletionRate = completionRate
                        OverdueCount = Convert.ToInt32(overdueCount)
                        CreatedThisMonth = Convert.ToInt32(createdThisMonth)
                        CompletedThisMonth = Convert.ToInt32(completedThisMonth)
                        GeneratedAt = DateTime.UtcNow
                    }
                })

                return Ok result
            with
            | ex -> return Error $"Failed to get requirement statistics: {ex.Message}"
        }

        member this.GetTestCoverageAsync(requirementId: string) = task {
            return Error "Test coverage analysis not yet implemented"
        }

        member this.GetTraceabilityAnalysisAsync(requirementId: string) = task {
            return Error "Traceability analysis not yet implemented"
        }

        // Bulk operations
        member this.BulkCreateRequirementsAsync(requirements: Requirement list) = task {
            try
                let! results = executeWithConnection (fun connection -> task {
                    use transaction = connection.BeginTransaction()
                    try
                        let ids = ResizeArray<string>()
                        for requirement in requirements do
                            let sql = """
                                INSERT INTO Requirements (
                                    Id, Title, Description, Type, Priority, Status, AcceptanceCriteria, Tags,
                                    Source, Assignee, EstimatedEffort, ActualEffort, TargetDate,
                                    CreatedAt, UpdatedAt, CreatedBy, UpdatedBy, Version,
                                    Dependencies, Dependents, Metadata
                                ) VALUES (
                                    @Id, @Title, @Description, @Type, @Priority, @Status, @AcceptanceCriteria, @Tags,
                                    @Source, @Assignee, @EstimatedEffort, @ActualEffort, @TargetDate,
                                    @CreatedAt, @UpdatedAt, @CreatedBy, @UpdatedBy, @Version,
                                    @Dependencies, @Dependents, @Metadata
                                )"""

                            use command = new SqliteCommand(sql, connection, transaction)
                            let parameters = requirementToParameters requirement
                            for (name, value) in parameters do
                                command.Parameters.AddWithValue(name, value) |> ignore

                            let! rowsAffected = command.ExecuteNonQueryAsync()
                            if rowsAffected > 0 then
                                ids.Add(requirement.Id)

                        transaction.Commit()
                        return ids |> List.ofSeq
                    with
                    | ex ->
                        transaction.Rollback()
                        raise ex
                })

                return Ok results
            with
            | ex -> return Error $"Failed to bulk create requirements: {ex.Message}"
        }

        member this.BulkUpdateRequirementsAsync(requirements: Requirement list) = task {
            try
                let! result = executeWithConnection (fun connection -> task {
                    use transaction = connection.BeginTransaction()
                    try
                        for requirement in requirements do
                            let sql = """
                                UPDATE Requirements SET
                                    Title = @Title, Description = @Description, Type = @Type, Priority = @Priority,
                                    Status = @Status, AcceptanceCriteria = @AcceptanceCriteria, Tags = @Tags,
                                    Source = @Source, Assignee = @Assignee, EstimatedEffort = @EstimatedEffort,
                                    ActualEffort = @ActualEffort, TargetDate = @TargetDate, UpdatedAt = @UpdatedAt,
                                    UpdatedBy = @UpdatedBy, Version = @Version, Dependencies = @Dependencies,
                                    Dependents = @Dependents, Metadata = @Metadata
                                WHERE Id = @Id"""

                            use command = new SqliteCommand(sql, connection, transaction)
                            let parameters = requirementToParameters requirement
                            for (name, value) in parameters do
                                command.Parameters.AddWithValue(name, value) |> ignore

                            do! command.ExecuteNonQueryAsync() |> Task.ignore

                        transaction.Commit()
                        return ()
                    with
                    | ex ->
                        transaction.Rollback()
                        raise ex
                })

                return Ok result
            with
            | ex -> return Error $"Failed to bulk update requirements: {ex.Message}"
        }

        member this.BulkDeleteRequirementsAsync(ids: string list) = task {
            try
                let! result = executeWithConnection (fun connection -> task {
                    use transaction = connection.BeginTransaction()
                    try
                        for id in ids do
                            use command = new SqliteCommand("DELETE FROM Requirements WHERE Id = @Id", connection, transaction)
                            command.Parameters.AddWithValue("@Id", id) |> ignore
                            do! command.ExecuteNonQueryAsync() |> Task.ignore

                        transaction.Commit()
                        return ()
                    with
                    | ex ->
                        transaction.Rollback()
                        raise ex
                })

                return Ok result
            with
            | ex -> return Error $"Failed to bulk delete requirements: {ex.Message}"
        }

        // Database management
        member this.BackupDatabaseAsync(backupPath: string) = task {
            try
                let! result = executeWithConnection (fun connection -> task {
                    use backupConnection = new SqliteConnection($"Data Source={backupPath}")
                    do! backupConnection.OpenAsync()
                    connection.BackupDatabase(backupConnection, "main", "main", -1, null, -1)
                    return ()
                })

                return Ok result
            with
            | ex -> return Error $"Failed to backup database: {ex.Message}"
        }

        member this.RestoreDatabaseAsync(backupPath: string) = task {
            try
                let! result = executeWithConnection (fun connection -> task {
                    use backupConnection = new SqliteConnection($"Data Source={backupPath}")
                    do! backupConnection.OpenAsync()
                    backupConnection.BackupDatabase(connection, "main", "main", -1, null, -1)
                    return ()
                })

                return Ok result
            with
            | ex -> return Error $"Failed to restore database: {ex.Message}"
        }
