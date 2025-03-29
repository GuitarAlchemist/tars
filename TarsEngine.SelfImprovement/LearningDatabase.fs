namespace TarsEngine.SelfImprovement

open System
open System.IO
open System.Text.Json
open System.Text.Json.Serialization
open FSharp.SystemTextJson
open System.Collections.Generic
open System.Threading.Tasks

/// <summary>
/// Represents a learning event in the improvement history
/// </summary>
type LearningEvent =
    { Id: string
      Timestamp: DateTime
      EventType: string
      FileName: string
      FileType: string
      OriginalContent: string option
      ModifiedContent: string option
      AnalysisResult: obj option
      ImprovementProposal: obj option
      Feedback: string option
      Success: bool
      Tags: string list
      Metadata: Dictionary<string, string> }

/// <summary>
/// Represents a collection of learning events
/// </summary>
type LearningDatabase =
    { Events: LearningEvent list
      Statistics: Dictionary<string, int>
      LastUpdated: DateTime }

/// <summary>
/// Functions for working with the learning database
/// </summary>
module LearningDatabase =
    let private databasePath =
        let appDataPath =
            match Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData) with
            | "" -> Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".tars")
            | path -> Path.Combine(path, "TARS")

        if not (Directory.Exists(appDataPath)) then
            Directory.CreateDirectory(appDataPath) |> ignore

        Path.Combine(appDataPath, "learning.json")

    /// <summary>
    /// Creates a new learning event
    /// </summary>
    let createEvent (eventType: string) (fileName: string) (fileType: string) (success: bool) =
        { Id = Guid.NewGuid().ToString()
          Timestamp = DateTime.UtcNow
          EventType = eventType
          FileName = fileName
          FileType = fileType
          OriginalContent = None
          ModifiedContent = None
          AnalysisResult = None
          ImprovementProposal = None
          Feedback = None
          Success = success
          Tags = []
          Metadata = new Dictionary<string, string>() }

    /// <summary>
    /// Creates a new learning database
    /// </summary>
    let createDatabase () =
        { Events = []
          Statistics = new Dictionary<string, int>()
          LastUpdated = DateTime.UtcNow }

    /// <summary>
    /// Loads the learning database from disk
    /// </summary>
    let loadDatabase () =
        async {
            try
                if File.Exists(databasePath) then
                    let! json = File.ReadAllTextAsync(databasePath) |> Async.AwaitTask
                    let options = JsonSerializerOptions()
                    options.Converters.Add(JsonFSharpConverter())
                    return JsonSerializer.Deserialize<LearningDatabase>(json, options)
                else
                    return createDatabase()
            with ex ->
                printfn "Error loading learning database: %s" ex.Message
                return createDatabase()
        }

    /// <summary>
    /// Saves the learning database to disk
    /// </summary>
    let saveDatabase (database: LearningDatabase) =
        async {
            try
                let options = JsonSerializerOptions()
                options.WriteIndented <- true
                options.Converters.Add(JsonFSharpConverter())
                let json = JsonSerializer.Serialize(database, options)
                do! File.WriteAllTextAsync(databasePath, json) |> Async.AwaitTask
                return true
            with ex ->
                printfn "Error saving learning database: %s" ex.Message
                return false
        }

    /// <summary>
    /// Updates the statistics in the learning database
    /// </summary>
    let updateStatistics (database: LearningDatabase) =
        let statistics = new Dictionary<string, int>()

        // Count total events
        statistics.["Total"] <- database.Events.Length

        // Count events by type
        database.Events
        |> List.groupBy (fun e -> e.EventType)
        |> List.iter (fun (eventType, events) ->
            statistics.[sprintf "EventType:%s" eventType] <- events.Length)

        // Count events by file type
        database.Events
        |> List.groupBy (fun e -> e.FileType)
        |> List.iter (fun (fileType, events) ->
            statistics.[sprintf "FileType:%s" fileType] <- events.Length)

        // Count successful vs. failed events
        let (successful, failed) =
            database.Events
            |> List.partition (fun e -> e.Success)

        statistics.["Successful"] <- successful.Length
        statistics.["Failed"] <- failed.Length

        // Count events by tag
        database.Events
        |> List.collect (fun e -> e.Tags)
        |> List.groupBy id
        |> List.iter (fun (tag, occurrences) ->
            statistics.[sprintf "Tag:%s" tag] <- occurrences.Length)

        { database with
            Statistics = statistics
            LastUpdated = DateTime.UtcNow }

    /// <summary>
    /// Adds an event to the learning database
    /// </summary>
    let addEvent (database: LearningDatabase) (event: LearningEvent) =
        async {
            let updatedDatabase =
                { database with
                    Events = database.Events @ [event]
                    LastUpdated = DateTime.UtcNow }

            let databaseWithStats = updateStatistics updatedDatabase
            let! success = saveDatabase databaseWithStats

            return
                if success then
                    databaseWithStats
                else
                    updatedDatabase
        }

    /// <summary>
    /// Creates an analysis event and adds it to the database
    /// </summary>
    let recordAnalysis (fileName: string) (fileType: string) (originalContent: string) (result: obj) =
        async {
            let! database = loadDatabase()

            let event =
                createEvent "Analysis" fileName fileType true
                |> fun e -> { e with
                                OriginalContent = Some originalContent
                                AnalysisResult = Some result }

            return! addEvent database event
        }

    /// <summary>
    /// Creates an improvement proposal event and adds it to the database
    /// </summary>
    let recordImprovement (fileName: string) (fileType: string) (proposal: obj) =
        async {
            let! database = loadDatabase()

            // Extract original and improved content from the proposal object
            let originalContent =
                try
                    let propInfo = proposal.GetType().GetProperty("OriginalContent")
                    propInfo.GetValue(proposal) :?> string
                with _ -> ""

            let improvedContent =
                try
                    let propInfo = proposal.GetType().GetProperty("ImprovedContent")
                    propInfo.GetValue(proposal) :?> string
                with _ -> ""

            let event =
                createEvent "Improvement" fileName fileType true
                |> fun e -> { e with
                                OriginalContent = Some originalContent
                                ModifiedContent = Some improvedContent
                                ImprovementProposal = Some proposal }

            return! addEvent database event
        }

    /// <summary>
    /// Creates a feedback event and adds it to the database
    /// </summary>
    let recordFeedback (eventId: string) (feedback: string) (success: bool) =
        async {
            let! database = loadDatabase()

            let updatedEvents =
                database.Events
                |> List.map (fun e ->
                    if e.Id = eventId then
                        { e with
                            Feedback = Some feedback
                            Success = success }
                    else
                        e)

            let updatedDatabase =
                { database with
                    Events = updatedEvents
                    LastUpdated = DateTime.UtcNow }

            let databaseWithStats = updateStatistics updatedDatabase
            let! success = saveDatabase databaseWithStats

            return
                if success then
                    databaseWithStats
                else
                    updatedDatabase
        }

    /// <summary>
    /// Gets events by file name
    /// </summary>
    let getEventsByFile (fileName: string) =
        async {
            let! database = loadDatabase()

            return
                database.Events
                |> List.filter (fun e -> e.FileName = fileName)
        }

    /// <summary>
    /// Gets events by event type
    /// </summary>
    let getEventsByType (eventType: string) =
        async {
            let! database = loadDatabase()

            return
                database.Events
                |> List.filter (fun e -> e.EventType = eventType)
        }

    /// <summary>
    /// Gets events by file type
    /// </summary>
    let getEventsByFileType (fileType: string) =
        async {
            let! database = loadDatabase()

            return
                database.Events
                |> List.filter (fun e -> e.FileType = fileType)
        }

    /// <summary>
    /// Gets events by tag
    /// </summary>
    let getEventsByTag (tag: string) =
        async {
            let! database = loadDatabase()

            return
                database.Events
                |> List.filter (fun e -> e.Tags |> List.contains tag)
        }

    /// <summary>
    /// Gets the most recent events
    /// </summary>
    let getMostRecentEvents (count: int) =
        async {
            let! database = loadDatabase()

            return
                database.Events
                |> List.sortByDescending (fun e -> e.Timestamp)
                |> List.truncate count
        }

    /// <summary>
    /// Gets the statistics from the learning database
    /// </summary>
    let getStatistics () =
        async {
            let! database = loadDatabase()
            return database.Statistics
        }

    /// <summary>
    /// Gets a formatted string of the statistics
    /// </summary>
    let getStatisticsString () =
        async {
            let! database = loadDatabase()

            let sb = new System.Text.StringBuilder()

            sb.AppendLine("Learning Statistics:") |> ignore
            sb.AppendLine("--------------------") |> ignore

            // Total events
            if database.Statistics.ContainsKey("Total") then
                sb.AppendLine(sprintf "Total Events: %d" database.Statistics.["Total"]) |> ignore

            // Success/Failure
            if database.Statistics.ContainsKey("Successful") then
                sb.AppendLine(sprintf "Successful Events: %d" database.Statistics.["Successful"]) |> ignore

            if database.Statistics.ContainsKey("Failed") then
                sb.AppendLine(sprintf "Failed Events: %d" database.Statistics.["Failed"]) |> ignore

            // Event types
            sb.AppendLine("\nEvent Types:") |> ignore

            database.Statistics
            |> Seq.filter (fun kvp -> kvp.Key.StartsWith("EventType:"))
            |> Seq.sortByDescending (fun kvp -> kvp.Value)
            |> Seq.iter (fun kvp ->
                let eventType = kvp.Key.Substring("EventType:".Length)
                sb.AppendLine(sprintf "  %s: %d" eventType kvp.Value) |> ignore
            )

            // File types
            sb.AppendLine("\nFile Types:") |> ignore

            database.Statistics
            |> Seq.filter (fun kvp -> kvp.Key.StartsWith("FileType:"))
            |> Seq.sortByDescending (fun kvp -> kvp.Value)
            |> Seq.iter (fun kvp ->
                let fileType = kvp.Key.Substring("FileType:".Length)
                sb.AppendLine(sprintf "  %s: %d" fileType kvp.Value) |> ignore
            )

            // Tags
            sb.AppendLine("\nTags:") |> ignore

            database.Statistics
            |> Seq.filter (fun kvp -> kvp.Key.StartsWith("Tag:"))
            |> Seq.sortByDescending (fun kvp -> kvp.Value)
            |> Seq.iter (fun kvp ->
                let tag = kvp.Key.Substring("Tag:".Length)
                sb.AppendLine(sprintf "  %s: %d" tag kvp.Value) |> ignore
            )

            return sb.ToString()
        }

    /// <summary>
    /// Clears the learning database
    /// </summary>
    let clearDatabase () =
        async {
            let database = createDatabase()
            return! saveDatabase database
        }
