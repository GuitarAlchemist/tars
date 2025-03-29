namespace TarsEngine.SelfImprovement

open System
open System.IO
open System.Text.Json
open System.Text.Json.Serialization
open System.Collections.Generic

/// <summary>
/// Represents a learning event
/// </summary>
type LearningEvent = {
    EventId: string
    EventType: string
    SourceFile: string
    Description: string
    Before: string option
    After: string option
    Feedback: string option
    Success: bool
    Timestamp: DateTime
    Metadata: Dictionary<string, string>
}

/// <summary>
/// Represents a learning history
/// </summary>
type LearningHistory = {
    Events: LearningEvent list
    Statistics: Dictionary<string, int>
}

/// <summary>
/// Functions for working with the learning system
/// </summary>
module LearningSystem =
    /// <summary>
    /// Creates a new learning event
    /// </summary>
    let createLearningEvent (eventType: string) (sourceFile: string) (description: string) (success: bool) =
        {
            EventId = Guid.NewGuid().ToString()
            EventType = eventType
            SourceFile = sourceFile
            Description = description
            Before = None
            After = None
            Feedback = None
            Success = success
            Timestamp = DateTime.UtcNow
            Metadata = new Dictionary<string, string>()
        }
    
    /// <summary>
    /// Creates a new learning history
    /// </summary>
    let createLearningHistory () =
        {
            Events = []
            Statistics = new Dictionary<string, int>()
        }
    
    /// <summary>
    /// Adds an event to the learning history
    /// </summary>
    let addEvent (history: LearningHistory) (event: LearningEvent) =
        // Update statistics
        let statistics = new Dictionary<string, int>(history.Statistics)
        
        // Update event type count
        let eventTypeKey = sprintf "EventType:%s" event.EventType
        if statistics.ContainsKey(eventTypeKey) then
            statistics.[eventTypeKey] <- statistics.[eventTypeKey] + 1
        else
            statistics.Add(eventTypeKey, 1)
        
        // Update success/failure count
        let successKey = if event.Success then "Success" else "Failure"
        if statistics.ContainsKey(successKey) then
            statistics.[successKey] <- statistics.[successKey] + 1
        else
            statistics.Add(successKey, 1)
        
        // Update file count
        let fileKey = sprintf "File:%s" event.SourceFile
        if statistics.ContainsKey(fileKey) then
            statistics.[fileKey] <- statistics.[fileKey] + 1
        else
            statistics.Add(fileKey, 1)
        
        // Update total count
        if statistics.ContainsKey("Total") then
            statistics.["Total"] <- statistics.["Total"] + 1
        else
            statistics.Add("Total", 1)
        
        { history with 
            Events = history.Events @ [event]
            Statistics = statistics 
        }
    
    /// <summary>
    /// Gets events by type
    /// </summary>
    let getEventsByType (history: LearningHistory) (eventType: string) =
        history.Events
        |> List.filter (fun e -> e.EventType = eventType)
    
    /// <summary>
    /// Gets events by file
    /// </summary>
    let getEventsByFile (history: LearningHistory) (sourceFile: string) =
        history.Events
        |> List.filter (fun e -> e.SourceFile = sourceFile)
    
    /// <summary>
    /// Gets successful events
    /// </summary>
    let getSuccessfulEvents (history: LearningHistory) =
        history.Events
        |> List.filter (fun e -> e.Success)
    
    /// <summary>
    /// Gets failed events
    /// </summary>
    let getFailedEvents (history: LearningHistory) =
        history.Events
        |> List.filter (fun e -> not e.Success)
    
    /// <summary>
    /// Gets events in a date range
    /// </summary>
    let getEventsByDateRange (history: LearningHistory) (startDate: DateTime) (endDate: DateTime) =
        history.Events
        |> List.filter (fun e -> e.Timestamp >= startDate && e.Timestamp <= endDate)
    
    /// <summary>
    /// Gets the most recent events
    /// </summary>
    let getMostRecentEvents (history: LearningHistory) (count: int) =
        history.Events
        |> List.sortByDescending (fun e -> e.Timestamp)
        |> List.truncate count
    
    /// <summary>
    /// Creates a code improvement event
    /// </summary>
    let createCodeImprovementEvent (sourceFile: string) (description: string) (before: string) (after: string) (success: bool) =
        let event = createLearningEvent "CodeImprovement" sourceFile description success
        { event with 
            Before = Some before
            After = Some after 
        }
    
    /// <summary>
    /// Creates a code analysis event
    /// </summary>
    let createCodeAnalysisEvent (sourceFile: string) (description: string) (feedback: string) (success: bool) =
        let event = createLearningEvent "CodeAnalysis" sourceFile description success
        { event with Feedback = Some feedback }
    
    /// <summary>
    /// Creates a learning feedback event
    /// </summary>
    let createFeedbackEvent (sourceFile: string) (description: string) (feedback: string) =
        let event = createLearningEvent "Feedback" sourceFile description true
        { event with Feedback = Some feedback }
    
    /// <summary>
    /// Serializes a learning event to JSON
    /// </summary>
    let serializeLearningEvent (event: LearningEvent) =
        let options = JsonSerializerOptions()
        options.WriteIndented <- true
        options.Converters.Add(JsonFSharpConverter())
        JsonSerializer.Serialize(event, options)
    
    /// <summary>
    /// Deserializes a learning event from JSON
    /// </summary>
    let deserializeLearningEvent (json: string) =
        let options = JsonSerializerOptions()
        options.Converters.Add(JsonFSharpConverter())
        JsonSerializer.Deserialize<LearningEvent>(json, options)
    
    /// <summary>
    /// Serializes a learning history to JSON
    /// </summary>
    let serializeLearningHistory (history: LearningHistory) =
        let options = JsonSerializerOptions()
        options.WriteIndented <- true
        options.Converters.Add(JsonFSharpConverter())
        JsonSerializer.Serialize(history, options)
    
    /// <summary>
    /// Deserializes a learning history from JSON
    /// </summary>
    let deserializeLearningHistory (json: string) =
        let options = JsonSerializerOptions()
        options.Converters.Add(JsonFSharpConverter())
        JsonSerializer.Deserialize<LearningHistory>(json, options)
    
    /// <summary>
    /// Saves a learning history to a file
    /// </summary>
    let saveLearningHistoryToFile (history: LearningHistory) (filePath: string) =
        let json = serializeLearningHistory history
        File.WriteAllText(filePath, json)
    
    /// <summary>
    /// Loads a learning history from a file
    /// </summary>
    let loadLearningHistoryFromFile (filePath: string) =
        let json = File.ReadAllText(filePath)
        deserializeLearningHistory json
    
    /// <summary>
    /// Gets learning statistics as a formatted string
    /// </summary>
    let getLearningStatisticsString (history: LearningHistory) =
        let sb = new System.Text.StringBuilder()
        
        sb.AppendLine("Learning Statistics:") |> ignore
        sb.AppendLine("--------------------") |> ignore
        
        // Total events
        if history.Statistics.ContainsKey("Total") then
            sb.AppendLine(sprintf "Total Events: %d" history.Statistics.["Total"]) |> ignore
        
        // Success/Failure
        if history.Statistics.ContainsKey("Success") then
            sb.AppendLine(sprintf "Successful Events: %d" history.Statistics.["Success"]) |> ignore
        
        if history.Statistics.ContainsKey("Failure") then
            sb.AppendLine(sprintf "Failed Events: %d" history.Statistics.["Failure"]) |> ignore
        
        // Event types
        sb.AppendLine("\nEvent Types:") |> ignore
        
        history.Statistics
        |> Seq.filter (fun kvp -> kvp.Key.StartsWith("EventType:"))
        |> Seq.sortByDescending (fun kvp -> kvp.Value)
        |> Seq.iter (fun kvp -> 
            let eventType = kvp.Key.Substring("EventType:".Length)
            sb.AppendLine(sprintf "  %s: %d" eventType kvp.Value) |> ignore
        )
        
        // Files
        sb.AppendLine("\nFiles:") |> ignore
        
        history.Statistics
        |> Seq.filter (fun kvp -> kvp.Key.StartsWith("File:"))
        |> Seq.sortByDescending (fun kvp -> kvp.Value)
        |> Seq.iter (fun kvp -> 
            let file = kvp.Key.Substring("File:".Length)
            sb.AppendLine(sprintf "  %s: %d" file kvp.Value) |> ignore
        )
        
        sb.ToString()
