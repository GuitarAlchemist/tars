namespace TarsEngine.SelfImprovement

open System
open System.IO
open System.Text.Json
open System.Text.Json.Serialization
open System.Threading.Tasks
open System.Collections.Generic

/// <summary>
/// Represents the frequency of a scheduled workflow
/// </summary>
type ScheduleFrequency =
    | OneTime
    | Daily
    | Weekly
    | Monthly

/// <summary>
/// Represents a scheduled workflow
/// </summary>
type ScheduledWorkflow = {
    /// <summary>
    /// The ID of the scheduled workflow
    /// </summary>
    Id: string
    
    /// <summary>
    /// The name of the workflow
    /// </summary>
    Name: string
    
    /// <summary>
    /// The frequency of the workflow
    /// </summary>
    Frequency: ScheduleFrequency
    
    /// <summary>
    /// The next run time of the workflow
    /// </summary>
    NextRunTime: DateTime
    
    /// <summary>
    /// The last run time of the workflow
    /// </summary>
    LastRunTime: DateTime option
    
    /// <summary>
    /// The target directories for the workflow
    /// </summary>
    TargetDirectories: string list
    
    /// <summary>
    /// The maximum duration of the workflow in minutes
    /// </summary>
    MaxDurationMinutes: int
    
    /// <summary>
    /// Whether the workflow is enabled
    /// </summary>
    IsEnabled: bool
    
    /// <summary>
    /// The time when the scheduled workflow was created
    /// </summary>
    CreatedAt: DateTime
    
    /// <summary>
    /// The time when the scheduled workflow was last updated
    /// </summary>
    LastUpdated: DateTime
}

/// <summary>
/// Module for managing workflow schedules
/// </summary>
module WorkflowScheduler =
    /// <summary>
    /// The default file path for storing workflow schedules
    /// </summary>
    let defaultSchedulesPath = "workflow_schedules.json"
    
    /// <summary>
    /// Creates a new scheduled workflow
    /// </summary>
    let create name frequency nextRunTime targetDirectories maxDurationMinutes =
        {
            Id = Guid.NewGuid().ToString()
            Name = name
            Frequency = frequency
            NextRunTime = nextRunTime
            LastRunTime = None
            TargetDirectories = targetDirectories
            MaxDurationMinutes = maxDurationMinutes
            IsEnabled = true
            CreatedAt = DateTime.UtcNow
            LastUpdated = DateTime.UtcNow
        }
    
    /// <summary>
    /// Calculates the next run time based on the frequency
    /// </summary>
    let calculateNextRunTime frequency (currentTime: DateTime) =
        match frequency with
        | OneTime -> 
            // One-time schedules don't have a next run time
            DateTime.MaxValue
        | Daily -> 
            currentTime.AddDays(1.0)
        | Weekly -> 
            currentTime.AddDays(7.0)
        | Monthly -> 
            currentTime.AddMonths(1)
    
    /// <summary>
    /// Updates the next run time of a scheduled workflow
    /// </summary>
    let updateNextRunTime schedule =
        { schedule with 
            NextRunTime = calculateNextRunTime schedule.Frequency DateTime.UtcNow
            LastRunTime = Some DateTime.UtcNow
            LastUpdated = DateTime.UtcNow }
    
    /// <summary>
    /// Enables a scheduled workflow
    /// </summary>
    let enable schedule =
        { schedule with 
            IsEnabled = true
            LastUpdated = DateTime.UtcNow }
    
    /// <summary>
    /// Disables a scheduled workflow
    /// </summary>
    let disable schedule =
        { schedule with 
            IsEnabled = false
            LastUpdated = DateTime.UtcNow }
    
    /// <summary>
    /// Gets the due schedules
    /// </summary>
    let getDueSchedules (schedules: ScheduledWorkflow list) =
        let now = DateTime.UtcNow
        schedules
        |> List.filter (fun s -> s.IsEnabled && s.NextRunTime <= now)
    
    /// <summary>
    /// Saves the workflow schedules to a file
    /// </summary>
    let save (schedules: ScheduledWorkflow list) (path: string) =
        task {
            let options = JsonSerializerOptions()
            options.WriteIndented <- true
            options.Converters.Add(JsonFSharpConverter())
            
            let json = JsonSerializer.Serialize(schedules, options)
            do! File.WriteAllTextAsync(path, json)
            return schedules
        }
    
    /// <summary>
    /// Loads the workflow schedules from a file
    /// </summary>
    let load (path: string) =
        task {
            if File.Exists(path) then
                let! json = File.ReadAllTextAsync(path)
                let options = JsonSerializerOptions()
                options.Converters.Add(JsonFSharpConverter())
                
                return JsonSerializer.Deserialize<ScheduledWorkflow list>(json, options)
            else
                return []
        }
    
    /// <summary>
    /// Adds a schedule to the list
    /// </summary>
    let addSchedule schedule schedules =
        schedule :: schedules
    
    /// <summary>
    /// Updates a schedule in the list
    /// </summary>
    let updateSchedule schedule schedules =
        schedules
        |> List.map (fun s -> if s.Id = schedule.Id then schedule else s)
    
    /// <summary>
    /// Removes a schedule from the list
    /// </summary>
    let removeSchedule scheduleId schedules =
        schedules
        |> List.filter (fun s -> s.Id <> scheduleId)
