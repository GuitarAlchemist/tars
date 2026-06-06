namespace Tars.Core

open System
open System.IO
open System.Text.Json
open System.Threading.Tasks
open Tars.Core.Errors

// ============================================================================
// INCIDENT DOMAIN
// ============================================================================

/// Status of an incident in the triage pipeline
type IncidentStatus =
    | New           // Just reported
    | Triaging      // Under investigation
    | Ignored       // False positive or low priority
    | Confirmed     // Valid issue to be fixed
    | InProgress    // Fix in progress
    | Resolved      // Fix verified
    | Closed        // Done

/// Represents a persisted failure incident
type Incident =
    { Id: Guid
      Error: TarsError
      Status: IncidentStatus
      CreatedAt: DateTime
      UpdatedAt: DateTime
      TriageNotes: string option
      Resolution: string option
      ImpactLevel: int option // 1-10 scale
      Tags: string list }

// ============================================================================
// DTOs & SERIALIZATION HELPER
// ============================================================================
module Serialization =
    type PersistedError =
        { Id: Guid
          Category: string // Simplified to string representation
          Severity: string
          Timestamp: DateTime
          CorrelationId: Guid option
          Context: Map<string, string>
          Recoverable: bool
          Message: string } // Store the formatted message

    type PersistedIncident =
        { Id: Guid
          Error: PersistedError
          Status: string
          CreatedAt: DateTime
          UpdatedAt: DateTime
          TriageNotes: string option
          Resolution: string option
          ImpactLevel: int option
          Tags: string list }

    let toDto (i: Incident) =
        let errDto =
            { Id = i.Error.Id
              Category = i.Error.Category.ToString()
              Severity = i.Error.Severity.ToString()
              Timestamp = i.Error.Timestamp
              CorrelationId = i.Error.CorrelationId
              Context = i.Error.Context
              Recoverable = i.Error.Recoverable
              Message = message i.Error }
              
        { Id = i.Id
          Error = errDto
          Status = i.Status.ToString()
          CreatedAt = i.CreatedAt
          UpdatedAt = i.UpdatedAt
          TriageNotes = i.TriageNotes
          Resolution = i.Resolution
          ImpactLevel = i.ImpactLevel
          Tags = i.Tags }

    let fromDto (d: PersistedIncident) : Incident =
        // Reconstruct Status
        let status = 
             match d.Status with
             | "New" -> New
             | "Triaging" -> Triaging
             | "Ignored" -> Ignored
             | "Confirmed" -> Confirmed
             | "InProgress" -> InProgress
             | "Resolved" -> Resolved
             | "Closed" -> Closed
             | _ -> New // Default fallback

        // Reconstruct Severity (best effort)
        let severity =
            match d.Error.Severity with
            | "Critical" -> ErrorSeverity.Critical
            | "High" -> ErrorSeverity.High
            | "Low" -> ErrorSeverity.Low
            | _ -> ErrorSeverity.Medium

        // Reconstruct Error (simplified category)
        let error =
            { Id = d.Error.Id
              Category = InternalError(d.Error.Category) // Wrap old category string in InternalError wrapper as fallback
              Severity = severity
              Timestamp = d.Error.Timestamp
              CorrelationId = d.Error.CorrelationId
              Context = d.Error.Context
              InnerException = None // Lost in serialization
              Recoverable = d.Error.Recoverable }

        { Id = d.Id
          Error = error
          Status = status
          CreatedAt = d.CreatedAt
          UpdatedAt = d.UpdatedAt
          TriageNotes = d.TriageNotes
          Resolution = d.Resolution
          ImpactLevel = d.ImpactLevel
          Tags = d.Tags }

// ============================================================================
// LEDGER INTERFACE
// ============================================================================

/// Interface for managing the incident pipeline
type IIncidentLedger =
    /// Record a new failure incident
    abstract member ReportAsync: TarsError -> Task<Incident>
    
    /// Get an incident by ID
    abstract member GetAsync: Guid -> Task<Incident option>
    
    /// List incidents with optional filtering
    abstract member ListAsync: status: IncidentStatus option * limit: int option -> Task<Incident list>
    
    /// Update an incident (e.g. change status, add notes)
    abstract member UpdateAsync: Incident -> Task<unit>
    
    /// Get statistics about incidents
    abstract member GetStatsAsync: unit -> Task<Map<IncidentStatus, int>>

// ============================================================================
// FILE-BASED IMPLEMENTATION
// ============================================================================

/// Simple file-based ledger storing incidents as JSON files
type FileSystemIncidentLedger(storagePath: string) =
    let ensureDirectory () =
        if not (Directory.Exists(storagePath)) then
            Directory.CreateDirectory(storagePath) |> ignore

    let getFilePath (id: Guid) =
        Path.Combine(storagePath, $"{id}.json")

    let jsonOptions = 
        let options = JsonSerializerOptions()
        options.WriteIndented <- true
        options

    interface IIncidentLedger with
        member _.ReportAsync(error: TarsError) =
            task {
                ensureDirectory()
                
                let incident =
                    { Id = error.Id
                      Error = error
                      Status = New
                      CreatedAt = DateTime.UtcNow
                      UpdatedAt = DateTime.UtcNow
                      TriageNotes = None
                      Resolution = None
                      ImpactLevel = None
                      Tags = [] }
                
                let dto = Serialization.toDto incident
                let path = getFilePath incident.Id
                let json = JsonSerializer.Serialize(dto, jsonOptions)
                do! File.WriteAllTextAsync(path, json)
                
                return incident
            }

        member _.GetAsync(id: Guid) =
            task {
                ensureDirectory()
                let path = getFilePath id
                
                if File.Exists(path) then
                    let! json = File.ReadAllTextAsync(path)
                    try
                        let dto = JsonSerializer.Deserialize<Serialization.PersistedIncident>(json, jsonOptions)
                        return Some (Serialization.fromDto dto)
                    with _ -> 
                        return None
                else
                    return None
            }

        member _.ListAsync(statusFilter: IncidentStatus option, limit: int option) =
            task {
                ensureDirectory()
                
                let files = 
                    Directory.GetFiles(storagePath, "*.json")
                    |> Array.sortByDescending File.GetCreationTime
                    
                let! results =
                    files
                    |> Array.map (fun f -> 
                        task {
                            try
                                let! json = File.ReadAllTextAsync(f)
                                let dto = JsonSerializer.Deserialize<Serialization.PersistedIncident>(json, jsonOptions)
                                return Some (Serialization.fromDto dto)
                            with _ -> 
                                return None
                        })
                    |> Task.WhenAll
                    
                let filtered =
                    results
                    |> Array.choose id
                    |> Array.filter (fun i -> 
                        match statusFilter with
                        | Some s -> i.Status = s
                        | None -> true)
                
                match limit with
                | Some n -> return filtered |> Array.truncate n |> Array.toList
                | None -> return filtered |> Array.toList
            }

        member _.UpdateAsync(incident: Incident) =
            task {
                ensureDirectory()
                let updated = { incident with UpdatedAt = DateTime.UtcNow }
                let dto = Serialization.toDto updated
                let path = getFilePath incident.Id
                let json = JsonSerializer.Serialize(dto, jsonOptions)
                do! File.WriteAllTextAsync(path, json)
            }
            
        member _.GetStatsAsync() =
            task {
                ensureDirectory()
                let files = Directory.GetFiles(storagePath, "*.json")
                 
                let! incidents =
                    files
                    |> Array.map (fun f -> 
                        task {
                            try
                                let! json = File.ReadAllTextAsync(f)
                                let dto = JsonSerializer.Deserialize<Serialization.PersistedIncident>(json, jsonOptions)
                                return Some (Serialization.fromDto dto)
                            with _ -> 
                                return None
                        })
                    |> Task.WhenAll
                 
                let valid = incidents |> Array.choose id
                 
                return
                    valid
                    |> Array.groupBy (fun i -> i.Status)
                    |> Array.map (fun (k, v) -> k, v.Length)
                    |> Map.ofArray
            }

// ============================================================================
// FAILURE PIPELINE 
// ============================================================================

/// Tracks the lifecycle of a failure from ingestion to resolution
module FailurePipeline =
    
    /// Create a new ledger in the default location
    let createDefault () =
        let path = 
            Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), 
                ".tars", 
                "incidents"
            )
        FileSystemIncidentLedger(path) :> IIncidentLedger

    /// Triage an incident (move to Triaging status)
    let startTriage (incident: Incident) (ledger: IIncidentLedger) =
        task {
            let updated = { incident with Status = Triaging }
            do! ledger.UpdateAsync(updated)
            return updated
        }
        
    /// Resolve an incident
    let resolve (incident: Incident) (resolution: string) (ledger: IIncidentLedger) =
        task {
            let updated = 
                { incident with 
                    Status = Resolved
                    Resolution = Some resolution }
            do! ledger.UpdateAsync(updated)
            return updated
        }
