namespace Tars.Core

open System
open System.IO
open System.Text.Json

/// Category of an architectural or operational critique
type CritiqueCategory = 
    | ArchCritique  // Design flaws, coupling, abstraction leaks
    | OpsCritique    // Performance, resource leaks, budget overruns
    | CogCritique    // Reasoning loops, hallucinations, grounding issues
    | SecCritique       // Policy violations, sandbox escapes
    | OtherCritique

/// Severity of the critique
type CritiqueSeverity = 
    | CriticalCrit // Blocks successful execution of canonical tasks
    | HighCrit     // Significant friction or reliability risk
    | MediumCrit   // Notable issue, affects efficiency
    | LowCrit      // Technical debt or minor refinement

/// Status of a critique
type CritiqueStatus = 
    | Open
    | Mitigating
    | Resolved
    | WontFix

/// A self-critique entry
type Critique = { 
    Id: Guid
    Category: CritiqueCategory
    Severity: CritiqueSeverity
    Summary: string
    Details: string
    MitigationPlan: string option
    Status: CritiqueStatus
    CreatedAt: DateTime
    ResolvedAt: DateTime option
}

/// Service for managing self-critiques
type CritiqueService(storagePath: string) = 
    let mutable critiques = []
    
    let load() = 
        if File.Exists storagePath then 
            try 
                let json = File.ReadAllText storagePath
                critiques <- JsonSerializer.Deserialize<Critique list>(json)
            with _ -> ()
            
    let save() = 
        try 
            let dir = Path.GetDirectoryName storagePath
            if not (String.IsNullOrWhiteSpace dir) && not (Directory.Exists dir) then 
                Directory.CreateDirectory dir |> ignore
            let json = JsonSerializer.Serialize(critiques, JsonSerializerOptions(WriteIndented = true))
            File.WriteAllText(storagePath, json)
        with _ -> ()

    do load()

    member _.AddCritique(category, severity, summary, details, ?mitigation) = 
        let critique = { 
            Id = Guid.NewGuid()
            Category = category
            Severity = severity
            Summary = summary
            Details = details
            MitigationPlan = mitigation
            Status = Open
            CreatedAt = DateTime.UtcNow
            ResolvedAt = None
        }
        critiques <- critique :: critiques
        save()
        critique

    member _.GetAll() = critiques
    
    member _.GetActive() = 
        critiques |> List.filter (fun c -> c.Status = Open || c.Status = Mitigating)

    member _.Resolve(id: Guid) = 
        critiques <- critiques |> List.map (fun c -> 
            if c.Id = id then 
                { c with Status = Resolved; ResolvedAt = Some DateTime.UtcNow }
            else c)
        save()

    member _.UpdateStatus(id: Guid, status: CritiqueStatus) = 
        critiques <- critiques |> List.map (fun c -> 
            if c.Id = id then { c with Status = status }
            else c)
        save()

    /// Format critiques for agent context
    member this.FormatForContext() = 
        let active = this.GetActive()
        if active.IsEmpty then ""
        else 
            let lines = 
                active 
                |> List.sortByDescending (fun c -> c.Severity)
                |> List.map (fun c ->
                    $"- [%A{c.Severity}][%A{c.Category}] %s{c.Summary}: %s{c.Details}")
                |> String.concat "\n"
            
            "\n\n[SELF-CRITIQUE: KNOWN SYSTEM DEFICIENCIES]\n" +
            "The following issues are currently identified in your architecture or operations.\n" +
            "Take these into account during reasoning and avoid repeating failed patterns.\n" +
            lines + "\n"