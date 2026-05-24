namespace Tars.Core

open System
open System.IO
open System.Text.Json
open System.Text.Json.Serialization

// =============================================================================
// GOVERNANCE GENERATION — Grammar-Constrained Demerzel Artifact Generation
// =============================================================================
//
// Generates valid Demerzel governance artifacts using EBNF grammars.
// Each artifact type has a corresponding grammar in v2/grammars/governance/
// and a JSON schema in governance/demerzel/schemas/contracts/.

/// Tetravalent logic value used in Demerzel belief states
[<JsonConverter(typeof<JsonStringEnumConverter>)>]
type TetraValue =
    | T  // True — verified with evidence
    | F  // False — refuted with evidence
    | U  // Unknown — insufficient evidence
    | C  // Contradictory — conflicting evidence

/// Directive priority levels
type DirectivePriority = P0 | P1 | P2 | P3

/// Directive types that Demerzel can issue
type DirectiveType =
    | GovernanceDirective
    | ComplianceRequest
    | ReconnaissanceRequest
    | KnowledgeTransfer
    | ViolationRemediation

/// Target repository for governance messages
type TargetRepo = Ix | Tars | Ga | Demerzel

/// A Demerzel governance directive
type GovernanceDirective = {
    DirectiveId: string
    Type: DirectiveType
    From: string
    To: TargetRepo
    Priority: DirectivePriority
    Subject: string
    Requirements: string list
    Deadline: DateOnly option
    ComplianceVerification: string option
}

/// Compliance status for a report
type ComplianceStatus =
    | Compliant
    | NonCompliant
    | Partial
    | NotApplicable

/// Evidence supporting compliance
type ComplianceEvidence = {
    Type: string
    Description: string
}

/// A compliance report sent to Demerzel
type ComplianceReport = {
    ReportId: string
    DirectiveId: string
    Repo: TargetRepo
    Status: ComplianceStatus
    Timestamp: DateTimeOffset
    Evidence: ComplianceEvidence list
    ArticlesChecked: int list
    Notes: string option
}

/// A single belief in a snapshot
type GovBeliefEntry = {
    Proposition: string
    Value: TetraValue
    Confidence: float
    Evidence: string list
    UpdatedAt: DateOnly option
}

/// A belief snapshot exported for reconnaissance
type BeliefSnapshot = {
    Repo: TargetRepo
    Timestamp: DateTimeOffset
    GovBeliefs: GovBeliefEntry list
}

/// PDCA improvement classification
type KaizenClassification = Reactive | Proactive | Innovative

/// PDCA phase
type PdcaPhase = Plan | Do | Check | Act

/// A Kaizen PDCA state
type GovPdcaState = {
    Name: string
    Phase: PdcaPhase
    Classification: KaizenClassification
    StartedAt: DateOnly
    Experiment: bool
    Hypothesis: string option
    SuccessMetrics: string list
    Observations: string list
}

/// Module for loading governance grammars
module GovernanceGrammars =

    let private grammarsDir () =
        let assemblyDir = AppContext.BaseDirectory
        // Try relative paths from various locations
        [ Path.Combine(assemblyDir, "..", "..", "..", "..", "grammars", "governance")
          Path.Combine(assemblyDir, "grammars", "governance")
          Path.Combine("v2", "grammars", "governance") ]
        |> List.tryFind Directory.Exists

    /// Load an EBNF grammar file by name
    let loadGrammar (name: string) : Result<string, string> =
        match grammarsDir () with
        | Some dir ->
            let path = Path.Combine(dir, $"{name}.ebnf")
            if File.Exists(path) then
                Result.Ok (File.ReadAllText(path))
            else
                Result.Error $"Grammar file not found: {path}"
        | None ->
            Result.Error "Governance grammars directory not found"

    /// Available governance grammar names
    let availableGrammars = [
        "governance-directive"
        "compliance-report"
        "belief-snapshot"
        "pdca-state"
    ]

    /// Load all governance grammars
    let loadAll () : Map<string, Result<string, string>> =
        availableGrammars
        |> List.map (fun name -> name, loadGrammar name)
        |> Map.ofList

/// Module for generating governance artifacts as JSON
module GovernanceGeneration =

    let private jsonOptions =
        let opts = JsonSerializerOptions(WriteIndented = true)
        opts.Converters.Add(JsonStringEnumConverter())
        opts

    let private repoToString = function
        | Ix -> "ix" | Tars -> "tars" | Ga -> "ga" | Demerzel -> "demerzel"

    let private priorityToString = function
        | P0 -> "P0" | P1 -> "P1" | P2 -> "P2" | P3 -> "P3"

    let private tetraToString = function
        | T -> "T" | F -> "F" | U -> "U" | C -> "C"

    let private phaseToString = function
        | Plan -> "plan" | Do -> "do" | Check -> "check" | Act -> "act"

    let private classToString = function
        | Reactive -> "reactive" | Proactive -> "proactive" | Innovative -> "innovative"

    let private statusToString = function
        | Compliant -> "compliant" | NonCompliant -> "non_compliant"
        | Partial -> "partial" | NotApplicable -> "not_applicable"

    let private typeToString = function
        | GovernanceDirective -> "governance_directive"
        | ComplianceRequest -> "compliance_request"
        | ReconnaissanceRequest -> "reconnaissance_request"
        | KnowledgeTransfer -> "knowledge_transfer"
        | ViolationRemediation -> "violation_remediation"

    /// Serialize a governance directive to JSON
    let serializeDirective (d: GovernanceDirective) : string =
        let doc = dict [
            "directive_id", box d.DirectiveId
            "type", box (typeToString d.Type)
            "from", box d.From
            "to", box (repoToString d.To)
            "priority", box (priorityToString d.Priority)
            "subject", box d.Subject
            "requirements", box (d.Requirements |> List.toArray)
        ]
        let mutable m = System.Collections.Generic.Dictionary<string, obj>(doc)
        d.Deadline |> Option.iter (fun dl -> m["deadline"] <- box (dl.ToString("yyyy-MM-dd")))
        d.ComplianceVerification |> Option.iter (fun cv -> m["compliance_verification"] <- box cv)
        JsonSerializer.Serialize(m, jsonOptions)

    /// Serialize a compliance report to JSON
    let serializeComplianceReport (r: ComplianceReport) : string =
        let evidence = r.Evidence |> List.map (fun e ->
            dict [ "type", box e.Type; "description", box e.Description ] :> obj)
        let doc = dict [
            "report_id", box r.ReportId
            "directive_id", box r.DirectiveId
            "repo", box (repoToString r.Repo)
            "status", box (statusToString r.Status)
            "timestamp", box (r.Timestamp.ToString("yyyy-MM-ddTHH:mm:ssZ"))
            "evidence", box (evidence |> List.toArray)
            "articles_checked", box (r.ArticlesChecked |> List.toArray)
        ]
        let mutable m = System.Collections.Generic.Dictionary<string, obj>(doc)
        r.Notes |> Option.iter (fun n -> m["notes"] <- box n)
        JsonSerializer.Serialize(m, jsonOptions)

    /// Serialize a belief snapshot to JSON
    let serializeBeliefSnapshot (s: BeliefSnapshot) : string =
        let beliefs = s.GovBeliefs |> List.map (fun b ->
            let entry = dict [
                "proposition", box b.Proposition
                "value", box (tetraToString b.Value)
                "confidence", box b.Confidence
                "evidence", box (b.Evidence |> List.toArray)
            ]
            let mutable m = System.Collections.Generic.Dictionary<string, obj>(entry)
            b.UpdatedAt |> Option.iter (fun d -> m["updated_at"] <- box (d.ToString("yyyy-MM-dd")))
            m :> obj)
        let doc = dict [
            "repo", box (repoToString s.Repo)
            "timestamp", box (s.Timestamp.ToString("yyyy-MM-ddTHH:mm:ssZ"))
            "beliefs", box (beliefs |> List.toArray)
        ]
        JsonSerializer.Serialize(doc, jsonOptions)

    /// Serialize a PDCA state to JSON
    let serializePdcaState (p: GovPdcaState) : string =
        let doc = dict [
            "name", box p.Name
            "phase", box (phaseToString p.Phase)
            "classification", box (classToString p.Classification)
            "started_at", box (p.StartedAt.ToString("yyyy-MM-dd"))
            "experiment", box p.Experiment
            "success_metrics", box (p.SuccessMetrics |> List.toArray)
            "observations", box (p.Observations |> List.toArray)
        ]
        let mutable m = System.Collections.Generic.Dictionary<string, obj>(doc)
        p.Hypothesis |> Option.iter (fun h -> m["hypothesis"] <- box h)
        JsonSerializer.Serialize(m, jsonOptions)

/// Module for validating generated artifacts against Demerzel schemas
module GovernanceValidation =

    /// Validate that a JSON string is parseable
    let isValidJson (json: string) : bool =
        try
            JsonDocument.Parse(json) |> ignore
            true
        with _ -> false

    /// Check if a JSON element has a property
    let private hasProperty (elem: JsonElement) (name: string) : bool =
        let mutable dummy = Unchecked.defaultof<JsonElement>
        elem.TryGetProperty(name, &dummy)

    /// Get a property from a JSON element
    let private tryGetProp (elem: JsonElement) (name: string) : JsonElement option =
        let mutable result = Unchecked.defaultof<JsonElement>
        if elem.TryGetProperty(name, &result) then Some result else None

    /// Validate a directive has required fields
    let validateDirective (json: string) : Result<unit, string list> =
        try
            let doc = JsonDocument.Parse(json)
            let root = doc.RootElement
            let requiredFields = ["directive_id"; "type"; "from"; "to"; "priority"; "subject"; "requirements"]
            let errors =
                requiredFields
                |> List.choose (fun name ->
                    if hasProperty root name then None
                    else Some $"Missing required field: {name}")
            if errors.IsEmpty then Result.Ok () else Result.Error errors
        with ex ->
            Result.Error [$"Invalid JSON: {ex.Message}"]

    /// Validate a belief snapshot has required fields and valid tetravalent values
    let validateBeliefSnapshot (json: string) : Result<unit, string list> =
        try
            let doc = JsonDocument.Parse(json)
            let root = doc.RootElement
            let fieldErrors =
                ["repo"; "timestamp"; "beliefs"]
                |> List.choose (fun name ->
                    if hasProperty root name then None
                    else Some $"Missing required field: {name}")

            let valueErrors =
                match tryGetProp root "beliefs" with
                | Some beliefsElem ->
                    [ for belief in beliefsElem.EnumerateArray() do
                        match tryGetProp belief "value" with
                        | Some valueElem ->
                            let v = valueElem.GetString()
                            if not (["T"; "F"; "U"; "C"] |> List.contains v) then
                                yield $"Invalid tetravalent value: {v}"
                        | None -> () ]
                | None -> []

            let allErrors = fieldErrors @ valueErrors
            if allErrors.IsEmpty then Result.Ok () else Result.Error allErrors
        with ex ->
            Result.Error [$"Invalid JSON: {ex.Message}"]
