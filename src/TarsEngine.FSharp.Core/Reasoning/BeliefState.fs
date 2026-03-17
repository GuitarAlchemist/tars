namespace TarsEngine.FSharp.Core.Reasoning

open System
open System.IO
open System.Text.Json
open System.Text.Json.Serialization

/// Tetravalent Belief State Engine
/// Implements four-valued logic: T (True), F (False), U (Unknown), C (Contradictory)
module BeliefState =

    // ============================================================================
    // TYPE DEFINITIONS
    // ============================================================================

    /// Tetravalent truth value
    type TruthValue =
        | T  // True
        | F  // False
        | U  // Unknown
        | C  // Contradictory

    /// Evidence item supporting or contradicting a belief
    type EvidenceItem = {
        source: string
        claim: string
        timestamp: DateTime option
        reliability: float option
    }

    /// Complete belief state record
    type BeliefState = {
        proposition: string
        truth_value: TruthValue
        confidence: float
        evidence: EvidenceMap option
        last_updated: DateTime
        evaluated_by: string option
    }

    /// Evidence organized by support/contradiction
    and EvidenceMap = {
        supporting: EvidenceItem list
        contradicting: EvidenceItem list
    }

    /// Belief state store for persistence
    type BeliefStore = {
        beliefs: Map<string, BeliefState>
        created_at: DateTime
        updated_at: DateTime
    }

    // ============================================================================
    // CORE FUNCTIONS
    // ============================================================================

    /// Create a new belief with minimal information
    let createBelief (proposition: string) (truthValue: TruthValue) (confidence: float) : BeliefState =
        {
            proposition = proposition
            truth_value = truthValue
            confidence = max 0.0 (min 1.0 confidence)  // Clamp to [0,1]
            evidence = None
            last_updated = DateTime.UtcNow
            evaluated_by = None
        }

    /// Create a belief with evidence
    let createBeliefWithEvidence (proposition: string) (truthValue: TruthValue) (confidence: float)
                                  (supporting: EvidenceItem list) (contradicting: EvidenceItem list) : BeliefState =
        {
            proposition = proposition
            truth_value = truthValue
            confidence = max 0.0 (min 1.0 confidence)
            evidence = Some { supporting = supporting; contradicting = contradicting }
            last_updated = DateTime.UtcNow
            evaluated_by = None
        }

    /// Add evaluator information
    let withEvaluator (evaluator: string) (belief: BeliefState) : BeliefState =
        { belief with evaluated_by = Some evaluator }

    /// Update a belief's truth value and confidence
    let updateBelief (newTruthValue: TruthValue) (newConfidence: float) (belief: BeliefState) : BeliefState =
        {
            belief with
                truth_value = newTruthValue
                confidence = max 0.0 (min 1.0 newConfidence)
                last_updated = DateTime.UtcNow
        }

    /// Add supporting evidence to a belief
    let addSupportingEvidence (evidence: EvidenceItem) (belief: BeliefState) : BeliefState =
        let updated_evidence =
            match belief.evidence with
            | Some ev -> { ev with supporting = ev.supporting @ [evidence] }
            | None -> { supporting = [evidence]; contradicting = [] }
        { belief with evidence = Some updated_evidence; last_updated = DateTime.UtcNow }

    /// Add contradicting evidence to a belief
    let addContradictingEvidence (evidence: EvidenceItem) (belief: BeliefState) : BeliefState =
        let updated_evidence =
            match belief.evidence with
            | Some ev -> { ev with contradicting = ev.contradicting @ [evidence] }
            | None -> { supporting = []; contradicting = [evidence] }
        { belief with evidence = Some updated_evidence; last_updated = DateTime.UtcNow }

    /// Assess confidence based on evidence
    let assessConfidence (belief: BeliefState) : float =
        match belief.evidence with
        | None -> belief.confidence
        | Some ev ->
            let supportingWeight =
                ev.supporting
                |> List.sumBy (fun e -> e.reliability |> Option.defaultValue 0.5)

            let contradictingWeight =
                ev.contradicting
                |> List.sumBy (fun e -> e.reliability |> Option.defaultValue 0.5)

            let totalWeight = supportingWeight + contradictingWeight
            if totalWeight = 0.0 then belief.confidence
            else supportingWeight / (supportingWeight + contradictingWeight)

    /// Detect if belief is stale (older than threshold)
    let detectStale (thresholdHours: int) (belief: BeliefState) : bool =
        let elapsed = DateTime.UtcNow - belief.last_updated
        elapsed.TotalHours > float thresholdHours

    /// Reconcile contradictory evidence and update truth value
    let reconcileContradictions (belief: BeliefState) : BeliefState =
        match belief.evidence with
        | None -> belief
        | Some ev ->
            if (List.length ev.supporting > 0) && (List.length ev.contradicting > 0) then
                // Evidence exists on both sides -> Contradictory
                { belief with truth_value = C }
            else if List.length ev.supporting > 0 && List.length ev.contradicting = 0 then
                // Only supporting evidence
                { belief with truth_value = T }
            else if List.length ev.contradicting > 0 && List.length ev.supporting = 0 then
                // Only contradicting evidence
                { belief with truth_value = F }
            else
                // No evidence
                { belief with truth_value = U }

    // ============================================================================
    // PERSISTENCE FUNCTIONS
    // ============================================================================

    /// Convert TruthValue to string for JSON serialization
    let truthValueToString (tv: TruthValue) : string =
        match tv with
        | T -> "T"
        | F -> "F"
        | U -> "U"
        | C -> "C"

    /// Convert string to TruthValue
    let stringToTruthValue (s: string) : TruthValue =
        match s with
        | "T" -> T
        | "F" -> F
        | "U" -> U
        | "C" -> C
        | _ -> U  // Default to Unknown for invalid values

    /// Save belief state to JSON file
    let saveBeliefToFile (directory: string) (belief: BeliefState) : Result<string, string> =
        try
            Directory.CreateDirectory(directory) |> ignore

            let fileName = Path.Combine(directory, $"{Guid.NewGuid()}.json")
            let json = JsonSerializer.Serialize(belief, JsonSerializerOptions(WriteIndented = true))
            File.WriteAllText(fileName, json)
            Ok fileName
        with
        | ex -> Error $"Failed to save belief: {ex.Message}"

    /// Load belief state from JSON file
    let loadBeliefFromFile (filePath: string) : Result<BeliefState, string> =
        try
            let json = File.ReadAllText(filePath)
            let belief = JsonSerializer.Deserialize<BeliefState>(json)
            if belief = null then
                Error "Deserialized belief is null"
            else
                Ok belief
        with
        | ex -> Error $"Failed to load belief: {ex.Message}"

    /// Save multiple beliefs to store file
    let saveBeliefStore (directory: string) (beliefs: BeliefState list) : Result<string, string> =
        try
            Directory.CreateDirectory(directory) |> ignore

            let store: BeliefStore = {
                beliefs = beliefs |> List.map (fun b -> (b.proposition, b)) |> Map.ofList
                created_at = DateTime.UtcNow
                updated_at = DateTime.UtcNow
            }

            let filePath = Path.Combine(directory, "belief_store.json")
            let json = JsonSerializer.Serialize(store, JsonSerializerOptions(WriteIndented = true))
            File.WriteAllText(filePath, json)
            Ok filePath
        with
        | ex -> Error $"Failed to save belief store: {ex.Message}"

    /// Load belief store from file
    let loadBeliefStore (filePath: string) : Result<BeliefStore, string> =
        try
            let json = File.ReadAllText(filePath)
            let store = JsonSerializer.Deserialize<BeliefStore>(json)
            if store = null then
                Error "Deserialized store is null"
            else
                Ok store
        with
        | ex -> Error $"Failed to load belief store: {ex.Message}"

    // ============================================================================
    // COMPLIANCE & REPORTING
    // ============================================================================

    /// Generate Galactic Protocol compliance report for a belief
    let generateComplianceReport (belief: BeliefState) : Map<string, obj> =
        Map.ofList [
            ("proposition", belief.proposition :> obj)
            ("truth_value", truthValueToString belief.truth_value :> obj)
            ("confidence", belief.confidence :> obj)
            ("last_updated", belief.last_updated.ToString("O") :> obj)
            ("evaluated_by", belief.evaluated_by |> Option.defaultValue "unknown" :> obj)
            ("evidence_count",
                match belief.evidence with
                | None -> 0 :> obj
                | Some ev -> (List.length ev.supporting + List.length ev.contradicting) :> obj)
            ("is_stale", detectStale 24 belief :> obj)
            ("assessed_confidence", assessConfidence belief :> obj)
        ]

    /// Generate report for multiple beliefs
    let generateBulkComplianceReport (beliefs: BeliefState list) : Map<string, obj> =
        let reports = beliefs |> List.map generateComplianceReport

        let staleCount = beliefs |> List.filter (detectStale 24) |> List.length
        let contradictionCount = beliefs |> List.filter (fun b -> b.truth_value = C) |> List.length
        let unknownCount = beliefs |> List.filter (fun b -> b.truth_value = U) |> List.length

        Map.ofList [
            ("total_beliefs", beliefs.Length :> obj)
            ("stale_beliefs", staleCount :> obj)
            ("contradictory_beliefs", contradictionCount :> obj)
            ("unknown_beliefs", unknownCount :> obj)
            ("average_confidence",
                if beliefs.IsEmpty then 0.0 :> obj
                else (beliefs |> List.sumBy (fun b -> b.confidence) |> fun sum -> sum / float beliefs.Length) :> obj)
            ("timestamp", DateTime.UtcNow.ToString("O") :> obj)
            ("beliefs", reports :> obj)
        ]

    // ============================================================================
    // UTILITIES
    // ============================================================================

    /// Create evidence item with defaults
    let createEvidence (source: string) (claim: string) (reliability: float) : EvidenceItem =
        {
            source = source
            claim = claim
            timestamp = Some DateTime.UtcNow
            reliability = Some (max 0.0 (min 1.0 reliability))
        }

    /// Filter beliefs by truth value
    let filterByTruthValue (tv: TruthValue) (beliefs: BeliefState list) : BeliefState list =
        beliefs |> List.filter (fun b -> b.truth_value = tv)

    /// Filter stale beliefs
    let filterStale (thresholdHours: int) (beliefs: BeliefState list) : BeliefState list =
        beliefs |> List.filter (detectStale thresholdHours)
