namespace Tars.LinkedData

open System
open Tars.Core
open Tars.Knowledge

/// Tools for reasoning across multiple Linked Data sources
module MultiSourceReasoning =

    type Conflict = {
        Subject: string
        Predicate: RelationType
        Values: Map<string, string> // Source URI (string) -> Value
    }

    /// Detect conflicts in a set of beliefs by looking for the same (Subject, Predicate) with different objects
    let detectConflicts (beliefs: Belief list) : Conflict list =
        beliefs
        |> List.groupBy (fun (b: Belief) -> (b.Subject.Value, b.Predicate))
        |> List.choose (fun ((s, p), group) ->
            // Filter to beliefs that have external provenance
            let values = 
                group 
                |> List.choose (fun (b: Belief) -> 
                    match b.Provenance.Source with
                    | External uri -> Some (uri.ToString(), b.Object.Value)
                    | _ -> None)
                |> Map.ofList
            
            // If we have values from multiple sources and they aren't all the same
            let uniqueValues = values |> Map.toSeq |> Seq.map snd |> Seq.distinct |> Seq.toList
            if values.Count > 1 && uniqueValues.Length > 1 then
                Some { Subject = s; Predicate = p; Values = values }
            else
                None
        )

    /// Calculate a consensus value if possible (e.g., majority vote)
    let resolveConsensus (conflict: Conflict) : string option =
        let valueCounts = 
            conflict.Values 
            |> Map.toSeq 
            |> Seq.map snd 
            |> Seq.countBy id 
            |> Seq.sortByDescending snd
            |> Seq.toList
        
        match valueCounts with
        | (value, count) :: rest when rest.IsEmpty || count > (rest |> List.head |> snd) ->
            Some value // Clear winner
        | _ -> None // Tie or no data

    /// Score confidence in a belief based on source authority and consensus
    let scoreConfidence (sources: Map<string, float>) (belief: Belief) : float =
        match belief.Provenance.Source with
        | External uri -> 
            sources |> Map.tryFind (uri.ToString()) |> Option.defaultValue 0.5
        | _ -> 0.1