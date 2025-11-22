// TARS.GI Belief Graph - Symbolic Working Memory with Four-Valued Logic
// Explicit, inspectable reasoning and long-horizon consistency
// Contradiction detection and belief maintenance

namespace Tars.Core

open System
open Types

/// Belief Graph operations for symbolic reasoning
module BeliefGraph =
    
    /// Calculate entropy of belief set
    let calculateEntropy (beliefs: Belief list) : float =
        if beliefs.IsEmpty then 0.0
        else
            let totalBeliefs = float beliefs.Length
            let truthCounts = 
                beliefs
                |> List.groupBy (fun b -> b.Truth)
                |> List.map (fun (truth, group) -> (truth, float group.Length))
            
            truthCounts
            |> List.sumBy (fun (_, count) -> 
                let p = count / totalBeliefs
                if p > 0.0 then -p * Math.Log2(p) else 0.0)
    
    /// Add belief with contradiction detection
    let addBelief (beliefs: Belief list) (newBelief: Belief) : Belief list * Belief list =
        let contradictions = 
            beliefs
            |> List.filter (fun b -> 
                b.Proposition = newBelief.Proposition && 
                b.Truth <> newBelief.Truth && 
                b.Truth <> Unknown && 
                newBelief.Truth <> Unknown)
        
        let updatedBelief = 
            if not contradictions.IsEmpty then
                { newBelief with Truth = Both } // Mark as contradiction
            else
                newBelief
        
        (updatedBelief :: beliefs, contradictions)

/// Symbolic Working Memory with Four-Valued Logic
type SymbolicMemory(maxBeliefs: int) =
    let mutable beliefs = Map.empty<string, Belief>
    
    /// Add or update belief with provenance tracking
    member this.AddBelief(belief: Belief) =
        let currentBeliefs = beliefs |> Map.values |> List.ofSeq
        let (updatedBeliefs, contradictions) = BeliefGraph.addBelief currentBeliefs belief
        
        for b in updatedBeliefs do
            beliefs <- Map.add b.Id b beliefs
        
        if beliefs.Count > maxBeliefs then
            let oldestBelief = beliefs |> Map.values |> Seq.minBy (fun b -> b.Timestamp)
            beliefs <- Map.remove oldestBelief.Id beliefs
        
        contradictions
    
    /// Calculate belief entropy
    member _.CalculateEntropy() =
        beliefs |> Map.values |> List.ofSeq |> BeliefGraph.calculateEntropy
    
    /// Get all beliefs
    member _.GetAllBeliefs() = 
        beliefs |> Map.values |> List.ofSeq
