module Sandbox

open System
open Tars.Core
open Tars.Core.TemporalKnowledgeGraph
open Tars.Core.CommunityDetection

[<EntryPoint>]
let main argv =
    printfn "Running LPA Verification..."
    
    let graph = TemporalGraph()
    
    // Community 1: A <-> B
    let a = ConceptE { Name = "A"; Description = ""; RelatedConcepts = [] }
    let b = ConceptE { Name = "B"; Description = ""; RelatedConcepts = [] }
    graph.AddFact(SimilarTo(a, b, 1.0)) |> ignore
    
    // Community 2: C <-> D
    let c = ConceptE { Name = "C"; Description = ""; RelatedConcepts = [] }
    let d = ConceptE { Name = "D"; Description = ""; RelatedConcepts = [] }
    graph.AddFact(SimilarTo(c, d, 1.0)) |> ignore
    
    let communities = labelPropagation graph 10 DateTime.UtcNow
    
    let idA = TarsEntity.getId a
    let idB = TarsEntity.getId b
    let idC = TarsEntity.getId c
    let idD = TarsEntity.getId d
    
    printfn "Community A: %s" communities.[idA]
    printfn "Community B: %s" communities.[idB]
    printfn "Community C: %s" communities.[idC]
    printfn "Community D: %s" communities.[idD]
    
    let success = 
        communities.[idA] = communities.[idB] &&
        communities.[idC] = communities.[idD] &&
        communities.[idA] <> communities.[idC]
        
    if success then
        printfn "SUCCESS: Communities detected correctly."
        0
    else
        printfn "FAILURE: Community detection failed."
        1
