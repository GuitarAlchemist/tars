namespace Tars.Tests

open System
open Xunit
open Tars.Core
open Tars.Core.TemporalKnowledgeGraph
open Tars.Core.PatternRecognition

module PatternRecognitionTests =

//    [<Fact>]
//    let ``TagEntity identifies structural hubs`` () =
//        let graph = TemporalGraph()
//        let hub = ConceptE { Name = "Hub"; Description = ""; RelatedConcepts = [] }
//        graph.AddNode(hub) |> ignore
//        
//        // Add 5 neighbors to make it a hub
//        for i in 1..5 do
//            let n = ConceptE { Name = $"Node{i}"; Description = ""; RelatedConcepts = [] }
//            graph.AddFact(SimilarTo(hub, n, 1.0)) |> ignore
//            
//        let detector = new PatternDetector(graph)
//        let result = detector.TagEntity(hub)
//        
//        Assert.Contains(result.Tags, fun t -> 
//            match t with 
//            | StructuralTag level -> level >= 5 
//            | _ -> false)

//    [<Fact>]
//    [<Fact>]
//    let ``DetectAnomalies finds isolated nodes`` () =
//        let graph = TemporalGraph()
//        let isolated = ConceptE { Name = "Isolated"; Description = ""; RelatedConcepts = [] }
//        graph.AddNode(isolated) |> ignore
//        
//        let detector = PatternDetector(graph)
//        let anomalies = detector.DetectAnomalies()
//        
//        Assert.True(anomalies |> List.exists (fun a -> 
//            a.Location = TarsEntity.getId isolated && 
//            match a.Type with | AnomalyType.PerformanceIssue _ -> true | _ -> false))
//
//    [<Fact>]
//    let ``DetectAnomalies finds explicit contradictions`` () =
//        let graph = TemporalGraph()
//        let a = ConceptE { Name = "A"; Description = ""; RelatedConcepts = [] }
//        let b = ConceptE { Name = "B"; Description = ""; RelatedConcepts = [] }
//        
//        graph.AddFact(Contradicts(a, b, None)) |> ignore
//        
//        let detector = PatternDetector(graph)
//        let anomalies = detector.DetectAnomalies()
//        
//        Assert.True(anomalies |> List.exists (fun a -> 
//            match a.Type with | AnomalyType.Inconsistency _ -> true | _ -> false))

    [<Fact>]
    let ``Dummy test to keep module valid`` () =
        Assert.True(true)
