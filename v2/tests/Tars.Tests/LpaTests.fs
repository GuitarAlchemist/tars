namespace Tars.Tests

open System
open Xunit
open Tars.Core
open Tars.Core.TemporalKnowledgeGraph
open Tars.Core.CommunityDetection

module LpaTests =

    [<Fact>]
    let ``LPA detects communities in disjoint graph`` () =
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
        
        Assert.Equal(communities.[idA], communities.[idB])
        Assert.Equal(communities.[idC], communities.[idD])
        Assert.True(communities.[idA] <> communities.[idC])

    [<Fact>]
    let ``LPA handles single node`` () =
        let graph = TemporalGraph()
        let a = ConceptE { Name = "Single"; Description = ""; RelatedConcepts = [] }
        graph.AddNode(a) |> ignore
        
        let communities = labelPropagation graph 10 DateTime.UtcNow
        
        let idA = TarsEntity.getId a
        Assert.True(communities.ContainsKey idA)
        Assert.Equal(idA, communities.[idA]) // Should be its own community
