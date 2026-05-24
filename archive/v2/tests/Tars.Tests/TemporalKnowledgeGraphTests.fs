namespace Tars.Tests

open System
open Xunit
open Tars.Core
open Tars.Core.TemporalKnowledgeGraph

module TemporalKnowledgeGraphTests =

    [<Fact>]
    let ``Adding a fact adds implicit nodes`` () =
        let graph = TemporalGraph()
        let e1 = ConceptE { Name = "Time"; Description = "Dimension"; RelatedConcepts = [] }
        let e2 = ConceptE { Name = "Space"; Description = "Dimension"; RelatedConcepts = [] }
        let fact = SimilarTo(e1, e2, 0.9)

        let edgeId = graph.AddFact(fact)
        
        let facts = graph.GetCurrentFacts()
        Assert.Single(facts) |> ignore
        Assert.Equal(fact, facts.Head)

    [<Fact>]
    let ``EvolvedFrom invalidates the source entity`` () =
        let graph = TemporalGraph()
        
        // Initial belief
        let belief1 = AgentBeliefE { 
            Statement = "Sky is green"
            Confidence = 0.8
            DerivedFrom = []
            AgentId = "agent1"
            ValidFrom = DateTime.UtcNow.AddHours(-1.0)
            InvalidAt = None 
        }
        
        // Add belief to graph (via a fact, or directly)
        // Let's say it belongs to a community
        let fact1 = BelongsTo(belief1, "community1")
        graph.AddFact(fact1) |> ignore
        
        // Verify it's valid
        let facts1 = graph.GetCurrentFacts()
        Assert.Single(facts1) |> ignore
        
        // New belief evolves from old one
        let belief2 = AgentBeliefE { 
            Statement = "Sky is blue"
            Confidence = 0.99
            DerivedFrom = []
            AgentId = "agent1"
            ValidFrom = DateTime.UtcNow
            InvalidAt = None 
        }
        
        let evolutionFact = EvolvedFrom(belief2, belief1, "Correction")
        graph.AddFact(evolutionFact) |> ignore
        
        // Verify invalidation
        // The implementation of ProcessInvalidation invalidates the TARGET of EvolvedFrom (which is belief1)
        
        // Check if belief1 node is invalid
        let belief1Id = TarsEntity.getId belief1
        let belief1Node = graph.GetNode(belief1Id)
        Assert.True(belief1Node.IsSome)
        Assert.True(belief1Node.Value.Validity.InvalidAt.IsSome)
        
        // Check if snapshot excludes the old fact because the node is invalid
        let facts2 = graph.GetCurrentFacts()
        // Should contain the new evolution fact, but NOT the old 'fact1' (BelongsTo)
        // because fact1's source (belief1) is now invalid.
        
        Assert.Contains(evolutionFact, facts2)
        Assert.DoesNotContain(fact1, facts2)
