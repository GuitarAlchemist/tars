namespace Tars.Tests

open System
open Xunit
open Tars.Core
open Tars.Cortex

type KnowledgeGraphTests() =

    [<Fact>]
    member _.``Can add nodes and edges``() =
        let graph = KnowledgeGraph()
        let nodeA = Concept "A"
        let nodeB = Concept "B"
        let edge = RelatesTo 1.0

        graph.AddEdge(nodeA, nodeB, edge)

        let neighbors = graph.GetNeighbors(nodeA)
        Assert.Single(neighbors) |> ignore
        Assert.Equal(nodeB, fst neighbors.Head)

    [<Fact>]
    member _.``Can find path``() =
        let graph = KnowledgeGraph()
        let nodeA = Concept "A"
        let nodeB = Concept "B"
        let nodeC = Concept "C"

        graph.AddEdge(nodeA, nodeB, RelatesTo 1.0)
        graph.AddEdge(nodeB, nodeC, RelatesTo 1.0)

        let path = graph.FindPath(nodeA, nodeC)

        Assert.True(path.IsSome)
        Assert.Equal(2, path.Value.Length) // A->B, B->C
