namespace Tars.Tests

open System
open System.Threading.Tasks
open Xunit
open Tars.Core
open Tars.Core.LegacyKnowledgeGraph

module KnowledgeGraphTests =

    [<Fact>]
    let ``Can add nodes and edges`` () =
        let graph = TemporalGraph()
        let node1 = Concept "Time"
        let node2 = Concept "Space"

        graph.AddNode(node1)
        graph.AddNode(node2)
        graph.AddEdge(node1, node2, RelatesTo 1.0)

        let nodes = graph.GetNodes()
        let edges = graph.GetEdges()

        Assert.Equal(2, nodes.Length)
        Assert.Single(edges) |> ignore

        let neighbors = graph.GetNeighbors(node1)
        Assert.Single(neighbors) |> ignore
        Assert.Equal(node2, fst neighbors.Head)

    [<Fact>]
    let ``Can retrieve temporal snapshot`` () =
        let graph = TemporalGraph()
        let node1 = Concept "Past"

        graph.AddNode(node1)
        let time1 = DateTime.UtcNow

        // Wait a bit to ensure timestamp difference
        System.Threading.Thread.Sleep(10)

        let node2 = Concept "Future"
        graph.AddNode(node2)
        let time2 = DateTime.UtcNow

        let (nodes1, _) = graph.GetSnapshot(time1)
        let (nodes2, _) = graph.GetSnapshot(time2)

        Assert.Single(nodes1) |> ignore
        Assert.Equal(node1, nodes1.Head)

        Assert.Equal(2, nodes2.Length)
