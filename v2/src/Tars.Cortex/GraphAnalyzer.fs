namespace Tars.Cortex

open System
open Tars.Core

module GraphAnalyzer =

    /// Represents a directed graph as an adjacency matrix
    type AdjacencyMatrix =
        { NodeIndex: Map<AgentId, int>
          Matrix: float[,] }

    /// Tools for K-Theory analysis of the agent graph
    module KTheory =

        /// Computes the Kernel (Null Space) of a matrix using Gaussian Elimination
        /// Returns the dimension of the kernel (Nullity) and basis vectors
        let computeKernel (matrix: float[,]) =
            let rows = Array2D.length1 matrix
            let cols = Array2D.length2 matrix
            let eps = 1e-10

            // Working copy
            let m = Array2D.copy matrix

            // Gaussian elimination to Row Echelon Form
            let mutable pivotRow = 0
            let mutable col = 0

            while pivotRow < rows && col < cols do
                // Find pivot
                let mutable maxRow = pivotRow

                for i in pivotRow + 1 .. rows - 1 do
                    if abs m.[i, col] > abs m.[maxRow, col] then
                        maxRow <- i

                if abs m.[maxRow, col] < eps then
                    col <- col + 1
                else
                    // Swap rows
                    for j in col .. cols - 1 do
                        let temp = m.[pivotRow, j]
                        m.[pivotRow, j] <- m.[maxRow, j]
                        m.[maxRow, j] <- temp

                    // Normalize pivot row
                    let pivotVal = m.[pivotRow, col]

                    for j in col .. cols - 1 do
                        m.[pivotRow, j] <- m.[pivotRow, j] / pivotVal

                    // Eliminate other rows
                    for i in 0 .. rows - 1 do
                        if i <> pivotRow then
                            let factor = m.[i, col]

                            for j in col .. cols - 1 do
                                m.[i, j] <- m.[i, j] - factor * m.[pivotRow, j]

                    pivotRow <- pivotRow + 1
                    col <- col + 1

            // Count non-zero rows (Rank)
            let mutable rank = 0

            for i in 0 .. rows - 1 do
                let mutable isZero = true

                for j in 0 .. cols - 1 do
                    if abs m.[i, j] > eps then
                        isZero <- false

                if not isZero then
                    rank <- rank + 1

            // Nullity = Cols - Rank (Rank-Nullity Theorem)
            let nullity = cols - rank

            // For K1, we just need to know if Nullity > 0 (existence of cycles)
            // In a flow network, Nullity corresponds to the number of independent cycles.
            nullity

        /// Analyzes the graph for K1 invariants (cycles)
        /// Returns the number of independent cycles using the cyclomatic complexity formula.
        /// For a graph, the number of independent cycles = edges - vertices + connected_components
        /// This corresponds to the first Betti number in algebraic topology.
        let detectCycles (agentIds: AgentId list) (edges: (AgentId * AgentId) list) =
            let n = agentIds.Length

            if n = 0 then
                0
            else
                let indexMap = agentIds |> List.mapi (fun i id -> id, i) |> Map.ofList

                // Filter to valid edges only
                let validEdges =
                    edges
                    |> List.filter (fun (u, v) -> Map.containsKey u indexMap && Map.containsKey v indexMap)

                let m = validEdges.Length

                // Build adjacency list (treating graph as undirected for component counting)
                let adj = Array.init n (fun _ -> ResizeArray<int>())
                for (u, v) in validEdges do
                    let i = indexMap.[u]
                    let j = indexMap.[v]
                    adj.[i].Add(j)
                    adj.[j].Add(i) // Undirected for connectivity

                // Count connected components using BFS
                let visited = Array.create n false
                let mutable components = 0

                for start in 0 .. n - 1 do
                    if not visited.[start] then
                        components <- components + 1
                        let queue = System.Collections.Generic.Queue<int>()
                        queue.Enqueue(start)
                        visited.[start] <- true
                        while queue.Count > 0 do
                            let current = queue.Dequeue()
                            for neighbor in adj.[current] do
                                if not visited.[neighbor] then
                                    visited.[neighbor] <- true
                                    queue.Enqueue(neighbor)

                // Cyclomatic complexity: M = E - V + C
                // Where E = edges, V = vertices, C = connected components
                // This gives the number of independent cycles
                max 0 (m - n + components)

    type CycleDetectionResult =
        { HasCycles: bool
          CycleCount: int
          Message: string }

    let analyzeGraph (agents: Agent list) (interactions: (AgentId * AgentId) list) =
        let ids = agents |> List.map (fun a -> a.Id)
        let k1 = KTheory.detectCycles ids interactions

        { HasCycles = k1 > 0
          CycleCount = k1
          Message =
            if k1 > 0 then
                $"Detected {k1} independent cycle(s) in agent graph (K1 Invariant violation)."
            else
                "No cycles detected." }
