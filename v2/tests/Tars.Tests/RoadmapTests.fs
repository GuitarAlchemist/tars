namespace Tars.Tests

open System
open System.Threading
open System.Threading.Tasks
open Xunit
open Tars.Core

/// Tests for Phase 2.5.2, 6.1, 6.7.2, 6.7.3 roadmap items
module RoadmapTests =

    // =========================================================================
    // BeliefGraph Tests (Phase 2.5.2)
    // =========================================================================

    module BeliefGraphTests =

        let createTestBelief statement status =
            { Id = Guid.NewGuid()
              Statement = statement
              Context = "test"
              Status = status
              Confidence = 0.8
              DerivedFrom = []
              CreatedAt = DateTime.UtcNow
              LastVerified = DateTime.UtcNow }

        [<Fact>]
        let ``BeliefGraph: Empty graph has no beliefs`` () =
            let graph = BeliefGraph.empty ()
            Assert.Empty(graph.Beliefs)
            Assert.Empty(graph.Edges)

        [<Fact>]
        let ``BeliefGraph: Add belief stores it`` () =
            let belief = createTestBelief "Test belief" Hypothesis
            let graph = BeliefGraph.empty () |> BeliefGraph.addBelief belief

            Assert.Equal<int>(1, graph.Beliefs.Count)
            Assert.True(graph.Beliefs.ContainsKey(belief.Id))

        [<Fact>]
        let ``BeliefGraph: Remove belief removes it`` () =
            let belief = createTestBelief "Test belief" Hypothesis

            let graph =
                BeliefGraph.empty ()
                |> BeliefGraph.addBelief belief
                |> BeliefGraph.removeBelief belief.Id

            Assert.Empty(graph.Beliefs)

        [<Fact>]
        let ``BeliefGraph: Add edge creates relationship`` () =
            let b1 = createTestBelief "Belief 1" Hypothesis
            let b2 = createTestBelief "Belief 2" Hypothesis

            let graph =
                BeliefGraph.empty ()
                |> BeliefGraph.addBelief b1
                |> BeliefGraph.addBelief b2
                |> BeliefGraph.addEdge b1.Id b2.Id (SupportsBy 0.9)

            Assert.Single(graph.Edges) |> ignore
            Assert.Equal(b1.Id, graph.Edges.Head.SourceId)
            Assert.Equal(b2.Id, graph.Edges.Head.TargetId)

        [<Fact>]
        let ``BeliefGraph: Get beliefs by status`` () =
            let h1 = createTestBelief "Hypothesis 1" Hypothesis
            let h2 = createTestBelief "Hypothesis 2" Hypothesis
            let p1 = createTestBelief "Principle 1" UniversalPrinciple

            let graph =
                BeliefGraph.empty ()
                |> BeliefGraph.addBelief h1
                |> BeliefGraph.addBelief h2
                |> BeliefGraph.addBelief p1

            let hypotheses = BeliefGraph.getBeliefsByStatus Hypothesis graph
            let principles = BeliefGraph.getBeliefsByStatus UniversalPrinciple graph

            Assert.Equal<int>(2, hypotheses.Length)
            Assert.Single(principles) |> ignore

        [<Fact>]
        let ``BeliefGraph: Update status changes status`` () =
            let belief = createTestBelief "Test" Hypothesis

            let graph =
                BeliefGraph.empty ()
                |> BeliefGraph.addBelief belief
                |> BeliefGraph.updateStatus belief.Id VerifiedFact

            let updated = BeliefGraph.tryGetBelief belief.Id graph
            Assert.True(updated.IsSome)
            Assert.Equal<EpistemicStatus>(VerifiedFact, updated.Value.Status)

        [<Fact>]
        let ``BeliefGraph: Count by status returns correct counts`` () =
            let h1 = createTestBelief "H1" Hypothesis
            let h2 = createTestBelief "H2" Hypothesis
            let p1 = createTestBelief "P1" UniversalPrinciple

            let graph =
                BeliefGraph.empty ()
                |> BeliefGraph.addBelief h1
                |> BeliefGraph.addBelief h2
                |> BeliefGraph.addBelief p1

            let counts = BeliefGraph.countByStatus graph
            Assert.Equal<int>(2, counts.[Hypothesis])
            Assert.Equal<int>(1, counts.[UniversalPrinciple])

        [<Fact>]
        let ``BeliefGraph: Get supporting beliefs finds supporters`` () =
            let b1 = createTestBelief "Support" Hypothesis
            let b2 = createTestBelief "Main" Hypothesis

            let graph =
                BeliefGraph.empty ()
                |> BeliefGraph.addBelief b1
                |> BeliefGraph.addBelief b2
                |> BeliefGraph.addEdge b1.Id b2.Id (SupportsBy 0.9)

            let supporting = BeliefGraph.getSupportingBeliefs b2.Id graph
            Assert.Single(supporting) |> ignore
            Assert.Equal(b1.Id, supporting.Head.Id)

    // =========================================================================
    // BoundedChannel Tests (Phase 6.1)
    // =========================================================================

    module BoundedChannelTests =

        [<Fact>]
        let ``BoundedChannel: Write within capacity succeeds`` () =
            use channel = new BoundedChannel<int>(5, RejectNew)

            let result = channel.TryWrite(1)

            match result with
            | Written -> Assert.True(true)
            | _ -> Assert.Fail("Expected Written")

        [<Fact>]
        let ``BoundedChannel: Write at capacity rejects when RejectNew`` () =
            use channel = new BoundedChannel<int>(2, RejectNew)

            channel.TryWrite(1) |> ignore
            channel.TryWrite(2) |> ignore
            let result = channel.TryWrite(3)

            match result with
            | Rejected _ -> Assert.True(true)
            | _ -> Assert.Fail("Expected Rejected")

        [<Fact>]
        let ``BoundedChannel: Write at capacity drops oldest when DropOldest`` () =
            use channel = new BoundedChannel<int>(2, DropOldest)

            channel.TryWrite(1) |> ignore
            channel.TryWrite(2) |> ignore
            let result = channel.TryWrite(3)

            match result with
            | Dropped _ -> Assert.True(true)
            | _ -> Assert.Fail("Expected Dropped")

        [<Fact>]
        let ``BoundedChannel: Read returns items in order`` () =
            use channel = new BoundedChannel<int>(5, RejectNew)

            channel.TryWrite(1) |> ignore
            channel.TryWrite(2) |> ignore
            channel.TryWrite(3) |> ignore

            Assert.Equal<int option>(Some 1, channel.TryRead())
            Assert.Equal<int option>(Some 2, channel.TryRead())
            Assert.Equal<int option>(Some 3, channel.TryRead())
            Assert.Equal<int option>(None, channel.TryRead())

        [<Fact>]
        let ``BoundedChannel: Stats track operations`` () =
            use channel = new BoundedChannel<int>(3, RejectNew)

            channel.TryWrite(1) |> ignore
            channel.TryWrite(2) |> ignore
            channel.TryRead() |> ignore
            channel.TryWrite(3) |> ignore
            channel.TryWrite(4) |> ignore
            channel.TryWrite(5) |> ignore // Should be rejected

            let stats = channel.Stats
            Assert.Equal<int64>(4L, stats.TotalWrites) // 4 successful writes
            Assert.Equal<int64>(1L, stats.TotalReads)
            Assert.Equal<int64>(1L, stats.TotalRejections)

        [<Fact>]
        let ``BoundedChannel: DrainAll returns all items`` () =
            use channel = new BoundedChannel<int>(10, RejectNew)

            for i in 1..5 do
                channel.TryWrite(i) |> ignore

            let items = channel.DrainAll()
            Assert.Equal<int>(5, items.Length)
            Assert.Equal<int list>([ 1; 2; 3; 4; 5 ], items)
            Assert.True(channel.IsEmpty)

        [<Fact>]
        let ``BoundedChannel: Async read blocks until item available`` () =
            task {
                use channel = new BoundedChannel<int>(5, RejectNew)

                // Write immediately
                channel.TryWrite(42) |> ignore

                // Read should succeed (item already available)
                let! result = channel.ReadAsync(TimeSpan.FromMilliseconds(1000.0))
                Assert.Equal<int option>(Some 42, result)
            }

    // =========================================================================
    // Gate Tests (Phase 6.7.3)
    // =========================================================================

    module GateTests =

        [<Fact>]
        let ``Gate: Passes when condition is true`` () =
            task {
                let gate = Gate.create "TestGate" (fun () -> Task.FromResult(true))
                let! result = Gate.tryPassAsync gate

                match result with
                | Passed -> Assert.True(true)
                | _ -> Assert.Fail("Expected Passed")
            }

        [<Fact>]
        let ``Gate: Blocks when condition is false`` () =
            task {
                let gate = Gate.create "TestGate" (fun () -> Task.FromResult(false))
                let! result = Gate.tryPassAsync gate

                match result with
                | Blocked _ -> Assert.True(true)
                | _ -> Assert.Fail("Expected Blocked")
            }

        [<Fact>]
        let ``MutexGate: Limits concurrent access`` () =
            task {
                use mutex = new MutexGate(2)

                Assert.Equal(2, mutex.Available)
                Assert.True(mutex.TryAcquire())
                Assert.Equal(1, mutex.Available)
                Assert.True(mutex.TryAcquire())
                Assert.Equal(0, mutex.Available)
                Assert.False(mutex.TryAcquire())

                mutex.Release()
                Assert.Equal(1, mutex.Available)
            }

        [<Fact>]
        let ``MutexGate: WithLockAsync executes action`` () =
            task {
                use mutex = new MutexGate(1)
                let mutable executed = false

                do!
                    mutex.WithLockAsync(fun () ->
                        task {
                            executed <- true
                            return ()
                        })

                Assert.True(executed)
                Assert.Equal(1, mutex.Available) // Released after action
            }

        [<Fact>]
        let ``MutexGate: Tracks acquire and release counts`` () =
            task {
                use mutex = new MutexGate(3)

                mutex.TryAcquire() |> ignore
                mutex.TryAcquire() |> ignore
                mutex.Release()
                mutex.TryAcquire() |> ignore

                Assert.Equal(3, mutex.TotalAcquires)
                Assert.Equal(1, mutex.TotalReleases)
            }

        [<Fact>]
        let ``JoinGate: Opens after required signals`` () =
            task {
                let gate = JoinGate(3)

                Assert.False(gate.IsOpen)
                gate.Signal()
                Assert.False(gate.IsOpen)
                gate.Signal()
                Assert.False(gate.IsOpen)
                gate.Signal()
                Assert.True(gate.IsOpen)
            }

        [<Fact>]
        let ``JoinGate: WaitAsync completes when gate opens`` () =
            task {
                let gate = JoinGate(2)

                // Signal in background
                let signalTask =
                    task {
                        do! Task.Delay(50)
                        gate.Signal()
                        gate.Signal()
                    }

                let! opened = gate.WaitAsync(TimeSpan.FromSeconds(1.0))
                Assert.True(opened)
                do! signalTask
            }

        [<Fact>]
        let ``JoinGate: WaitAsync times out if not enough signals`` () =
            task {
                let gate = JoinGate(5)
                gate.Signal() // Only 1 of 5 signals

                let! opened = gate.WaitAsync(TimeSpan.FromMilliseconds(100.0))
                Assert.False(opened)
            }

        [<Fact>]
        let ``JoinGate: Reset clears signal count`` () =
            let gate = JoinGate(3)

            gate.Signal()
            gate.Signal()
            Assert.Equal(2, gate.SignalCount)

            gate.Reset()
            Assert.Equal(0, gate.SignalCount)
            Assert.False(gate.IsOpen)

    // =========================================================================
    // Circuit Breaker Tests (Existing in Resilience.fs)
    // =========================================================================

    module CircuitBreakerTests =
        open Tars.Core.Resilience

        [<Fact>]
        let ``CircuitBreaker: Starts closed`` () =
            let cb = CircuitBreaker(3, TimeSpan.FromSeconds(30.0))
            Assert.Equal(Closed, cb.State)

        [<Fact>]
        let ``CircuitBreaker: Opens after threshold failures`` () =
            task {
                let cb = CircuitBreaker(2, TimeSpan.FromSeconds(30.0))

                // Trigger failures
                try
                    do! cb.ExecuteAsync(fun () -> Task.FromException<unit>(Exception("Fail 1")))
                with _ ->
                    ()

                try
                    do! cb.ExecuteAsync(fun () -> Task.FromException<unit>(Exception("Fail 2")))
                with _ ->
                    ()

                match cb.State with
                | Open _ -> Assert.True(true)
                | _ -> Assert.Fail("Expected Open state")
            }

        [<Fact>]
        let ``CircuitBreaker: Successful calls reset failure count`` () =
            task {
                let cb = CircuitBreaker(3, TimeSpan.FromSeconds(30.0))

                // One failure
                try
                    do! cb.ExecuteAsync(fun () -> Task.FromException<unit>(Exception("Fail")))
                with _ ->
                    ()

                // Successful call
                let! _ = cb.ExecuteAsync(fun () -> Task.FromResult(42))

                // Should still be closed (success reset failures)
                Assert.Equal(Closed, cb.State)
            }
