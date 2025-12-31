namespace Tars.Tests

open System
open Xunit
open Tars.Core
open Tars.Cortex

module KTheoryTests =

    // Helper to create a test agent with a deterministic ID based on a seed string
    let createAgent (seed: string) =
        // Use a simple deterministic GUID generation (not secure, but fine for tests)
        let bytes = Array.create 16 0uy
        let seedBytes = System.Text.Encoding.UTF8.GetBytes(seed)
        Array.Copy(seedBytes, bytes, Math.Min(seedBytes.Length, 16))
        let guid = new Guid(bytes)

        { Id = AgentId guid
          Name = seed
          Version = "1.0"
          ParentVersion = None
          CreatedAt = DateTime.UtcNow
          Model = "test-model"
          SystemPrompt = ""
          Tools = []
          Capabilities = []
          Memory = []
          State = Idle
          Fitness = 0.5
          Drives = { Accuracy = 0.5; Speed = 0.5; Creativity = 0.5; Safety = 0.5 }
          Constitution = AgentConstitution.Create(AgentId guid, GeneralReasoning) }

    // --- K1: Cycle Detection Tests ---

    [<Fact>]
    let ``K1: No cycles in empty graph`` () =
        let agents = []
        let edges = []
        let result = GraphAnalyzer.analyzeGraph agents edges
        Assert.False(result.HasCycles)
        Assert.Equal(0, result.CycleCount)

    [<Fact>]
    let ``K1: No cycles in single node graph`` () =
        let agentA = createAgent "AgentA"
        let agents = [ agentA ]
        let edges = []
        let result = GraphAnalyzer.analyzeGraph agents edges
        Assert.False(result.HasCycles)
        Assert.Equal(0, result.CycleCount)

    [<Fact>]
    let ``K1: Detects self-loop`` () =
        let agentA = createAgent "AgentA"
        let agents = [ agentA ]
        let edges = [ (agentA.Id, agentA.Id) ]
        let result = GraphAnalyzer.analyzeGraph agents edges
        Assert.True(result.HasCycles)
        Assert.Equal(1, result.CycleCount)

    [<Fact>]
    let ``K1: Detects simple 2-cycle`` () =
        let agentA = createAgent "AgentA"
        let agentB = createAgent "AgentB"
        let agents = [ agentA; agentB ]
        let edges = [ (agentA.Id, agentB.Id); (agentB.Id, agentA.Id) ]
        let result = GraphAnalyzer.analyzeGraph agents edges
        Assert.True(result.HasCycles)
        Assert.Equal(1, result.CycleCount)

    [<Fact>]
    let ``K1: No cycles in linear chain`` () =
        let agentA = createAgent "AgentA"
        let agentB = createAgent "AgentB"
        let agentC = createAgent "AgentC"
        let agents = [ agentA; agentB; agentC ]
        let edges = [ (agentA.Id, agentB.Id); (agentB.Id, agentC.Id) ]
        let result = GraphAnalyzer.analyzeGraph agents edges
        Assert.False(result.HasCycles)
        Assert.Equal(0, result.CycleCount)

    [<Fact>]
    let ``K1: Detects 3-cycle`` () =
        let agentA = createAgent "AgentA"
        let agentB = createAgent "AgentB"
        let agentC = createAgent "AgentC"
        let agents = [ agentA; agentB; agentC ]

        let edges =
            [ (agentA.Id, agentB.Id); (agentB.Id, agentC.Id); (agentC.Id, agentA.Id) ]

        let result = GraphAnalyzer.analyzeGraph agents edges
        Assert.True(result.HasCycles)
        Assert.Equal(1, result.CycleCount)

    [<Fact>]
    let ``K1: Detects disjoint cycles`` () =
        let agentA = createAgent "AgentA"
        let agentB = createAgent "AgentB"
        let agentC = createAgent "AgentC"
        let agentD = createAgent "D"
        let agents = [ agentA; agentB; agentC; agentD ]
        // Cycle 1: A <-> B
        // Cycle 2: C <-> D
        let edges =
            [ (agentA.Id, agentB.Id)
              (agentB.Id, agentA.Id)
              (agentC.Id, agentD.Id)
              (agentD.Id, agentC.Id) ]

        let result = GraphAnalyzer.analyzeGraph agents edges
        Assert.True(result.HasCycles)
        Assert.Equal(2, result.CycleCount)

    [<Fact>]
    let ``K1: Detects complex figure-8 cycle`` () =
        // A -> B -> C -> A (Cycle 1)
        // C -> D -> E -> C (Cycle 2)
        let agentA = createAgent "A"
        let agentB = createAgent "B"
        let agentC = createAgent "C"
        let agentD = createAgent "D"
        let agentE = createAgent "E"

        let agents = [ agentA; agentB; agentC; agentD; agentE ]

        let edges =
            [ (agentA.Id, agentB.Id)
              (agentB.Id, agentC.Id)
              (agentC.Id, agentA.Id)
              (agentC.Id, agentD.Id)
              (agentD.Id, agentE.Id)
              (agentE.Id, agentC.Id) ]

        let result = GraphAnalyzer.analyzeGraph agents edges
        Assert.True(result.HasCycles)
        Assert.Equal(2, result.CycleCount)

    [<Fact>]
    let ``K1: No cycles in large DAG`` () =
        // Binary tree structure (DAG)
        //       A
        //     /   \
        //    B     C
        //   / \   / \
        //  D   E F   G
        let agentA = createAgent "A"
        let agentB = createAgent "B"
        let agentC = createAgent "C"
        let agentD = createAgent "D"
        let agentE = createAgent "E"
        let agentF = createAgent "F"
        let agentG = createAgent "G"

        let agents = [ agentA; agentB; agentC; agentD; agentE; agentF; agentG ]

        let edges =
            [ (agentA.Id, agentB.Id)
              (agentA.Id, agentC.Id)
              (agentB.Id, agentD.Id)
              (agentB.Id, agentE.Id)
              (agentC.Id, agentF.Id)
              (agentC.Id, agentG.Id) ]

        let result = GraphAnalyzer.analyzeGraph agents edges
        Assert.False(result.HasCycles)
        Assert.Equal(0, result.CycleCount)

    // --- K0: Resource Conservation Tests ---

    [<Fact>]
    let ``K0: Cost Monoid Associativity`` () =
        let a =
            { Cost.Zero with
                Tokens = 10<token>
                Money = 1m<usd> }

        let b =
            { Cost.Zero with
                Tokens = 20<token>
                Duration = 5.0<ms> }

        let c =
            { Cost.Zero with
                Tokens = 5<token>
                CallCount = 1<requests> }

        let left = (a + b) + c
        let right = a + (b + c)

        Assert.Equal(left, right)

    [<Fact>]
    let ``K0: Cost Monoid Identity`` () =
        let a =
            { Cost.Zero with
                Tokens = 10<token>
                Money = 1m<usd> }

        Assert.Equal(a, a + Cost.Zero)
        Assert.Equal(a, Cost.Zero + a)

    [<Fact>]
    let ``K0: Conservation Law (Remaining + Consumed = Total)`` () =
        let totalTokens = 100<token>

        let budget =
            { Budget.Infinite with
                MaxTokens = Some totalTokens }

        let governor = BudgetGovernor(budget)

        let consumeAmount = 30<token>

        let _ =
            governor.Consume(
                { Cost.Zero with
                    Tokens = consumeAmount }
            )

        let remaining = governor.Remaining.MaxTokens
        let consumed = governor.Consumed.Tokens

        // Invariant: Remaining + Consumed = Total (for bounded resources)
        Assert.Equal(Some totalTokens, remaining |> Option.map (fun r -> r + consumed))

    [<Fact>]
    let ``K0: Conservation Law (Mixed Resources)`` () =
        let budget =
            { Budget.Infinite with
                MaxTokens = Some 1000<token>
                MaxRam = Some 1024L<bytes>
                MaxNodes = Some 50<nodes> }

        let governor = BudgetGovernor(budget)

        let cost1 =
            { Cost.Zero with
                Tokens = 100<token>
                Ram = 256L<bytes> }

        let cost2 =
            { Cost.Zero with
                Tokens = 50<token>
                Nodes = 5<nodes> }

        let _ = governor.Consume(cost1)
        let _ = governor.Consume(cost2)

        let remaining = governor.Remaining
        let consumed = governor.Consumed

        // Tokens: 1000 - (100 + 50) = 850
        Assert.Equal(Some 850<token>, remaining.MaxTokens)
        Assert.Equal(150<token>, consumed.Tokens)

        // RAM: 1024 - 256 = 768
        Assert.Equal(Some 768L<bytes>, remaining.MaxRam)
        Assert.Equal(256L<bytes>, consumed.Ram)

        // Nodes: 50 - 5 = 45
        Assert.Equal(Some 45<nodes>, remaining.MaxNodes)
        Assert.Equal(5<nodes>, consumed.Nodes)

    [<Fact>]
    let ``K0: Allocation Conservation (Parent - Child = NewParent)`` () =
        let parentTokens = 100<token>

        let parentBudget =
            { Budget.Infinite with
                MaxTokens = Some parentTokens }

        let parentGovernor = BudgetGovernor(parentBudget)

        let childTokens = 40<token>

        let childBudget =
            { Budget.Infinite with
                MaxTokens = Some childTokens }

        // Allocate to child
        let childGovernorResult = parentGovernor.Allocate(childBudget)

        match childGovernorResult with
        | Result.Ok _ ->
            let parentRemaining = parentGovernor.Remaining.MaxTokens.Value

            // Invariant: ParentRemaining + ChildTotal = ParentTotal
            Assert.Equal(parentTokens, parentRemaining + childTokens)
        | Result.Error e -> Assert.Fail($"Allocation failed: {e}")

    // === Edge Case Tests for Budget ===

    [<Fact>]
    let ``Budget: TryConsume fails when over budget`` () =
        let budget =
            { Budget.Infinite with
                MaxTokens = Some 50<token> }

        let governor = BudgetGovernor(budget)

        let result =
            governor.TryConsume(
                { Cost.Zero with
                    Tokens = 100<token> }
            )

        match result with
        | Result.Error _ -> () // Expected
        | Result.Ok _ -> Assert.Fail("Should fail when cost exceeds budget")

        Assert.Equal(0<token>, governor.Consumed.Tokens)

    [<Fact>]
    let ``Budget: TryConsume succeeds when under budget`` () =
        let budget =
            { Budget.Infinite with
                MaxTokens = Some 100<token> }

        let governor = BudgetGovernor(budget)

        let result =
            governor.TryConsume(
                { Cost.Zero with
                    Tokens = 50<token> }
            )

        match result with
        | Result.Ok _ -> ()
        | Result.Error e -> Assert.Fail($"Should succeed when cost is under budget: {e}")

        Assert.Equal(50<token>, governor.Consumed.Tokens)

    [<Fact>]
    let ``Budget: CanAfford does not consume`` () =
        let budget =
            { Budget.Infinite with
                MaxTokens = Some 100<token> }

        let governor = BudgetGovernor(budget)

        let canAfford =
            governor.CanAfford(
                { Cost.Zero with
                    Tokens = 50<token> }
            )

        Assert.True(canAfford)
        Assert.Equal(0<token>, governor.Consumed.Tokens) // Nothing consumed

    [<Fact>]
    let ``Budget: IsCritical detects low resources`` () =
        let budget =
            { Budget.Infinite with
                MaxTokens = Some 100<token> }

        let governor = BudgetGovernor(budget)

        // Consume 95%, remaining is 5%
        governor.Consume({ Cost.Zero with Tokens = 95<token> }) |> ignore

        // IsCritical(percentage) checks if remaining/total < percentage
        // At 5% remaining: 0.05 < 0.10 is true, 0.05 < 0.03 is false
        Assert.True(governor.IsCritical(0.10), "Should be critical when remaining (5%) < threshold (10%)")
        Assert.False(governor.IsCritical(0.03), "Should not be critical when remaining (5%) >= threshold (3%)")

    [<Fact>]
    let ``Budget: Infinite budget always affordable`` () =
        let governor = BudgetGovernor(Budget.Infinite)

        let hugeCost =
            { Cost.Zero with
                Tokens = 1000000<token>
                Money = 10000m<usd>
                Ram = 1000000000L<bytes> }

        Assert.True(governor.CanAfford(hugeCost))

        match governor.TryConsume(hugeCost) with
        | Result.Ok _ -> ()
        | Result.Error e -> Assert.Fail($"Infinite budget should always succeed: {e}")

    [<Fact>]
    let ``Budget: Allocation fails when insufficient`` () =
        let budget =
            { Budget.Infinite with
                MaxTokens = Some 50<token> }

        let parentGovernor = BudgetGovernor(budget)

        let childBudget =
            { Budget.Infinite with
                MaxTokens = Some 100<token> } // More than parent has

        let result = parentGovernor.Allocate(childBudget)

        match result with
        | Result.Error _ -> () // Expected
        | Result.Ok _ -> Assert.Fail("Allocation should fail when child budget exceeds parent")

    [<Fact>]
    let ``Budget: Multiple dimensions checked independently`` () =
        let budget =
            { Budget.Infinite with
                MaxTokens = Some 100<token>
                MaxCalls = Some 5<requests> }

        let governor = BudgetGovernor(budget)

        // Use up all calls but not tokens
        for _ in 1..5 do
            governor.Consume({ Cost.Zero with CallCount = 1<requests> }) |> ignore

        // Should fail even though tokens available
        let result =
            governor.TryConsume(
                { Cost.Zero with
                    Tokens = 10<token>
                    CallCount = 1<requests> }
            )

        match result with
        | Result.Error _ -> () // Expected
        | Result.Ok _ -> Assert.Fail("Should fail when any dimension is over budget")

    // === Edge Case Tests for Graph Analysis ===

    [<Fact>]
    let ``K1: Handles duplicate edges`` () =
        let agentA = createAgent "A"
        let agentB = createAgent "B"
        let agents = [ agentA; agentB ]

        // Same edge twice
        let edges =
            [ (agentA.Id, agentB.Id)
              (agentA.Id, agentB.Id)
              (agentB.Id, agentA.Id) ]

        let result = GraphAnalyzer.analyzeGraph agents edges
        Assert.True(result.HasCycles)

    [<Fact>]
    let ``K1: Handles edges with unknown nodes`` () =
        let agentA = createAgent "A"
        let agentB = createAgent "B"
        let unknownId = AgentId(Guid.NewGuid())
        let agents = [ agentA; agentB ]

        // Edge to unknown node should be ignored
        let edges = [ (agentA.Id, unknownId); (agentA.Id, agentB.Id) ]

        let result = GraphAnalyzer.analyzeGraph agents edges
        Assert.False(result.HasCycles) // Just A -> B, no cycle

    [<Fact>]
    let ``K1: Large star graph has no cycles`` () =
        // Hub with many spokes
        let hub = createAgent "Hub"
        let spokes = [ for i in 1..20 -> createAgent $"Spoke{i}" ]
        let agents = hub :: spokes

        let edges = spokes |> List.map (fun s -> (hub.Id, s.Id))

        let result = GraphAnalyzer.analyzeGraph agents edges
        Assert.False(result.HasCycles)
        Assert.Equal(0, result.CycleCount)
