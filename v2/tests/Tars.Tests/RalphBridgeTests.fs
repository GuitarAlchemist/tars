namespace Tars.Tests

open System
open Xunit
open Tars.Cortex
open Tars.Cortex.RalphBridge
open Tars.Core.MetaCognition

/// Tests for RalphBridge (Ralph Loop integration).
module RalphBridgeTests =

    // =========================================================================
    // State parsing tests
    // =========================================================================

    [<Fact>]
    let ``parseState parses valid frontmatter`` () =
        let content = """---
active: true
iteration: 3
session_id: abc123
max_iterations: 10
completion_promise: "DONE"
started_at: "2026-03-12T10:00:00Z"
---

Fix the bugs.
"""
        let state = RalphBridge.parseState content
        Assert.True(state.Active)
        Assert.Equal(3, state.Iteration)
        Assert.Equal(Some 10, state.MaxIterations)
        Assert.Equal(Some "DONE", state.CompletionPromise)
        Assert.Equal(Some "abc123", state.SessionId)
        Assert.True(state.StartedAt.IsSome)
        Assert.Equal("Fix the bugs.", state.Prompt)

    [<Fact>]
    let ``parseState handles null optionals`` () =
        let content = """---
active: true
iteration: 1
max_iterations: null
completion_promise: null
session_id: null
---

Improve things.
"""
        let state = RalphBridge.parseState content
        Assert.True(state.Active)
        Assert.Equal(1, state.Iteration)
        Assert.Equal(None, state.MaxIterations)
        Assert.Equal(None, state.CompletionPromise)
        Assert.Equal(None, state.SessionId)
        Assert.Equal("Improve things.", state.Prompt)

    [<Fact>]
    let ``parseState handles inactive loop`` () =
        let content = """---
active: false
iteration: 5
max_iterations: 5
---

Done.
"""
        let state = RalphBridge.parseState content
        Assert.False(state.Active)
        Assert.Equal(5, state.Iteration)

    [<Fact>]
    let ``parseState extracts multiline prompt`` () =
        let content = """---
active: true
iteration: 1
---

Line one.
Line two.
Line three.
"""
        let state = RalphBridge.parseState content
        Assert.Contains("Line one.", state.Prompt)
        Assert.Contains("Line two.", state.Prompt)
        Assert.Contains("Line three.", state.Prompt)

    // =========================================================================
    // Termination logic tests
    // =========================================================================

    [<Fact>]
    let ``shouldTerminate returns true when at max`` () =
        let state =
            { RalphState.Active = true
              Iteration = 10
              MaxIterations = Some 10
              CompletionPromise = None
              SessionId = None
              StartedAt = None
              Prompt = "" }
        Assert.True(RalphBridge.shouldTerminate state)

    [<Fact>]
    let ``shouldTerminate returns false when under max`` () =
        let state =
            { RalphState.Active = true
              Iteration = 3
              MaxIterations = Some 10
              CompletionPromise = None
              SessionId = None
              StartedAt = None
              Prompt = "" }
        Assert.False(RalphBridge.shouldTerminate state)

    [<Fact>]
    let ``shouldTerminate returns false when no max`` () =
        let state =
            { RalphState.Active = true
              Iteration = 999
              MaxIterations = None
              CompletionPromise = None
              SessionId = None
              StartedAt = None
              Prompt = "" }
        Assert.False(RalphBridge.shouldTerminate state)

    // =========================================================================
    // Prompt generation tests
    // =========================================================================

    [<Fact>]
    let ``generateTarsPrompt includes gaps`` () =
        let gaps =
            [ { GapId = "gap-1"
                Domain = "search"
                Description = "Search pattern failures"
                FailureRate = 0.6
                SampleSize = 5
                Confidence = 0.8
                SuggestedRemedy = GapRemedy.LearnPattern "Improve search"
                RelatedClusters = []
                DetectedAt = DateTime.UtcNow } ]
        let prompt = RalphBridge.generateTarsPrompt gaps None
        Assert.Contains("search", prompt)
        Assert.Contains("60%", prompt)
        Assert.Contains("TARS GAPS RESOLVED", prompt)

    [<Fact>]
    let ``generateTarsPrompt includes focus area`` () =
        let prompt = RalphBridge.generateTarsPrompt [] (Some "ReAct pattern")
        Assert.Contains("ReAct pattern", prompt)
        Assert.Contains("Focus Area", prompt)

    [<Fact>]
    let ``generateTarsPrompt handles no gaps`` () =
        let prompt = RalphBridge.generateTarsPrompt [] None
        Assert.Contains("No specific gaps", prompt)

    [<Fact>]
    let ``generateTaskPrompt includes goal and promise`` () =
        let prompt = RalphBridge.generateTaskPrompt "Add caching layer" "All cache tests pass" "CACHING DONE"
        Assert.Contains("Add caching layer", prompt)
        Assert.Contains("All cache tests pass", prompt)
        Assert.Contains("CACHING DONE", prompt)

    // =========================================================================
    // File I/O tests (using temp directory)
    // =========================================================================

    [<Fact>]
    let ``startLoop and readState round-trip`` () =
        let tempDir = IO.Path.Combine(IO.Path.GetTempPath(), Guid.NewGuid().ToString("N"))
        IO.Directory.CreateDirectory(tempDir) |> ignore
        try
            let stateDir = IO.Path.Combine(tempDir, ".claude")
            IO.Directory.CreateDirectory(stateDir) |> ignore

            RalphBridge.startLoop tempDir "Fix the bugs" (Some 5) (Some "FIXED")
            let state = RalphBridge.readState tempDir
            Assert.True(state.IsSome)
            let s = state.Value
            Assert.True(s.Active)
            Assert.Equal(1, s.Iteration)
            Assert.Equal(Some 5, s.MaxIterations)
            Assert.Equal(Some "FIXED", s.CompletionPromise)
            Assert.Contains("Fix the bugs", s.Prompt)
        finally
            try IO.Directory.Delete(tempDir, true) with _ -> ()

    [<Fact>]
    let ``isActive detects active loop`` () =
        let tempDir = IO.Path.Combine(IO.Path.GetTempPath(), Guid.NewGuid().ToString("N"))
        IO.Directory.CreateDirectory(tempDir) |> ignore
        try
            let stateDir = IO.Path.Combine(tempDir, ".claude")
            IO.Directory.CreateDirectory(stateDir) |> ignore

            Assert.False(RalphBridge.isActive tempDir)
            RalphBridge.startLoop tempDir "Test" (Some 3) None
            Assert.True(RalphBridge.isActive tempDir)
        finally
            try IO.Directory.Delete(tempDir, true) with _ -> ()

    [<Fact>]
    let ``stopLoop removes state file`` () =
        let tempDir = IO.Path.Combine(IO.Path.GetTempPath(), Guid.NewGuid().ToString("N"))
        IO.Directory.CreateDirectory(tempDir) |> ignore
        try
            let stateDir = IO.Path.Combine(tempDir, ".claude")
            IO.Directory.CreateDirectory(stateDir) |> ignore

            RalphBridge.startLoop tempDir "Test" (Some 3) None
            Assert.True(RalphBridge.isActive tempDir)
            let stopped = RalphBridge.stopLoop tempDir
            Assert.True(stopped)
            Assert.False(RalphBridge.isActive tempDir)
        finally
            try IO.Directory.Delete(tempDir, true) with _ -> ()

    [<Fact>]
    let ``stopLoop returns false when no loop`` () =
        let tempDir = IO.Path.Combine(IO.Path.GetTempPath(), Guid.NewGuid().ToString("N"))
        IO.Directory.CreateDirectory(tempDir) |> ignore
        try
            Assert.False(RalphBridge.stopLoop tempDir)
        finally
            try IO.Directory.Delete(tempDir, true) with _ -> ()

    [<Fact>]
    let ``incrementIteration updates counter`` () =
        let tempDir = IO.Path.Combine(IO.Path.GetTempPath(), Guid.NewGuid().ToString("N"))
        IO.Directory.CreateDirectory(tempDir) |> ignore
        try
            let stateDir = IO.Path.Combine(tempDir, ".claude")
            IO.Directory.CreateDirectory(stateDir) |> ignore

            RalphBridge.startLoop tempDir "Test" (Some 10) None
            let result = RalphBridge.incrementIteration tempDir
            Assert.Equal(Some 2, result)
            let state = RalphBridge.readState tempDir
            Assert.Equal(2, state.Value.Iteration)
        finally
            try IO.Directory.Delete(tempDir, true) with _ -> ()
