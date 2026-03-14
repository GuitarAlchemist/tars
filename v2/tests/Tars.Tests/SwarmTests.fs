namespace Tars.Tests

open System
open Xunit
open Tars.Connectors.Redis

/// Tests for SwarmBus and SwarmWorker (require Redis/Sider on localhost:6379).
module SwarmTests =

    // =========================================================================
    // Unit tests (no Redis required) - test type construction and serialization
    // =========================================================================

    [<Fact>]
    let ``SwarmJob record round-trips through fields`` () =
        let job =
            { JobId = "test-001"
              Goal = "Analyze code quality"
              PatternHint = Some "react"
              MaxSteps = 5
              Priority = 1
              PostedBy = "unit-test"
              PostedAt = DateTime(2026, 1, 1, 0, 0, 0, DateTimeKind.Utc) }

        Assert.Equal("test-001", job.JobId)
        Assert.Equal("Analyze code quality", job.Goal)
        Assert.Equal(Some "react", job.PatternHint)
        Assert.Equal(5, job.MaxSteps)
        Assert.Equal(1, job.Priority)
        Assert.Equal("unit-test", job.PostedBy)

    [<Fact>]
    let ``SwarmResult record captures execution outcome`` () =
        let result =
            { JobId = "test-001"
              WorkerId = "tars-worker-abc12345"
              Success = true
              Output = "Executed 3 steps via react"
              PatternUsed = "react"
              DurationMs = 1500L
              StepCount = 3
              CompletedAt = DateTime.UtcNow }

        Assert.True(result.Success)
        Assert.Equal("react", result.PatternUsed)
        Assert.Equal(3, result.StepCount)
        Assert.Equal(1500L, result.DurationMs)

    [<Fact>]
    let ``SwarmHeartbeat captures worker state`` () =
        let hb =
            { WorkerId = "tars-worker-abc12345"
              Status = "idle"
              CurrentJobId = None
              UptimeMs = 60000L
              CompletedJobs = 5
              Timestamp = DateTime.UtcNow }

        Assert.Equal("idle", hb.Status)
        Assert.True(hb.CurrentJobId.IsNone)
        Assert.Equal(5, hb.CompletedJobs)

    [<Fact>]
    let ``SwarmHeartbeat busy state has job id`` () =
        let hb =
            { WorkerId = "tars-worker-abc12345"
              Status = "busy"
              CurrentJobId = Some "job-42"
              UptimeMs = 120000L
              CompletedJobs = 10
              Timestamp = DateTime.UtcNow }

        Assert.Equal("busy", hb.Status)
        Assert.Equal(Some "job-42", hb.CurrentJobId)

    [<Fact>]
    let ``SwarmBelief captures knowledge triple`` () =
        let belief =
            { Subject = "F#"
              Predicate = "is-a"
              Object = "programming language"
              Confidence = 0.95
              SourceWorker = "tars-worker-abc12345" }

        Assert.Equal("F#", belief.Subject)
        Assert.Equal("is-a", belief.Predicate)
        Assert.Equal(0.95, belief.Confidence)

    [<Fact>]
    let ``SwarmChannels produces correct key patterns`` () =
        Assert.Equal("tars:work", SwarmChannels.WorkQueue)
        Assert.Equal("tars:results", SwarmChannels.Results)
        Assert.Equal("tars:heartbeat", SwarmChannels.Heartbeat)
        Assert.Equal("tars:knowledge", SwarmChannels.Knowledge)
        Assert.Equal("tars:control", SwarmChannels.Control)
        Assert.Equal("tars:worker:w1", SwarmChannels.workerKey "w1")
        Assert.Equal("tars:job:j1", SwarmChannels.jobKey "j1")
        Assert.Equal("tars:result:j1", SwarmChannels.resultKey "j1")

    [<Fact>]
    let ``SwarmJob with None PatternHint uses default`` () =
        let job =
            { JobId = "test-002"
              Goal = "Default pattern"
              PatternHint = None
              MaxSteps = 3
              Priority = 2
              PostedBy = "test"
              PostedAt = DateTime.UtcNow }

        Assert.True(job.PatternHint.IsNone)

    // =========================================================================
    // Integration tests (require Redis/Sider)
    // =========================================================================

    [<Fact>]
    let ``SwarmBus connects to Redis`` () =
        if not (TestHelpers.requireRedis ()) then () else

        use bus = new SwarmBus("localhost:6379")
        Assert.True(bus.Connect())
        Assert.True(bus.IsConnected)

    [<Fact>]
    let ``SwarmBus post and take job round-trip`` () =
        if not (TestHelpers.requireRedis ()) then () else

        use bus = new SwarmBus("localhost:6379")
        bus.Connect() |> ignore
        bus.FlushSwarm() // Clean state

        let job =
            { JobId = "rt-001"
              Goal = "Test round trip"
              PatternHint = Some "cot"
              MaxSteps = 3
              Priority = 1
              PostedBy = "test"
              PostedAt = DateTime.UtcNow }

        bus.PostJob(job)
        Assert.Equal(1, bus.QueueLength())

        let taken = bus.TakeJob(1)
        Assert.True(taken.IsSome)
        Assert.Equal("rt-001", taken.Value.JobId)
        Assert.Equal("Test round trip", taken.Value.Goal)
        Assert.Equal(0, bus.QueueLength())

        bus.FlushSwarm()

    [<Fact>]
    let ``SwarmBus submit and get result`` () =
        if not (TestHelpers.requireRedis ()) then () else

        use bus = new SwarmBus("localhost:6379")
        bus.Connect() |> ignore
        bus.FlushSwarm()

        let result =
            { JobId = "res-001"
              WorkerId = "test-worker"
              Success = true
              Output = "Done"
              PatternUsed = "cot"
              DurationMs = 100L
              StepCount = 2
              CompletedAt = DateTime.UtcNow }

        bus.SubmitResult(result)

        let fetched = bus.GetResult("res-001")
        Assert.True(fetched.IsSome)
        Assert.True(fetched.Value.Success)
        Assert.Equal("cot", fetched.Value.PatternUsed)

        bus.FlushSwarm()

    [<Fact>]
    let ``SwarmBus heartbeat and worker discovery`` () =
        if not (TestHelpers.requireRedis ()) then () else

        use bus = new SwarmBus("localhost:6379")
        bus.Connect() |> ignore
        bus.FlushSwarm()

        let hb =
            { WorkerId = "test-hb-worker"
              Status = "idle"
              CurrentJobId = None
              UptimeMs = 5000L
              CompletedJobs = 0
              Timestamp = DateTime.UtcNow }

        bus.SendHeartbeat(hb)

        let workers = bus.GetWorkers()
        Assert.True(workers.Length >= 1)
        Assert.Contains(workers, fun w -> w.WorkerId = "test-hb-worker")

        bus.FlushSwarm()

    [<Fact>]
    let ``SwarmBus flush clears all state`` () =
        if not (TestHelpers.requireRedis ()) then () else

        use bus = new SwarmBus("localhost:6379")
        bus.Connect() |> ignore

        // Add some data
        bus.PostJob(
            { JobId = "flush-test"
              Goal = "test"
              PatternHint = None
              MaxSteps = 1
              Priority = 1
              PostedBy = "test"
              PostedAt = DateTime.UtcNow })

        bus.FlushSwarm()
        Assert.Equal(0, bus.QueueLength())

    [<Fact>]
    let ``SwarmBus stats returns summary`` () =
        if not (TestHelpers.requireRedis ()) then () else

        use bus = new SwarmBus("localhost:6379")
        bus.Connect() |> ignore
        bus.FlushSwarm()

        let stats = bus.Stats()
        Assert.Equal(0, stats.QueueLength)
        Assert.True(stats.IsConnected)

        bus.FlushSwarm()
