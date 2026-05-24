namespace TarsEngine.SelfImprovement.Tests

open System
open System.Collections.Generic
open System.IO
open System.Text.Json
open Microsoft.Extensions.Logging.Abstractions
open Xunit
open TarsEngine.FSharp.SelfImprovement
open TarsEngine.FSharp.SelfImprovement.GovernanceLedger
open TarsEngine.FSharp.SelfImprovement.PersistentAdaptiveMemory

[<CollectionDefinition("Serial", DisableParallelization = true)>]
type SerialCollectionDefinition() = class end

[<Collection("Serial")>]
module GovernanceLedgerTests =

    let private stubPolicySnapshot =
        { requireConsensus = false
          requireCritic = false
          stopOnFailure = false
          captureLogs = false
          patchCommands = 0
          validationCommands = 0
          benchmarkCommands = 0
          hasAgentProvider = false
          hasTraceProvider = false
          hasFeedbackSink = false
          hasReasoningCritic = false
          consensusRule = None }

    [<Fact>]
    let ``recordIteration persists metrics artifacts and commands`` () =
        let originalDirectory = Environment.CurrentDirectory
        let tempRoot = Path.Combine(Path.GetTempPath(), $"ledger-test-{Guid.NewGuid():N}")
        Directory.CreateDirectory(tempRoot) |> ignore
        Environment.CurrentDirectory <- tempRoot

        try
            let runId = Guid.NewGuid()
            let timestamp = DateTime.UtcNow

            let telemetry = Dictionary<string, obj>()
            telemetry["inference.latency_ms"] <- box 42.5
            telemetry["inference.model"] <- box "demo-model"
            telemetry["inference.tokens"] <- box 128

            let memoryEntry: MemoryEntry =
                { runId = runId
                  specId = "feature-123"
                  specPath = "specs/feature/spec.md"
                  description = Some "Spec iteration run"
                  timestamp = timestamp
                  consensus = None
                  critic = None
                  reasoning = []
                  policyBefore = stubPolicySnapshot
                  policyAfter = stubPolicySnapshot
                  policyChanges = []
                  inferenceTelemetry = telemetry
                  agentFeedback = []
                  validatorFindings = []
                  validatorDisagreements = []
                  harness =
                    Some
                        { status = "passed"
                          failureReason = None
                          failedCommandCount = Some 0 } }

            let metrics =
                Map.ofList [
                    "capability.pass_ratio", box 0.82
                    "feedback.total", box 3
                ]

            let artifacts =
                [ "patches/iter_0001.patch"
                  "logs/iter_0001.log" ]

            let commands =
                [ { CommandSnapshot.Name = "dotnet-restore"
                    Stage = "pre-validation"
                    ExitCode = 0
                    DurationSeconds = 4.5
                    StartedAt = timestamp.AddSeconds(-6.0)
                    CompletedAt = timestamp.AddSeconds(-1.5) }
                  { CommandSnapshot.Name = "dotnet-test"
                    Stage = "validation"
                    ExitCode = 0
                    DurationSeconds = 12.8
                    StartedAt = timestamp.AddSeconds(-1.5)
                    CompletedAt = timestamp } ]

            let nextSteps = [ "Investigate benchmark deltas"; "Promote successful artifacts" ]

            GovernanceLedger.recordIteration NullLogger.Instance memoryEntry "passed" metrics artifacts commands nextSteps

            let ledgerDir = Path.Combine(tempRoot, ".specify", "ledger", "iterations")
            Assert.True(Directory.Exists(ledgerDir), "Ledger directory should exist.")

            let entries =
                Directory.GetFiles(ledgerDir, "*.json")
                |> Array.filter (fun path -> not (path.EndsWith("latest.json", StringComparison.OrdinalIgnoreCase)))

            Assert.Equal(1, entries.Length)

            let payload = File.ReadAllText(entries.[0])
            use document = JsonDocument.Parse(payload)
            let root = document.RootElement

            Assert.Equal("passed", root.GetProperty("status").GetString())

            let metricsElement = root.GetProperty("metrics")
            let mutable passRatioElement = Unchecked.defaultof<JsonElement>
            Assert.True(metricsElement.TryGetProperty("capability.pass_ratio", &passRatioElement))
            Assert.Equal(0.82, passRatioElement.GetDouble(), 2)

            let artifactsElement = root.GetProperty("artifacts")
            Assert.Equal(2, artifactsElement.GetArrayLength())
            let recordedArtifacts =
                artifactsElement.EnumerateArray()
                |> Seq.map (fun item -> item.GetString())
                |> Seq.toList
            Assert.Contains("patches/iter_0001.patch", recordedArtifacts)
            Assert.Contains("logs/iter_0001.log", recordedArtifacts)

            let telemetryElement = root.GetProperty("inferenceTelemetry")
            Assert.Equal(42.5, telemetryElement.GetProperty("inference.latency_ms").GetDouble(), 2)
            Assert.Equal("demo-model", telemetryElement.GetProperty("inference.model").GetString())
            Assert.Equal(128, telemetryElement.GetProperty("inference.tokens").GetInt32())

            let commandsElement = root.GetProperty("commands")
            Assert.Equal(2, commandsElement.GetArrayLength())
            let validationCommand = commandsElement[1]
            Assert.Equal("dotnet-test", validationCommand.GetProperty("name").GetString())
            Assert.Equal("validation", validationCommand.GetProperty("stage").GetString())
            Assert.Equal(0, validationCommand.GetProperty("exitCode").GetInt32())

            let stepsElement = root.GetProperty("nextSteps")
            Assert.Equal(2, stepsElement.GetArrayLength())
            Assert.Equal("Investigate benchmark deltas", stepsElement[0].GetString())

        finally
            Environment.CurrentDirectory <- originalDirectory
            if Directory.Exists(tempRoot) then
                Directory.Delete(tempRoot, true)
