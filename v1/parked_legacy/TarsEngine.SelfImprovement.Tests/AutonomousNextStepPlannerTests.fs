namespace TarsEngine.SelfImprovement.Tests

open System
open System.IO
open Microsoft.Extensions.Logging.Abstractions
open Xunit
open TarsEngine.FSharp.SelfImprovement.AutonomousNextStepPlanner

[<Xunit.Collection("Serial")>]
module AutonomousNextStepPlannerTests =

    let private writeSpec (root: string) =
        let specDir = Path.Combine(root, ".specify", "specs", "feature-demo")
        Directory.CreateDirectory(specDir) |> ignore

        let specContent =
            """
# Feature Specification: Planner Demo

**Feature Branch**: `planner-demo`
**Created**: 2025-03-15
**Status**: Draft

### User Story - Autonomous prioritisation (Priority: P0)

**Acceptance Scenarios**:
1. Planner should surface insight for high priority roadmap work.

### Edge Cases
- Spec missing tasks.
"""
        File.WriteAllText(Path.Combine(specDir, "spec.md"), specContent.Trim())

        let tasksContent =
            """
## Phase 1
- [ ] T900 [P0] Improve capability metrics handling
"""
        File.WriteAllText(Path.Combine(specDir, "tasks.md"), tasksContent.Trim())

    let private writeLedger (root: string) =
        let ledgerDir = Path.Combine(root, ".specify", "ledger", "iterations")
        Directory.CreateDirectory(ledgerDir) |> ignore
        let latestPath = Path.Combine(ledgerDir, "latest.json")
        let payload =
            """
{
  "runId": "00000000-0000-0000-0000-000000000001",
  "timestamp": "2025-03-15T12:00:00Z",
  "specId": "feature-demo",
  "specPath": "spec.md",
  "description": "sample run",
  "status": "passed",
  "consensus": null,
  "critic": null,
  "policyBefore": null,
  "policyAfter": null,
  "policyChanges": [],
  "agentFeedback": [],
  "harness": null,
  "metrics": {
    "capability.pass_ratio": 0.9,
    "safety.consensus_avg_confidence": 0.85,
    "safety.critic_status": "accept"
  },
  "artifacts": [],
  "commands": [],
  "nextSteps": [],
  "appendPath": null
}
"""
        File.WriteAllText(latestPath, payload.Trim())

        let history =
            [ ("20250314120000_00000000000000000000000000000002.json", 0.8, 0.82, "needs_review")
              ("20250313120000_00000000000000000000000000000003.json", 0.7, 0.78, "reject") ]

        for (fileName, capability, consensus, critic) in history do
            let path = Path.Combine(ledgerDir, fileName)
            let historyPayload =
                $"""
{{
  "runId": "00000000-0000-0000-0000-000000000002",
  "timestamp": "2025-03-14T12:00:00Z",
  "specId": "feature-demo",
  "specPath": "spec.md",
  "description": "history",
  "status": "passed",
  "consensus": null,
  "critic": null,
  "policyBefore": null,
  "policyAfter": null,
  "policyChanges": [],
  "agentFeedback": [],
  "harness": null,
  "metrics": {{
    "capability.pass_ratio": {capability},
    "safety.consensus_avg_confidence": {consensus},
    "safety.critic_status": "{critic}"
  }},
  "artifacts": [],
  "commands": [],
  "nextSteps": [],
  "appendPath": null
}}
"""
            File.WriteAllText(path, historyPayload.Trim())

    [<Fact>]
    let ``plan incorporates governance metrics in rationale`` () =
        let originalDir = Environment.CurrentDirectory
        let tempRoot = Path.Combine(Path.GetTempPath(), $"planner-metrics-{Guid.NewGuid():N}")
        Directory.CreateDirectory(tempRoot) |> ignore
        Environment.CurrentDirectory <- tempRoot

        try
            writeSpec tempRoot
            writeLedger tempRoot

            let logger = NullLoggerFactory.Instance.CreateLogger("planner-test")
            let recommendations =
                plan logger None
                |> Async.RunSynchronously

            let recommendation = Assert.Single(recommendations)
            let rationaleText = recommendation.Rationale |> String.concat " "
            Assert.Contains("Capability pass ratio", rationaleText)
            Assert.Contains("trend 0.75", rationaleText)
            Assert.Contains("Consensus confidence", rationaleText)
            Assert.Contains("trend 0.80", rationaleText)
            Assert.Contains("Critic reject trend 0.50", rationaleText)
        finally
            Environment.CurrentDirectory <- originalDir
            if Directory.Exists(tempRoot) then
                Directory.Delete(tempRoot, true)
