namespace TarsEngine.SelfImprovement.Tests

open System
open Xunit
open TarsEngine.FSharp.Core.Specs
open TarsEngine.FSharp.SelfImprovement.SpecKitWorkspace
open TarsEngine.FSharp.SelfImprovement.ProductBacklogDecomposer
open TarsEngine.FSharp.SelfImprovement.TeamConfiguration

module ProductBacklogDecomposerTests =

    [<Fact>]
    let ``decomposeFeature creates epics stories and spikes`` () =
        let summary =
            { Title = "GPU Safety Enforcement"
              Status = Some "Draft"
              FeatureBranch = Some "feature/gpu-safety"
              Created = Some "2025-03-01"
              UserStories =
                [ { Title = "Ensure harness throttles unsafe GPU loads"
                    Priority = Some "P1"
                    AcceptanceCriteria =
                        [ "1. When GPU temperature exceeds safe limits, the system throttles vector benchmarks automatically."
                          "2. Metrics are exported for the safety governor within 30 seconds." ] } ]
              EdgeCases = [ "GPU thermals exceed thresholds while telemetry link is unavailable." ] }

        let tasks =
            [ { LineNumber = 12
                Phase = Some "Phase 1"
                Status = "todo"
                TaskId = Some "T901"
                Priority = Some "P1"
                StoryTag = Some "US1"
                Description = "Investigate GPU throttling telemetry gaps and propose mitigation."
                Raw = "- [ ] T901 [P1] [US1] Investigate GPU throttling telemetry gaps and propose mitigation." }
              { LineNumber = 18
                Phase = Some "Phase 1"
                Status = "todo"
                TaskId = Some "T902"
                Priority = Some "P2"
                StoryTag = None
                Description = "Implement safety dashboard integration for GPU metrics."
                Raw = "- [ ] T902 [P2] Implement safety dashboard integration for GPU metrics." } ]

        let feature =
            { Id = "gpu-safety"
              Directory = "/specs/gpu-safety"
              SpecPath = "/specs/gpu-safety/spec.md"
              PlanPath = None
              TasksPath = None
              Summary = summary
              Tasks = tasks }

        let result = decomposeFeature feature

        Assert.Equal("gpu-safety", result.FeatureId)
        Assert.Equal(summary.Title, result.FeatureTitle)

        let epic = Assert.Single(result.Epics)
        Assert.Equal(BacklogItemType.Epic, epic.ItemType)
        Assert.Equal(BacklogStatus.Ready, epic.Status)
        Assert.Empty(epic.Participants)

        Assert.True(result.Stories |> List.exists (fun story -> story.EpicId = Some epic.Id && story.ItemType = BacklogItemType.Story))
        Assert.True(result.Spikes |> List.exists (fun spike -> spike.ItemType = BacklogItemType.Spike))

        Assert.True(result.Spikes |> List.forall (fun spike -> spike.Status = BacklogStatus.Ready))

        let trace = Assert.Single(result.Relationships)
        Assert.Equal(epic.Id, trace.EpicId)
        Assert.True(trace.StoryIds.Length >= 1)
        Assert.True(trace.SpikeIds.Length >= 1)

        let spikeFromInvestigation =
            result.Spikes
            |> List.tryFind (fun spike -> spike.Description.Contains("Investigate GPU throttling", StringComparison.OrdinalIgnoreCase))

        Assert.True(spikeFromInvestigation.IsSome, "Spike should be generated from investigation task.")

    [<Fact>]
    let ``decomposeFeature attaches participants from team configuration`` () =
        let yaml =
            """
teams:
  - name: SafetySquad
    description: GPU safety enforcement team
    participants:
      - id: safety-governor
        role: SafetyGovernor
        metascript: metascripts/safety-governor.trsx
      - id: telemetry-analyst
        role: TelemetryAnalyst
    assignments:
      - tags: ["edge-case"]
        itemTypes: ["spike"]
        participants: ["safety-governor"]
      - tags: ["spec-kit", "gpu-safety"]
        itemTypes: ["story"]
        participants: ["telemetry-analyst"]
"""

        let registry = loadFromText yaml

        let summary =
            { Title = "GPU Safety Enforcement"
              Status = Some "Draft"
              FeatureBranch = Some "feature/gpu-safety"
              Created = Some "2025-03-01"
              UserStories =
                [ { Title = "Ensure harness throttles unsafe GPU loads"
                    Priority = Some "P1"
                    AcceptanceCriteria = [ "1. Ensure telemetry is recorded for throttling events." ] } ]
              EdgeCases = [ "GPU thermals exceed thresholds while telemetry link is unavailable." ] }

        let feature =
            { Id = "gpu-safety"
              Directory = "/specs/gpu-safety"
              SpecPath = "/specs/gpu-safety/spec.md"
              PlanPath = None
              TasksPath = None
              Summary = summary
              Tasks = [] }

        let result = decomposeFeatureWithTeams feature (Some registry)

        let epic = Assert.Single(result.Epics)
        Assert.Empty(epic.Participants)

        let story = Assert.Single(result.Stories)
        Assert.Contains(story.Participants, fun participant -> participant.Id = "telemetry-analyst")

        let spike = Assert.Single(result.Spikes)
        let assigned = Assert.Single(spike.Participants)
        Assert.Equal("safety-governor", assigned.Id)
