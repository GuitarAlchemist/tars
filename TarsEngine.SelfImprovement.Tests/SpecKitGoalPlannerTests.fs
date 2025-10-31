[<Xunit.Collection("Serial")>]
module TarsEngine.SelfImprovement.Tests.SpecKitGoalPlannerTests

open System
open System.IO
open System.Text.Json
open Xunit
open TarsEngine.FSharp.Core.Specs
open TarsEngine.FSharp.SelfImprovement.SpecKitWorkspace
open TarsEngine.FSharp.SelfImprovement.SpecKitGoalPlanner

let private createSampleSelection root =
    let featureDir = Path.Combine(root, "feature-demo")
    Directory.CreateDirectory(featureDir) |> ignore
    File.WriteAllText(Path.Combine(featureDir, "spec.md"), "# Feature Specification: Demo\n")
    let summary: SpecKitSummary =
        { Title = "Auto Goal Demo"
          Status = Some "Draft"
          FeatureBranch = Some "demo-branch"
          Created = Some "2025-03-15"
          UserStories = []
          EdgeCases = [] }
    let feature =
        { Id = "feature-demo"
          Directory = featureDir
          SpecPath = Path.Combine(featureDir, "spec.md")
          PlanPath = None
          TasksPath = None
          Summary = summary
          Tasks = [] }
    let task =
        { LineNumber = 1
          Phase = Some "Phase 1"
          Status = "todo"
          TaskId = Some "T001"
          Priority = Some "P1"
          StoryTag = Some "US1"
          Description = "Investigate failing consensus"
          Raw = "- [ ] T001 Investigate failing consensus" }
    { Feature = feature; Task = task; PriorityRank = 1 }

[<Fact>]
let ``recordGoal creates spec and backlog entry`` () =
    let originalDir = Environment.CurrentDirectory
    let tempRoot = Directory.CreateDirectory(Path.Combine(Path.GetTempPath(), $"spec-kit-goal-{Guid.NewGuid():N}"))
    Environment.CurrentDirectory <- tempRoot.FullName

    try
        let selection = createSampleSelection tempRoot.FullName
        recordGoal selection None

        let backlogFile = Path.Combine(tempRoot.FullName, ".specify", "backlog.json")
        Assert.True(File.Exists(backlogFile))

        let backlogContent = File.ReadAllText(backlogFile)
        Assert.Contains("feature-demo", backlogContent)

        let autoSpecsDir = Path.Combine(tempRoot.FullName, ".specify", "specs", "auto-generated")
        Assert.True(Directory.Exists(autoSpecsDir))
        let generatedSpec =
            Directory.GetFiles(autoSpecsDir, "*.md")
            |> Array.tryHead

        Assert.True(generatedSpec.IsSome, "Expected generated spec file.")
        let specContent = File.ReadAllText(generatedSpec.Value)
        Assert.Contains("```metascript", specContent)
        Assert.Contains("feature-demo", specContent)
    finally
        Environment.CurrentDirectory <- originalDir
        Directory.Delete(tempRoot.FullName, true)

[<Fact>]
let ``recordRemediationTicket enqueues backlog entry and spec`` () =
    let originalDir = Environment.CurrentDirectory
    let tempRoot = Directory.CreateDirectory(Path.Combine(Path.GetTempPath(), $"spec-kit-remediation-{Guid.NewGuid():N}"))
    Environment.CurrentDirectory <- tempRoot.FullName

    try
        let specId = "superintelligence-roadmap"
        let iterationId = Guid.NewGuid()
        let artifactDir = Path.Combine(tempRoot.FullName, "output", "remediation_tasks")
        Directory.CreateDirectory(artifactDir) |> ignore
        let artifactPath = Path.Combine(artifactDir, "iteration.md")
        File.WriteAllText(artifactPath, "# Remediation artifact")
        let relativeArtifact = Path.GetRelativePath(tempRoot.FullName, artifactPath)
        let actions = [ "Reviewer", [ "Initiate rollback"; "Schedule manual review" ]; "SafetyGovernor", [] ]

        recordRemediationTicket specId iterationId artifactPath relativeArtifact actions

        let backlogFile = Path.Combine(tempRoot.FullName, ".specify", "backlog.json")
        Assert.True(File.Exists(backlogFile), "Expected backlog.json to be created.")

        use jsonDoc = JsonDocument.Parse(File.ReadAllText(backlogFile))
        Assert.Equal(1, jsonDoc.RootElement.GetArrayLength())

        let autoSpecsDir = Path.Combine(tempRoot.FullName, ".specify", "specs", "auto-generated")
        Assert.True(Directory.Exists(autoSpecsDir))
        let generatedSpec =
            Directory.GetFiles(autoSpecsDir, "*.md")
            |> Array.tryHead

        Assert.True(generatedSpec.IsSome, "Expected remediation spec to be generated.")
        let specContents = File.ReadAllText(generatedSpec.Value)
        Assert.Contains("Remediation Ticket", specContents)
        Assert.Contains(relativeArtifact, specContents)
        Assert.Contains("Initiate rollback", specContents)

        // Duplicate registration should not create additional backlog entries
        recordRemediationTicket specId iterationId artifactPath relativeArtifact actions
        use jsonDocAfter = JsonDocument.Parse(File.ReadAllText(backlogFile))
        Assert.Equal(1, jsonDocAfter.RootElement.GetArrayLength())
    finally
        Environment.CurrentDirectory <- originalDir
        Directory.Delete(tempRoot.FullName, true)
