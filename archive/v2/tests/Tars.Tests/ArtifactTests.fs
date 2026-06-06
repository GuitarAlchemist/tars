module Tars.Tests.ArtifactTests

open Xunit
open Tars.Core.Artifact

// ============================================================================
// ArtifactFormat Tests
// ============================================================================

[<Fact>]
let ``ArtifactFormat types are correctly defined`` () =
    let formats = [ Markdown; JSON; Code "fsharp"; Binary; Directory ]
    Assert.Equal(5, formats.Length)

// ============================================================================
// StageArtifact Tests
// ============================================================================

[<Fact>]
let ``createArtifact creates artifact with correct fields`` () =
    let artifact = createArtifact "a1" "readme.md" "/path/readme.md" Markdown

    Assert.Equal("a1", artifact.Id)
    Assert.Equal("readme.md", artifact.Name)
    Assert.Equal("/path/readme.md", artifact.Path)
    Assert.Equal(Markdown, artifact.Format)

[<Fact>]
let ``createArtifactFromFile includes size and stage`` () =
    let artifact =
        createArtifactFromFile "a2" "code.fs" "/path/code.fs" (Code "fsharp") 1024L "Development"

    Assert.Equal(1024L, artifact.Size)
    Assert.Equal("Development", artifact.CreatedBy)

// ============================================================================
// Blueprint Distillation Tests
// ============================================================================

[<Fact>]
let ``generateBlueprint includes file list when configured`` () =
    let artifacts =
        [ createArtifactFromFile "a1" "spec.md" "/spec.md" Markdown 100L "Vision"
          createArtifactFromFile "a2" "design.json" "/design.json" JSON 200L "Vision" ]

    let config =
        { defaultBlueprintConfig with
            IncludeFileList = true }

    let blueprint = generateBlueprint config artifacts

    Assert.Contains("spec.md", blueprint)
    Assert.Contains("design.json", blueprint)

[<Fact>]
let ``generateBlueprint excludes file list when configured`` () =
    let artifacts =
        [ createArtifactFromFile "a1" "spec.md" "/spec.md" Markdown 100L "Vision" ]

    let config =
        { defaultBlueprintConfig with
            IncludeFileList = false }

    let blueprint = generateBlueprint config artifacts

    Assert.Contains("1 artifacts", blueprint)

[<Fact>]
let ``generateBlueprint respects max length`` () =
    let artifacts =
        [ createArtifactFromFile "a1" "very_long_filename_that_should_be_truncated.md" "/path" Markdown 100L "Stage"
          createArtifactFromFile "a2" "another_very_long_filename.md" "/path" Markdown 100L "Stage"
          createArtifactFromFile "a3" "yet_another_file.md" "/path" Markdown 100L "Stage" ]

    let config =
        { defaultBlueprintConfig with
            MaxLength = 50 }

    let blueprint = generateBlueprint config artifacts

    Assert.True(blueprint.Length <= 50)
    Assert.EndsWith("...", blueprint)

// ============================================================================
// ArtifactManager Tests
// ============================================================================

[<Fact>]
let ``ArtifactManager records and retrieves artifacts`` () =
    let manager = ArtifactManager("/tmp/artifacts")

    let artifact =
        createArtifactFromFile "a1" "test.md" "/test.md" Markdown 100L "Vision"

    manager.RecordArtifact("proj-1", "Vision", artifact)
    let retrieved = manager.GetStageArtifacts("proj-1", "Vision")

    Assert.Equal(1, retrieved.Length)
    Assert.Equal("test.md", retrieved.[0].Name)

[<Fact>]
let ``ArtifactManager generates handoff context`` () =
    let manager = ArtifactManager("/tmp/artifacts")

    let artifact =
        createArtifactFromFile "a1" "vision.md" "/vision.md" Markdown 100L "Vision"

    manager.RecordArtifact("proj-1", "Vision", artifact)
    let context = manager.GetHandoffContext("proj-1", "Vision")

    Assert.True(context.IsSome)
    Assert.Contains("Previous Stage: Vision", context.Value)

[<Fact>]
let ``ArtifactManager clears project artifacts`` () =
    let manager = ArtifactManager("/tmp/artifacts")

    let artifact =
        createArtifactFromFile "a1" "test.md" "/test.md" Markdown 100L "Vision"

    manager.RecordArtifact("proj-1", "Vision", artifact)
    Assert.Equal(1, manager.GetStageArtifacts("proj-1", "Vision").Length)

    manager.ClearProject("proj-1")
    Assert.Equal(0, manager.GetStageArtifacts("proj-1", "Vision").Length)

[<Fact>]
let ``ArtifactManager completes stage with blueprint`` () =
    let manager = ArtifactManager("/tmp/artifacts")
    manager.RecordArtifact("proj-1", "Vision", createArtifactFromFile "a1" "spec.md" "/spec.md" Markdown 100L "Vision")

    manager.RecordArtifact(
        "proj-1",
        "Vision",
        createArtifactFromFile "a2" "draft.md" "/draft.md" Markdown 200L "Vision"
    )

    let completed = manager.CompleteStage("proj-1", "Vision")

    Assert.True(completed.IsSome)
    Assert.True(completed.Value.Summary.IsSome)
    Assert.Equal(2, completed.Value.Artifacts.Length)
