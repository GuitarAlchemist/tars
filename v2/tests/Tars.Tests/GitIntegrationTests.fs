module Tars.Tests.GitIntegrationTests

open System
open System.IO
open Xunit
open Tars.Core.GitIntegration

// ============================================================================
// GitResult Tests
// ============================================================================

[<Fact>]
let ``GitResult Success contains output`` () =
    let result = Success "commit abc123"

    match result with
    | Success output -> Assert.Contains("abc123", output)
    | Error _ -> Assert.Fail("Expected Success")

[<Fact>]
let ``GitResult Error contains message`` () =
    let result = Error "fatal: not a git repository"

    match result with
    | Error msg -> Assert.Contains("not a git repository", msg)
    | Success _ -> Assert.Fail("Expected Error")

// ============================================================================
// GitCommit Tests
// ============================================================================

[<Fact>]
let ``GitCommit has all required fields`` () =
    let commit =
        { Hash = "abc123def456"
          ShortHash = "abc123d"
          Author = "Test User"
          Date = DateTime.UtcNow
          Message = "Test commit" }

    Assert.Equal("abc123def456", commit.Hash)
    Assert.Equal("abc123d", commit.ShortHash)
    Assert.Equal("Test User", commit.Author)
    Assert.Equal("Test commit", commit.Message)

// ============================================================================
// GitBranch Tests
// ============================================================================

[<Fact>]
let ``GitBranch has correct properties`` () =
    let branch =
        { Name = "main"
          IsRemote = false
          IsCurrent = true }

    Assert.Equal("main", branch.Name)
    Assert.False(branch.IsRemote)
    Assert.True(branch.IsCurrent)

// ============================================================================
// GitStatusEntry Tests
// ============================================================================

[<Fact>]
let ``GitStatusEntry captures file status`` () =
    let entry = { Path = "src/test.fs"; Status = "M" }
    Assert.Equal("src/test.fs", entry.Path)
    Assert.Equal("M", entry.Status)

// ============================================================================
// GitRepoState Tests
// ============================================================================

[<Fact>]
let ``GitRepoState captures repository state`` () =
    let state =
        { IsRepo = true
          CurrentBranch = Some "main"
          HasUncommittedChanges = false
          RemoteUrl = Some "https://github.com/test/repo.git" }

    Assert.True(state.IsRepo)
    Assert.Equal(Some "main", state.CurrentBranch)
    Assert.False(state.HasUncommittedChanges)
    Assert.True(state.RemoteUrl.IsSome)

// ============================================================================
// GitManager Tests (isolated, no actual git required)
// ============================================================================

[<Fact>]
let ``createManager creates manager for path`` () =
    let manager = createManager "/tmp/test-repo"
    Assert.NotNull(manager)

[<Fact>]
let ``GitManager IsRepository returns false for non-repo`` () =
    // Use a temp directory that's definitely not a git repo
    let tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString())
    Directory.CreateDirectory(tempDir) |> ignore

    try
        let manager = GitManager(tempDir)
        let isRepo = manager.IsRepository() |> Async.AwaitTask |> Async.RunSynchronously
        Assert.False(isRepo)
    finally
        Directory.Delete(tempDir, true)

[<Fact>]
let ``GitManager GetRepoState returns non-repo state for non-repo`` () =
    let tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString())
    Directory.CreateDirectory(tempDir) |> ignore

    try
        let manager = GitManager(tempDir)
        let state = manager.GetRepoState() |> Async.AwaitTask |> Async.RunSynchronously
        Assert.False(state.IsRepo)
        Assert.True(state.CurrentBranch.IsNone)
    finally
        Directory.Delete(tempDir, true)

// Note: Full integration tests would require actual git operations
// which should be run in a controlled CI environment
