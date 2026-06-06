module Tars.Tests.SelfImprovementTests

open System
open System.Threading.Tasks
open Xunit
open Xunit.Abstractions
open Tars.Core
open Tars.Cortex.SelfImprovement

/// Unit tests for the Self-Improvement module (Phase 4)
type SelfImprovementTests(output: ITestOutputHelper) =

    // Mock EpistemicGovernor for testing
    let createMockGovernor () =
        { new IEpistemicGovernor with
            member _.GenerateVariants(_, count) =
                Task.FromResult([ for i in 1..count -> $"Variant %d{i}" ])

            member _.VerifyGeneralization(_, _, variants) =
                Task.FromResult(
                    { IsVerified = true
                      Score = 0.85
                      Feedback = "All variants passed"
                      FailedVariants = [] }
                )

            member _.ExtractPrinciple(taskDesc, _) =
                Task.FromResult(
                    { Id = Guid.NewGuid()
                      Statement = $"Principle from: %s{taskDesc}"
                      Context = "test-context"
                      Status = EpistemicStatus.Hypothesis
                      Confidence = 0.5
                      DerivedFrom = []
                      CreatedAt = DateTime.UtcNow
                      LastVerified = DateTime.UtcNow }
                )

            member _.SuggestCurriculum(_, _, _) =
                Task.FromResult("Next: Learn about advanced patterns")

            member _.Verify(_) = Task.FromResult(true)
            member _.GetRelatedCodeContext(_) = Task.FromResult("mock context") }

    // Mock governor that fails verification
    let createFailingGovernor () =
        { new IEpistemicGovernor with
            member _.GenerateVariants(_, _) = Task.FromResult([ "Variant 1" ])

            member _.VerifyGeneralization(_, _, _) =
                Task.FromResult(
                    { IsVerified = false
                      Score = 0.2
                      Feedback = "Failed to generalize"
                      FailedVariants = [ "Variant 1" ] }
                )

            member _.ExtractPrinciple(taskDesc, _) =
                Task.FromResult(
                    { Id = Guid.NewGuid()
                      Statement = $"Failed principle: %s{taskDesc}"
                      Context = "test-context"
                      Status = EpistemicStatus.Hypothesis
                      Confidence = 0.5
                      DerivedFrom = []
                      CreatedAt = DateTime.UtcNow
                      LastVerified = DateTime.UtcNow }
                )

            member _.SuggestCurriculum(_, _, _) = Task.FromResult("")
            member _.Verify(_) = Task.FromResult(false)
            member _.GetRelatedCodeContext(_) = Task.FromResult("") }

    // =========================================================================
    // Session Management Tests
    // =========================================================================

    [<Fact>]
    member _.``CreateSession: Creates empty session``() =
        let session = createSession ()

        Assert.Empty(session.CompletedTasks)
        Assert.Empty(session.ExtractedBeliefs)
        Assert.Empty(session.CurriculumSuggestions)
        Assert.True(session.BeliefGraph.Beliefs.IsEmpty)

    [<Fact>]
    member _.``CreateSessionFromGraph: Preserves existing beliefs``() =
        let belief =
            { Id = Guid.NewGuid()
              Statement = "Existing belief"
              Context = "test"
              Status = EpistemicStatus.VerifiedFact
              Confidence = 0.9
              DerivedFrom = []
              CreatedAt = DateTime.UtcNow
              LastVerified = DateTime.UtcNow }

        let graph = BeliefGraph.empty () |> BeliefGraph.addBelief belief
        let session = createSessionFromGraph graph

        Assert.Single(session.ExtractedBeliefs) |> ignore
        Assert.Equal(belief.Statement, session.ExtractedBeliefs.Head.Statement)

    // =========================================================================
    // Learning Tests
    // =========================================================================

    [<Fact>]
    member _.``LearnFromTask: Extracts belief and updates session``() =
        async {
            let governor = createMockGovernor ()
            let session = createSession ()

            let! (result, updatedSession) =
                learnFromTaskAsync governor defaultConfig "Calculate sum" "def sum(a, b): return a + b" session

            Assert.Equal("Calculate sum", result.TaskDescription)
            Assert.True(result.ExtractedBelief.IsSome)
            Assert.True(result.WasVerified)
            Assert.True(result.Curriculum.IsSome)

            Assert.Single(updatedSession.CompletedTasks) |> ignore
            Assert.Single(updatedSession.ExtractedBeliefs) |> ignore
            Assert.Single(updatedSession.CurriculumSuggestions) |> ignore

            output.WriteLine $"Extracted belief: %s{result.ExtractedBelief.Value.Statement}"
            output.WriteLine $"Curriculum: %s{result.Curriculum.Value}"
        }
        |> Async.RunSynchronously

    [<Fact>]
    member _.``LearnFromTask: Sets VerifiedFact status when score is high``() =
        async {
            let governor = createMockGovernor ()
            let session = createSession ()

            let! (result, _) = learnFromTaskAsync governor defaultConfig "High confidence task" "solution" session

            Assert.True(result.ExtractedBelief.IsSome)
            Assert.Equal(EpistemicStatus.VerifiedFact, result.ExtractedBelief.Value.Status)
            Assert.True(result.ExtractedBelief.Value.Confidence >= 0.7)
        }
        |> Async.RunSynchronously

    [<Fact>]
    member _.``LearnFromTask: Sets Fallacy status when verification fails``() =
        async {
            let governor = createFailingGovernor ()
            let session = createSession ()

            let! (result, _) = learnFromTaskAsync governor defaultConfig "Failed task" "bad solution" session

            Assert.True(result.ExtractedBelief.IsSome)
            Assert.Equal(EpistemicStatus.Fallacy, result.ExtractedBelief.Value.Status)
            Assert.False(result.WasVerified)
        }
        |> Async.RunSynchronously

    [<Fact>]
    member _.``LearnFromTasks: Accumulates multiple learnings``() =
        async {
            let governor = createMockGovernor ()
            let session = createSession ()

            let tasks =
                [ ("Task 1", "Solution 1"); ("Task 2", "Solution 2"); ("Task 3", "Solution 3") ]

            let! (results, updatedSession) = learnFromTasksAsync governor defaultConfig tasks session

            Assert.Equal(3, results.Length)
            Assert.Equal(3, updatedSession.CompletedTasks.Length)
            Assert.Equal(3, updatedSession.ExtractedBeliefs.Length)
            Assert.Equal(3, updatedSession.CurriculumSuggestions.Length)
        }
        |> Async.RunSynchronously

    // =========================================================================
    // Analysis Tests
    // =========================================================================

    [<Fact>]
    member _.``GetVerifiedPrinciples: Filters correctly``() =
        async {
            let governor = createMockGovernor ()
            let session = createSession ()

            let! (_, updatedSession) = learnFromTaskAsync governor defaultConfig "Test task" "solution" session

            let principles = getVerifiedPrinciples updatedSession
            Assert.Single(principles) |> ignore
        }
        |> Async.RunSynchronously

    [<Fact>]
    member _.``GetKnowledgeGaps: Returns refuted contexts``() =
        async {
            let governor = createFailingGovernor ()
            let session = createSession ()

            let! (_, updatedSession) = learnFromTaskAsync governor defaultConfig "Failed task" "bad" session

            let gaps = getKnowledgeGaps updatedSession
            Assert.Single(gaps) |> ignore
        }
        |> Async.RunSynchronously

    // =========================================================================
    // Statistics Tests
    // =========================================================================

    [<Fact>]
    member _.``GetSessionStats: Returns correct counts``() =
        async {
            let governor = createMockGovernor ()
            let session = createSession ()

            let! (_, s1) = learnFromTaskAsync governor defaultConfig "Task 1" "sol" session
            let! (_, s2) = learnFromTaskAsync governor defaultConfig "Task 2" "sol" s1

            let stats = getSessionStats s2

            Assert.Equal(2, stats.TasksCompleted)
            Assert.Equal(2, stats.BeliefsExtracted)
            Assert.True(stats.VerifiedPrinciples >= 0)
        }
        |> Async.RunSynchronously

    [<Fact>]
    member _.``GetSessionSummary: Produces readable output``() =
        async {
            let governor = createMockGovernor ()
            let session = createSession ()

            let! (_, updated) = learnFromTaskAsync governor defaultConfig "Test" "solution" session

            let summary = getSessionSummary updated
            output.WriteLine(summary)

            Assert.Contains("Tasks Completed:", summary)
            Assert.Contains("Beliefs Extracted:", summary)
        }
        |> Async.RunSynchronously

    // =========================================================================
    // Curriculum Tests
    // =========================================================================

    [<Fact>]
    member _.``GetNextCurriculumTask: Returns most recent suggestion``() =
        async {
            let governor = createMockGovernor ()
            let session = createSession ()

            let! (_, updated) = learnFromTaskAsync governor defaultConfig "Test" "solution" session

            let next = getNextCurriculumTask updated
            Assert.True(next.IsSome)
            output.WriteLine $"Next curriculum: %s{next.Value}"
        }
        |> Async.RunSynchronously
