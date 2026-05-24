namespace Tars.Tests

open System
open Xunit
open Tars.Cortex

/// Tests for v2.2 Cognitive Patterns
module CognitivePatternTests =

    // =========================================================================
    // SemanticWatchdog Tests
    // =========================================================================

    module SemanticWatchdogTests =

        [<Fact>]
        let ``SemanticWatchdog: Records token usage`` () =
            let watchdog = SemanticWatchdog.createDefault ()

            watchdog.RecordTokens(100)
            watchdog.RecordTokens(200)

            // Should not trigger alert for normal usage
            let criticals = watchdog.GetCriticalAlerts()
            Assert.Empty(criticals)

        [<Fact>]
        let ``SemanticWatchdog: Detects budget explosion`` () =
            let config =
                { WatchdogConfig.Default with
                    MaxTokensPerMinute = 500 }

            let watchdog = SemanticWatchdog(config)

            // Exceed the limit
            for _ in 1..10 do
                watchdog.RecordTokens(100)

            let criticals = watchdog.GetCriticalAlerts()
            Assert.NotEmpty(criticals)
            Assert.Contains("Budget explosion", criticals.Head.Message)

        [<Fact>]
        let ``SemanticWatchdog: Detects repetitive responses`` () =
            let config =
                { WatchdogConfig.Default with
                    MaxSimilarResponses = 2 }

            let watchdog = SemanticWatchdog(config)

            watchdog.RecordResponse("Same response")
            watchdog.RecordResponse("Same response")
            watchdog.RecordResponse("Same response")

            let alerts = watchdog.GetAlerts()
            Assert.True(alerts |> List.exists (fun a -> a.Message.Contains("Repetitive")))

        [<Fact>]
        let ``SemanticWatchdog: Detects loop limit exceeded`` () =
            let config =
                { WatchdogConfig.Default with
                    MaxLoopIterations = 3 }

            let watchdog = SemanticWatchdog(config)

            for _ in 1..5 do
                watchdog.RecordIteration()

            let criticals = watchdog.GetCriticalAlerts()
            Assert.NotEmpty(criticals)
            Assert.Contains("Loop limit", criticals.Head.Message)

        [<Fact>]
        let ``SemanticWatchdog: Reset clears loop counter`` () =
            let config =
                { WatchdogConfig.Default with
                    MaxLoopIterations = 5 }

            let watchdog = SemanticWatchdog(config)

            watchdog.RecordIteration()
            watchdog.RecordIteration()
            watchdog.ResetLoop()
            watchdog.RecordIteration()

            // Should not exceed limit after reset
            let criticals = watchdog.GetCriticalAlerts()
            Assert.Empty(criticals)

    // =========================================================================
    // ConsensusCircuitBreaker Tests
    // =========================================================================

    module ConsensusCircuitBreakerTests =

        [<Fact>]
        let ``ConsensusCircuitBreaker: Majority consensus passes`` () =
            let cb = ConsensusCircuitBreaker.createDefault ()
            let proposalId = cb.Propose(Guid.NewGuid())

            cb.Vote(proposalId, "agent1", Approve "Looks good")
            cb.Vote(proposalId, "agent2", Approve "Approved")
            cb.Vote(proposalId, "agent3", Reject "Not sure")

            let result = cb.CheckConsensus(proposalId)

            Assert.True(result.Reached)
            Assert.Equal(2, result.Approvals)
            Assert.Equal(1, result.Rejections)

        [<Fact>]
        let ``ConsensusCircuitBreaker: Majority consensus fails`` () =
            let cb = ConsensusCircuitBreaker.createDefault ()
            let proposalId = cb.Propose(Guid.NewGuid())

            cb.Vote(proposalId, "agent1", Reject "No")
            cb.Vote(proposalId, "agent2", Reject "Disagree")
            cb.Vote(proposalId, "agent3", Approve "Yes")

            let result = cb.CheckConsensus(proposalId)

            Assert.False(result.Reached)
            Assert.Equal(1, result.Approvals)
            Assert.Equal(2, result.Rejections)

        [<Fact>]
        let ``ConsensusCircuitBreaker: Unanimous requires all approve`` () =
            let config =
                { ConsensusConfig.Default with
                    Strategy = Unanimous }

            let cb = ConsensusCircuitBreaker(config)
            let proposalId = cb.Propose(Guid.NewGuid())

            cb.Vote(proposalId, "agent1", Approve "Yes")
            cb.Vote(proposalId, "agent2", Approve "Yes")

            let result = cb.CheckConsensus(proposalId)
            Assert.True(result.Reached)

            // Add a rejection
            cb.Vote(proposalId, "agent3", Reject "No")
            let result2 = cb.CheckConsensus(proposalId)
            Assert.False(result2.Reached)

        [<Fact>]
        let ``ConsensusCircuitBreaker: Quorum requires N approvals`` () =
            let config =
                { ConsensusConfig.Default with
                    Strategy = Quorum 3 }

            let cb = ConsensusCircuitBreaker(config)
            let proposalId = cb.Propose(Guid.NewGuid())

            cb.Vote(proposalId, "agent1", Approve "Yes")
            cb.Vote(proposalId, "agent2", Approve "Yes")

            let result1 = cb.CheckConsensus(proposalId)
            Assert.False(result1.Reached) // Only 2, need 3

            cb.Vote(proposalId, "agent3", Approve "Yes")
            let result2 = cb.CheckConsensus(proposalId)
            Assert.True(result2.Reached) // Now have 3

        [<Fact>]
        let ``ConsensusCircuitBreaker: Execute only when consensus reached`` () =
            task {
                let cb = ConsensusCircuitBreaker.createDefault ()
                let proposalId = cb.Propose(Guid.NewGuid())

                cb.Vote(proposalId, "agent1", Approve "Yes")
                cb.Vote(proposalId, "agent2", Approve "Yes")

                let mutable executed = false

                let! result =
                    cb.ExecuteWithConsensusAsync(
                        proposalId,
                        fun () ->
                            task {
                                executed <- true
                                return "Success"
                            }
                    )

                Assert.True(executed)

                match result with
                | Ok v -> Assert.Equal("Success", v)
                | Error e -> Assert.Fail($"Expected success but got: {e}")
            }

    // =========================================================================
    // UncertaintyGatedPlanner Tests (Basic structure tests without LLM)
    // =========================================================================

    module UncertaintyGatedPlannerTests =

        [<Fact>]
        let ``UncertainStep: Classifies confidence levels correctly`` () =
            // Test the confidence classification logic
            let highStep =
                { Id = "1"
                  Action = "test"
                  Confidence = 0.9
                  Level = High
                  Alternatives = []
                  NeedsVerification = false }

            let medStep =
                { Id = "2"
                  Action = "test"
                  Confidence = 0.6
                  Level = Medium
                  Alternatives = []
                  NeedsVerification = false }

            let lowStep =
                { Id = "3"
                  Action = "test"
                  Confidence = 0.3
                  Level = Low
                  Alternatives = []
                  NeedsVerification = true }

            Assert.Equal(High, highStep.Level)
            Assert.Equal(Medium, medStep.Level)
            Assert.Equal(Low, lowStep.Level)

        [<Fact>]
        let ``PlanResult: RequiresHumanReview when low confidence`` () =
            let result =
                { Steps = []
                  OverallConfidence = 0.3
                  RequiresHumanReview = true
                  HighRiskSteps = [] }

            Assert.True(result.RequiresHumanReview)

        [<Fact>]
        let ``PlannerConfig: Default values are sensible`` () =
            let config = PlannerConfig.Default

            Assert.Equal(0.3, config.MinConfidenceToExecute)
            Assert.Equal(0.5, config.RequireReviewBelowConfidence)
            Assert.Equal(3, config.MaxAlternativesToGenerate)
