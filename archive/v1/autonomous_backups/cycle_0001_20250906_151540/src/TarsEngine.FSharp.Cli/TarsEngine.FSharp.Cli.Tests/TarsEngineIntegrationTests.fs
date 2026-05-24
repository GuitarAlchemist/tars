namespace TarsEngine.FSharp.Cli.Tests

open System
open Microsoft.Extensions.Logging
open Microsoft.Extensions.Logging.Abstractions
open Xunit
open TarsEngine.FSharp.Cli.Core

module TarsEngineIntegrationTests =

    // Test fixtures and helpers
    let createTestLogger() = NullLogger<EnhancedTarsIntelligenceEngine>.Instance
    
    let createTestBelief content confidence = {
        content = content
        confidence = confidence
        position = Some { X = 0.5; Y = 0.5; Z = 0.5; W = 0.5 }
        timestamp = DateTime.UtcNow
        agentId = Some "test-agent"
    }
    
    let createTestSkill name complexity = {
        name = name
        pre = [createTestBelief "precondition" 0.8]
        post = [createTestBelief "postcondition" 0.9]
        checker = fun () -> true
        complexity = complexity
        verificationLevel = "standard"
    }
    
    let createTestPosition x y z w = { X = x; Y = y; Z = z; W = w }

    [<Fact>]
    let ``Enhanced infer should preserve base functionality`` () =
        // Arrange
        let engine = EnhancedTarsIntelligenceEngine(createTestLogger())
        let beliefs = [createTestBelief "test belief" 0.7]
        let sessionId = "test-session"
        
        // Act
        let result = engine.EnhancedInfer(beliefs, None, sessionId)
        
        // Assert
        Assert.NotEmpty(result)
        Assert.True(result.[0].confidence >= beliefs.[0].confidence)
        Assert.NotNull(result.[0].position)

    [<Fact>]
    let ``Enhanced infer should improve with collective intelligence`` () =
        // Arrange
        let engine = EnhancedTarsIntelligenceEngine(createTestLogger())
        let beliefs = [createTestBelief "collective belief" 0.6]
        let sessionId = "collective-session"
        
        // Register multiple agents
        engine.RegisterAgent("agent1", createTestPosition 0.3 0.4 0.5 0.6, sessionId)
        engine.RegisterAgent("agent2", createTestPosition 0.7 0.6 0.5 0.4, sessionId)
        
        // Act
        let result = engine.EnhancedInfer(beliefs, None, sessionId)
        
        // Assert
        Assert.NotEmpty(result)
        let metrics = engine.GetPerformanceMetrics(sessionId)
        Assert.True(metrics.tier6_consensus_rate > 0.0)
        Assert.True(metrics.total_inferences > 0L)

    [<Fact>]
    let ``Enhanced expectedFreeEnergy should handle simple plans`` () =
        // Arrange
        let engine = EnhancedTarsIntelligenceEngine(createTestLogger())
        let simplePlan = [createTestSkill "simple-skill" 1]
        let rollouts = [simplePlan]
        let sessionId = "simple-plan-session"
        
        // Act
        let (resultPlan, freeEnergy) = engine.EnhancedExpectedFreeEnergy(rollouts, sessionId)
        
        // Assert
        Assert.Equal<EnhancedPlan>(simplePlan, resultPlan)
        Assert.True(freeEnergy > 0.0)

    [<Fact>]
    let ``Enhanced expectedFreeEnergy should decompose complex plans`` () =
        // Arrange
        let engine = EnhancedTarsIntelligenceEngine(createTestLogger())
        let complexPlan = [
            createTestSkill "skill1" 2
            createTestSkill "skill2" 3
            createTestSkill "skill3" 2
            createTestSkill "skill4" 1
            createTestSkill "skill5" 2
        ]
        let rollouts = [complexPlan]
        let sessionId = "complex-plan-session"
        
        // Act
        let (resultPlan, freeEnergy) = engine.EnhancedExpectedFreeEnergy(rollouts, sessionId)
        
        // Assert
        Assert.True(resultPlan.Length <= complexPlan.Length)
        let metrics = engine.GetPerformanceMetrics(sessionId)
        Assert.True(metrics.tier7_decomposition_accuracy > 0.0)

    [<Fact>]
    let ``Enhanced executePlan should execute successfully`` () =
        // Arrange
        let engine = EnhancedTarsIntelligenceEngine(createTestLogger())
        let plan = [createTestSkill "executable-skill" 1]
        let sessionId = "execution-session"
        
        // Act
        let result = engine.EnhancedExecutePlan(plan, sessionId)
        
        // Assert
        let metrics = engine.GetPerformanceMetrics(sessionId)
        Assert.True(metrics.total_executions > 0L)

    [<Fact>]
    let ``Agent registration should update collective state`` () =
        // Arrange
        let engine = EnhancedTarsIntelligenceEngine(createTestLogger())
        let sessionId = "agent-registration-session"
        let position = createTestPosition 0.1 0.2 0.3 0.4
        
        // Act
        engine.RegisterAgent("test-agent", position, sessionId)
        
        // Assert
        let agents = engine.GetActiveAgents(sessionId)
        let singleAgent = Assert.Single(agents)
        let (agentId, agentPos, _) = singleAgent
        Assert.Equal("test-agent", agentId)
        Assert.Equal(position.X, agentPos.X)

    [<Fact>]
    let ``Agent unregistration should remove from collective state`` () =
        // Arrange
        let engine = EnhancedTarsIntelligenceEngine(createTestLogger())
        let sessionId = "agent-unregistration-session"
        let position = createTestPosition 0.1 0.2 0.3 0.4
        
        engine.RegisterAgent("test-agent", position, sessionId)
        
        // Act
        engine.UnregisterAgent("test-agent", sessionId)
        
        // Assert
        let agents = engine.GetActiveAgents(sessionId)
        Assert.Empty(agents)

    [<Fact>]
    let ``Performance metrics should be tracked accurately`` () =
        // Arrange
        let engine = EnhancedTarsIntelligenceEngine(createTestLogger())
        let sessionId = "metrics-session"
        let beliefs = [createTestBelief "metric belief" 0.8]
        
        // Act
        engine.EnhancedInfer(beliefs, None, sessionId) |> ignore
        
        // Assert
        let metrics = engine.GetPerformanceMetrics(sessionId)
        Assert.True(metrics.total_inferences > 0L)
        Assert.True(metrics.integration_overhead_ms >= 0.0)
        Assert.True(metrics.last_updated > DateTime.MinValue)

    [<Fact>]
    let ``Intelligence assessment should provide comprehensive status`` () =
        // Arrange
        let engine = EnhancedTarsIntelligenceEngine(createTestLogger())
        let sessionId = "assessment-session"
        
        // Register agents and perform operations
        engine.RegisterAgent("agent1", createTestPosition 0.2 0.3 0.4 0.5, sessionId)
        engine.RegisterAgent("agent2", createTestPosition 0.8 0.7 0.6 0.5, sessionId)
        
        let beliefs = [createTestBelief "assessment belief" 0.7]
        engine.EnhancedInfer(beliefs, None, sessionId) |> ignore
        
        // Act
        let assessment = engine.GetIntelligenceAssessment(sessionId)
        
        // Assert
        Assert.Equal(sessionId, assessment.session_id)
        Assert.True(assessment.tier6_collective_intelligence.active_agents >= 2)
        Assert.NotNull(assessment.tier6_collective_intelligence.status)
        Assert.NotNull(assessment.tier7_problem_decomposition.status)
        Assert.NotEmpty(assessment.honest_limitations)

    [<Fact>]
    let ``Collective intelligence should improve with agent diversity`` () =
        // Arrange
        let engine = EnhancedTarsIntelligenceEngine(createTestLogger())
        let sessionId = "diversity-session"
        
        // Register diverse agents
        engine.RegisterAgent("agent1", createTestPosition 0.1 0.1 0.1 0.1, sessionId)
        engine.RegisterAgent("agent2", createTestPosition 0.9 0.9 0.9 0.9, sessionId)
        engine.RegisterAgent("agent3", createTestPosition 0.5 0.2 0.8 0.3, sessionId)
        
        let beliefs = [createTestBelief "diversity belief" 0.6]
        
        // Act
        let result = engine.EnhancedInfer(beliefs, None, sessionId)
        
        // Assert
        let metrics = engine.GetPerformanceMetrics(sessionId)
        Assert.True(metrics.tier6_agent_efficiency > 0.0)
        Assert.True(result.[0].confidence > beliefs.[0].confidence)

    [<Fact>]
    let ``Problem decomposition should handle edge cases gracefully`` () =
        // Arrange
        let engine = EnhancedTarsIntelligenceEngine(createTestLogger())
        let emptyPlan = []
        let singleSkillPlan = [createTestSkill "single" 1]
        let sessionId = "edge-case-session"
        
        // Act & Assert - should not throw exceptions
        let (emptyResult, _) = engine.EnhancedExpectedFreeEnergy([emptyPlan], sessionId)
        let (singleResult, _) = engine.EnhancedExpectedFreeEnergy([singleSkillPlan], sessionId)
        
        Assert.Equal<EnhancedPlan>(emptyPlan, emptyResult)
        Assert.Equal<EnhancedPlan>(singleSkillPlan, singleResult)

    [<Fact>]
    let ``Session isolation should maintain separate states`` () =
        // Arrange
        let engine = EnhancedTarsIntelligenceEngine(createTestLogger())
        let session1 = "session-1"
        let session2 = "session-2"
        
        // Act
        engine.RegisterAgent("agent1", createTestPosition 0.1 0.2 0.3 0.4, session1)
        engine.RegisterAgent("agent2", createTestPosition 0.5 0.6 0.7 0.8, session2)
        
        // Assert
        let agents1 = engine.GetActiveAgents(session1)
        let agents2 = engine.GetActiveAgents(session2)
        
        let singleAgent1 = Assert.Single(agents1)
        let singleAgent2 = Assert.Single(agents2)
        Assert.NotEqual(singleAgent1, singleAgent2)

    [<Fact>]
    let ``Error handling should provide graceful degradation`` () =
        // Arrange
        let engine = EnhancedTarsIntelligenceEngine(createTestLogger())
        let sessionId = "error-session"
        
        // Create a skill that will fail checker
        let failingSkill = { createTestSkill "failing" 1 with checker = fun () -> false }
        let plan = [failingSkill]
        
        // Act & Assert - should not throw exceptions
        let result = engine.EnhancedExecutePlan(plan, sessionId)
        
        // Should handle gracefully
        let metrics = engine.GetPerformanceMetrics(sessionId)
        Assert.True(metrics.total_executions > 0L)
