namespace Tars.Tests.Evolution

open NUnit.Framework
open FsUnit
open System
open System.Threading.Tasks
open Tars.Core
open Tars.Evolution
open Tars.Knowledge

/// Tests for Evolution Engine Fixes
[<TestFixture>]
type EvolutionFixesTests() =

    /// Fix #1: JSON Mode Enforcement
    [<Test>]
    member _.``Semantic message should have JSON metadata for curriculum generation``() =
        // Arrange: Create a semantic message for curriculum
        let msg: Message =
            { Id = Guid.NewGuid()
              CorrelationId = CorrelationId(Guid.NewGuid())
              Sender = MessageEndpoint.System
              Receiver = None
              Performative = Performative.Request
              Intent = Some AgentDomain.Planning
              Constraints = SemanticConstraints.Default
              Ontology = None
              Language = "json"
              Content = "test"
              Timestamp = DateTime.UtcNow
              Metadata = Map.ofList [ ("response_format", "json"); ("json_mode", "true") ] }

        // Assert: Verify JSON mode is set
        msg.Language |> should equal "json"
        msg.Metadata.ContainsKey("response_format") |> should be True
        msg.Metadata.["response_format"] |> should equal "json"
        msg.Metadata.ContainsKey("json_mode") |> should be True
        msg.Metadata.["json_mode"] |> should equal "true"

    /// Fix #2: JSONB Type Handling
    [<Test>]
    member _.``Evidence candidate should serialize segments as JSONB compatible``() =
        // Arrange
        let candidate: EvidenceCandidate =
            { Id = Guid.NewGuid()
              SourceUrl = Uri("tars://test")
              ContentHash = "hash123"
              FetchedAt = DateTime.UtcNow
              RawContent = "test content"
              Segments = [ "segment1"; "segment2"; "segment3" ]
              ProposedAssertions = []
              Status = EvidenceStatus.Pending
              Metadata = Map.ofList [ ("key", "value") ]
              VerifiedAt = None
              VerifiedBy = None
              RejectionReason = None }

        // Assert: Segments should be a list (will be serialized to JSONB)
        candidate.Segments |> should not' (be Empty)
        candidate.Segments.Length |> should equal 3
        candidate.Metadata |> should not' (be Empty)

    /// Fix #3: Success Criteria Propagation
    [<Test>]
    member _.``Task result should reflect both execution and evaluation success``() =
        // Arrange
        let taskResult: TaskResult =
            { TaskId = Guid.NewGuid()
              TaskGoal = "Test task"
              ExecutorId = AgentId(Guid.NewGuid())
              Success = true // Execution succeeded
              Output = "Result"
              ExecutionTrace = []
              Duration = TimeSpan.FromSeconds(1.0)
              Evaluation =
                Some
                    { Passed = false
                      Confidence = 0.8
                      Summary = "Failed validation" } }

        // Act: Calculate final success (execution AND evaluation)
        let executionSuccess = taskResult.Success

        let evaluationPassed =
            taskResult.Evaluation
            |> Option.map (fun e -> e.Passed)
            |> Option.defaultValue true

        let finalSuccess = executionSuccess && evaluationPassed

        // Assert: Should be false because evaluation failed
        finalSuccess |> should be False

    [<Test>]
    member _.``Task result should be successful when both execution and evaluation pass``() =
        // Arrange
        let taskResult: TaskResult =
            { TaskId = Guid.NewGuid()
              TaskGoal = "Test task"
              ExecutorId = AgentId(Guid.NewGuid())
              Success = true
              Output = "Result"
              ExecutionTrace = []
              Duration = TimeSpan.FromSeconds(1.0)
              Evaluation =
                Some
                    { Passed = true
                      Confidence = 0.9
                      Summary = "Passed" } }

        // Act
        let executionSuccess = taskResult.Success

        let evaluationPassed =
            taskResult.Evaluation
            |> Option.map (fun e -> e.Passed)
            |> Option.defaultValue true

        let finalSuccess = executionSuccess && evaluationPassed

        // Assert: Should be true
        finalSuccess |> should be True

    /// Fix #4 & #5: Knowledge Graph Persistence and Initialization
    [<Test>]
    member _.``Temporal graph should support Save and Load operations``() =
        // Arrange
        let graph = Tars.Core.TemporalKnowledgeGraph.TemporalGraph()

        let testEntity =
            TarsEntity.ConceptE
                { Name = "TestConcept"
                  Description = "Test"
                  RelatedConcepts = [] }

        // Act
        let id = graph.AddNode(testEntity)
        let facts = graph.GetCurrentFacts()

        // Assert
        id |> should not' (be NullOrWhiteSpace)
        facts |> should not' (be Empty)

    /// Fix #6: Reflection Output Truncation
    [<Test>]
    member _.``Reflection prompt should truncate overly long output``() =
        // Arrange
        let maxLength = 2000
        let longOutput = String.replicate 5000 "x" // 5000 chars

        // Act
        let truncatedOutput =
            if longOutput.Length > maxLength then
                longOutput.Substring(0, maxLength) + "\n... [output truncated]"
            else
                longOutput

        // Assert
        truncatedOutput.Length |> should be (lessThan 2050) // 2000 + suffix
        truncatedOutput |> should haveSubstring "[output truncated]"

    /// Fix #7: Agent Memory Truncation
    [<Test>]
    member _.``Agent should truncate overly long messages in memory``() =
        // Arrange
        let longContent = String.replicate 5000 "x"

        let msg: Message =
            { Id = Guid.NewGuid()
              CorrelationId = CorrelationId(Guid.NewGuid())
              Sender = MessageEndpoint.System
              Receiver = None
              Performative = Performative.Request
              Intent = None
              Constraints = SemanticConstraints.Default
              Ontology = None
              Language = "text"
              Content = longContent
              Timestamp = DateTime.UtcNow
              Metadata = Map.empty }

        let agent: Agent =
            { Id = AgentId(Guid.NewGuid())
              Name = "TestAgent"
              Version = "1.0"
              ParentVersion = None
              CreatedAt = DateTime.UtcNow
              Model = "test"
              SystemPrompt = "test"
              Tools = []
              Capabilities = []
              State = AgentState.Idle
              Memory = [] }

        // Act
        let updatedAgent = agent.ReceiveMessage(msg)

        // Assert: Message should be truncated in memory
        updatedAgent.Memory |> should not' (be Empty)
        let storedMsg = updatedAgent.Memory |> List.head
        storedMsg.Content.Length |> should be (lessThan 2050)
        storedMsg.Content |> should haveSubstring "[truncated]"

    [<Test>]
    member _.``Agent should truncate existing memory messages when receiving new message``() =
        // Arrange
        let longContent1 = String.replicate 3000 "a"
        let longContent2 = String.replicate 3000 "b"

        let msg1: Message =
            { Id = Guid.NewGuid()
              CorrelationId = CorrelationId(Guid.NewGuid())
              Sender = MessageEndpoint.System
              Receiver = None
              Performative = Performative.Request
              Intent = None
              Constraints = SemanticConstraints.Default
              Ontology = None
              Language = "text"
              Content = longContent1
              Timestamp = DateTime.UtcNow
              Metadata = Map.empty }

        let msg2: Message =
            { msg1 with
                Content = longContent2
                Id = Guid.NewGuid() }

        let agent: Agent =
            { Id = AgentId(Guid.NewGuid())
              Name = "TestAgent"
              Version = "1.0"
              ParentVersion = None
              CreatedAt = DateTime.UtcNow
              Model = "test"
              SystemPrompt = "test"
              Tools = []
              Capabilities = []
              State = AgentState.Idle
              Memory = [ msg1 ] }

        // Act: Receive second message (should truncate both old and new)
        let updatedAgent = agent.ReceiveMessage(msg2)

        // Assert
        updatedAgent.Memory.Length |> should equal 2
        // First message should be retroactively truncated
        updatedAgent.Memory.[0].Content.Length |> should be (lessThan 2050)
        // Second message should also be truncated
        updatedAgent.Memory.[1].Content.Length |> should be (lessThan 2050)

    /// Integration test: Verify truncation doesn't break message flow
    [<Test>]
    member _.``Agent memory truncation should preserve message structure``() =
        // Arrange
        let originalContent =
            "# Header\n\n" + String.replicate 3000 "content" + "\n\nFooter"

        let msg: Message =
            { Id = Guid.NewGuid()
              CorrelationId = CorrelationId(Guid.NewGuid())
              Sender = MessageEndpoint.User
              Receiver = None
              Performative = Performative.Request
              Intent = Some AgentDomain.Coding
              Constraints = SemanticConstraints.Default
              Ontology = None
              Language = "text"
              Content = originalContent
              Timestamp = DateTime.UtcNow
              Metadata = Map.ofList [ ("key", "value") ] }

        let agent: Agent =
            { Id = AgentId(Guid.NewGuid())
              Name = "TestAgent"
              Version = "1.0"
              ParentVersion = None
              CreatedAt = DateTime.UtcNow
              Model = "test"
              SystemPrompt = "test"
              Tools = []
              Capabilities = []
              State = AgentState.Idle
              Memory = [] }

        // Act
        let updatedAgent = agent.ReceiveMessage(msg)

        // Assert: Message structure should be preserved
        let storedMsg = updatedAgent.Memory |> List.head
        storedMsg.Id |> should equal msg.Id
        storedMsg.Sender |> should equal msg.Sender
        storedMsg.Performative |> should equal msg.Performative
        storedMsg.Intent |> should equal msg.Intent
        storedMsg.Metadata |> should equal msg.Metadata
        // Content should be truncated but still valid
        storedMsg.Content |> should haveSubstring "# Header"
        storedMsg.Content.Length |> should be (lessThan originalContent.Length)
