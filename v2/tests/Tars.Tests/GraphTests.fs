namespace Tars.Tests

open System
open Xunit
open Xunit.Abstractions
open Tars.Graph

type GraphTests(output: ITestOutputHelper) =

    // ResponseParser Tests

    [<Fact>]
    member _.``ResponseParser: Parses tool call correctly``() =
        output.WriteLine("Starting test: Parses tool call correctly")
        // Arrange
        let response = "TOOL:Calculator:2+2"

        // Act
        let result = ResponseParser.parse response

        // Assert
        match result with
        | ResponseParser.ToolCall(name, input) ->
            Assert.Equal("Calculator", name)
            Assert.Equal("2+2", input)
            output.WriteLine($"Correctly parsed tool call: {name} with input: {input}")
        | ResponseParser.TextResponse text -> Assert.Fail($"Expected ToolCall but got TextResponse: {text}")

    [<Fact>]
    member _.``ResponseParser: Parses text response correctly``() =
        output.WriteLine("Starting test: Parses text response correctly")
        // Arrange
        let response = "Hello, how can I help you today?"

        // Act
        let result = ResponseParser.parse response

        // Assert
        match result with
        | ResponseParser.TextResponse text ->
            Assert.Equal("Hello, how can I help you today?", text)
            output.WriteLine($"Correctly parsed text response: {text}")
        | ResponseParser.ToolCall(name, _) -> Assert.Fail($"Expected TextResponse but got ToolCall: {name}")

    [<Fact>]
    member _.``ResponseParser: Handles tool call with colons in input``() =
        output.WriteLine("Starting test: Handles tool call with colons in input")
        // Arrange
        let response = "TOOL:WebSearch:query:with:colons"

        // Act
        let result = ResponseParser.parse response

        // Assert
        match result with
        | ResponseParser.ToolCall(name, input) ->
            Assert.Equal("WebSearch", name)
            Assert.Equal("query:with:colons", input)
            output.WriteLine($"Correctly handled colons in input: {input}")
        | ResponseParser.TextResponse text -> Assert.Fail($"Expected ToolCall but got TextResponse: {text}")

    [<Fact>]
    member _.``ResponseParser: Treats incomplete tool format as text``() =
        output.WriteLine("Starting test: Treats incomplete tool format as text")
        // Arrange
        let response = "TOOL:OnlyName"

        // Act
        let result = ResponseParser.parse response

        // Assert
        match result with
        | ResponseParser.TextResponse text ->
            Assert.Equal("TOOL:OnlyName", text)
            output.WriteLine("Correctly treated incomplete tool format as text")
        | ResponseParser.ToolCall(name, _) -> Assert.Fail($"Expected TextResponse but got ToolCall: {name}")

    [<Fact>]
    member _.``ResponseParser: Trims whitespace from response``() =
        output.WriteLine("Starting test: Trims whitespace from response")
        // Arrange
        let response = "   TOOL:Calculator:5*5   "

        // Act
        let result = ResponseParser.parse response

        // Assert
        match result with
        | ResponseParser.ToolCall(name, input) ->
            Assert.Equal("Calculator", name)
            Assert.Equal("5*5", input)
            output.WriteLine("Correctly trimmed whitespace and parsed tool call")
        | ResponseParser.TextResponse text -> Assert.Fail($"Expected ToolCall but got TextResponse: {text}")

    [<Fact>]
    member _.``ResponseParser: Case sensitive TOOL prefix``() =
        output.WriteLine("Starting test: Case sensitive TOOL prefix")
        // Arrange
        let response = "tool:Calculator:2+2"

        // Act
        let result = ResponseParser.parse response

        // Assert
        match result with
        | ResponseParser.TextResponse text ->
            Assert.Equal("tool:Calculator:2+2", text)
            output.WriteLine("Correctly treated lowercase 'tool' as text response")
        | ResponseParser.ToolCall(name, _) -> Assert.Fail($"Expected TextResponse but got ToolCall: {name}")

    // PromptBuilder Tests

    [<Fact>]
    member _.``PromptBuilder: Builds system prompt with agent info``() =
        output.WriteLine("Starting test: Builds system prompt with agent info")
        // Arrange
        let agent: Tars.Core.Agent =
            { Id = Tars.Core.AgentId(Guid.NewGuid())
              Name = "TestAgent"
              Version = "1.0.0"
              ParentVersion = None
              CreatedAt = DateTime.UtcNow
              Model = "test-model"
              SystemPrompt = "You are a helpful assistant."
              Tools = []
              Capabilities = []
              Memory = []
              State = Tars.Core.Idle }

        let history: Tars.Core.Message list = []

        // Act
        let prompt = PromptBuilder.buildSystemPrompt agent history

        // Assert
        Assert.Contains("You are a helpful assistant.", prompt)
        Assert.Contains("History:", prompt)
        Assert.Contains("Instructions:", prompt)
        output.WriteLine("Prompt built correctly with agent system prompt")

    [<Fact>]
    member _.``PromptBuilder: Includes message history``() =
        output.WriteLine("Starting test: Includes message history")
        // Arrange
        let agent: Tars.Core.Agent =
            { Id = Tars.Core.AgentId(Guid.NewGuid())
              Name = "TestAgent"
              Version = "1.0.0"
              ParentVersion = None
              CreatedAt = DateTime.UtcNow
              Model = "test-model"
              SystemPrompt = "You are a helpful assistant."
              Tools = []
              Capabilities = []
              Memory = []
              State = Tars.Core.Idle }

        let history: Tars.Core.Message list =
            [ { Id = Guid.NewGuid()
                CorrelationId = Tars.Core.CorrelationId(Guid.NewGuid())
                Sender = Tars.Core.User
                Receiver = Some Tars.Core.System
                Performative = Tars.Core.Inform
                Constraints = Tars.Core.SemanticConstraints.Default
                Ontology = None
                Language = "text"
                Content = "Hello there!"
                Timestamp = DateTime.UtcNow
                Metadata = Map.empty }
              { Id = Guid.NewGuid()
                CorrelationId = Tars.Core.CorrelationId(Guid.NewGuid())
                Sender = Tars.Core.Agent(agent.Id)
                Receiver = Some Tars.Core.User
                Performative = Tars.Core.Inform
                Constraints = Tars.Core.SemanticConstraints.Default
                Ontology = None
                Language = "text"
                Content = "Hi! How can I help?"
                Timestamp = DateTime.UtcNow
                Metadata = Map.empty } ]

        // Act
        let prompt = PromptBuilder.buildSystemPrompt agent history

        // Assert
        Assert.Contains("User: Hello there!", prompt)
        Assert.Contains("Assistant: Hi! How can I help?", prompt)
        output.WriteLine("History correctly included in prompt")
