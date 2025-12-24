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
        | _ -> Assert.Fail("Expected ToolCall response")

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
        | _ -> Assert.Fail("Expected TextResponse")

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
        | _ -> Assert.Fail("Expected ToolCall")

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
        | _ -> Assert.Fail("Expected TextResponse")

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
        | _ -> Assert.Fail("Expected ToolCall")

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
        | _ -> Assert.Fail("Expected TextResponse")

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
        Assert.Contains("Conversation History", prompt)
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
                Intent = None
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
                Intent = None
                Constraints = Tars.Core.SemanticConstraints.Default
                Ontology = None
                Language = "text"
                Content = "Hi! How can I help?"
                Timestamp = DateTime.UtcNow
                Metadata = Map.empty } ]

        // Act
        let prompt = PromptBuilder.buildSystemPrompt agent history

        // Assert
        Assert.Contains("User [Inform]: Hello there!", prompt)
        Assert.Contains("Assistant [Inform]: Hi! How can I help?", prompt)
        output.WriteLine("History correctly included in prompt")

    // New Tool Call Format Tests

    [<Fact>]
    member _.``ResponseParser: Parses JSON block tool call``() =
        output.WriteLine("Starting test: Parses JSON block tool call")

        let response =
            """Here is a tool call:
```tool
{"name": "Calculator", "arguments": {"expression": "2+2"}}
```"""

        let result = ResponseParser.parse response

        match result with
        | ResponseParser.ToolCall(name, input) ->
            Assert.Equal("Calculator", name)
            Assert.Contains("2+2", input)
            output.WriteLine($"Correctly parsed JSON block tool call: {name}")
        | _ -> Assert.Fail("Expected ToolCall from JSON block")

    [<Fact>]
    member _.``ResponseParser: Parses XML-style tool call``() =
        output.WriteLine("Starting test: Parses XML-style tool call")

        let response =
            """I'll search for that:
<tool_call>{"name": "WebSearch", "arguments": {"query": "F# async programming"}}</tool_call>"""

        let result = ResponseParser.parse response

        match result with
        | ResponseParser.ToolCall(name, input) ->
            Assert.Equal("WebSearch", name)
            Assert.Contains("F# async programming", input)
            output.WriteLine($"Correctly parsed XML-style tool call: {name}")
        | _ -> Assert.Fail("Expected ToolCall from XML-style")

    [<Fact>]
    member _.``ResponseParser: Parses inline JSON tool call``() =
        output.WriteLine("Starting test: Parses inline JSON tool call")
        let response = """{"name": "ReadFile", "arguments": {"path": "/tmp/test.txt"}}"""

        let result = ResponseParser.parse response

        match result with
        | ResponseParser.ToolCall(name, input) ->
            Assert.Equal("ReadFile", name)
            Assert.Contains("/tmp/test.txt", input)
            output.WriteLine($"Correctly parsed inline JSON tool call: {name}")
        | _ -> Assert.Fail("Expected ToolCall from inline JSON")

    [<Fact>]
    member _.``ResponseParser: Parses multi-tool call with inline JSON``() =
        output.WriteLine("Starting test: Parses multi-tool call with inline JSON")

        let response =
            """I'll use multiple tools:
{"name": "Calculator", "arguments": {"expression": "2+2"}}
{"name": "WebSearch", "arguments": {"query": "F# tutorials"}}"""

        let result = ResponseParser.parse response

        match result with
        | ResponseParser.MultiToolCall calls ->
            Assert.Equal(2, calls.Length)
            let (name1, _) = calls.[0]
            let (name2, _) = calls.[1]
            Assert.Equal("Calculator", name1)
            Assert.Equal("WebSearch", name2)
            output.WriteLine($"Correctly parsed {calls.Length} tool calls")
        | _ -> Assert.Fail("Expected MultiToolCall")

    [<Fact>]
    member _.``ResponseParser: Single tool call not treated as multi``() =
        output.WriteLine("Starting test: Single tool call not treated as multi")
        let response = "TOOL:Calculator:5*5"

        let result = ResponseParser.parse response

        match result with
        | ResponseParser.ToolCall(name, _) ->
            Assert.Equal("Calculator", name)
            output.WriteLine("Single tool correctly parsed as ToolCall, not MultiToolCall")
        | ResponseParser.MultiToolCall _ -> Assert.Fail("Single tool should not be MultiToolCall")
        | _ -> Assert.Fail("Expected ToolCall")

    [<Fact>]
    member _.``PromptBuilder: Includes tools in system prompt``() =
        output.WriteLine("Starting test: Includes tools in system prompt")

        let tool: Tars.Core.Tool =
            { Name = "Calculator"
              Description = "Performs arithmetic calculations"
              Version = "1.0.0"
              ParentVersion = None
              CreatedAt = DateTime.UtcNow
              Execute = fun _ -> async { return Ok "4" }
              ThingDescription = None }

        let agent: Tars.Core.Agent =
            { Id = Tars.Core.AgentId(Guid.NewGuid())
              Name = "TestAgent"
              Version = "1.0.0"
              ParentVersion = None
              CreatedAt = DateTime.UtcNow
              Model = "test-model"
              SystemPrompt = "You are a helpful assistant."
              Tools = [ tool ]
              Capabilities = []
              Memory = []
              State = Tars.Core.Idle }

        let prompt = PromptBuilder.buildSystemPrompt agent []

        Assert.Contains("Calculator", prompt)
        Assert.Contains("Performs arithmetic calculations", prompt)
        Assert.Contains("Available Tools", prompt)
        output.WriteLine("Tools correctly included in system prompt")
