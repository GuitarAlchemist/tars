module TarsEngine.DSL.Tests.SimpleDslUnitTests

open System
open Xunit
open TarsEngine.DSL.SimpleDsl

[<Fact>]
let ``Parse simple block`` () =
    let blockText = """DESCRIBE {
    name: "Test Block"
    description: "A test block"
}"""
    
    let block = parseBlock blockText
    
    Assert.Equal(BlockType.Describe, block.Type)
    Assert.Equal(None, block.Name)
    Assert.Equal(2, block.Properties.Count)
    Assert.Equal(StringValue("Test Block"), block.Properties.["name"])
    Assert.Equal(StringValue("A test block"), block.Properties.["description"])
    Assert.Empty(block.NestedBlocks)

[<Fact>]
let ``Parse block with name`` () =
    let blockText = """VARIABLE message {
    value: "Hello, World!"
}"""
    
    let block = parseBlock blockText
    
    Assert.Equal(BlockType.Variable, block.Type)
    Assert.Equal(Some "message", block.Name)
    Assert.Equal(1, block.Properties.Count)
    Assert.Equal(StringValue("Hello, World!"), block.Properties.["value"])
    Assert.Empty(block.NestedBlocks)

[<Fact>]
let ``Parse nested blocks`` () =
    let blockText = """IF {
    condition: "true"
    
    ACTION {
        type: "log"
        message: "This is a test"
    }
}"""
    
    let block = parseBlock blockText
    
    Assert.Equal(BlockType.If, block.Type)
    Assert.Equal(None, block.Name)
    Assert.Equal(1, block.Properties.Count)
    Assert.Equal(StringValue("true"), block.Properties.["condition"])
    Assert.Equal(1, block.NestedBlocks.Length)
    
    let nestedBlock = block.NestedBlocks.[0]
    Assert.Equal(BlockType.Action, nestedBlock.Type)
    Assert.Equal(None, nestedBlock.Name)
    Assert.Equal(2, nestedBlock.Properties.Count)
    Assert.Equal(StringValue("log"), nestedBlock.Properties.["type"])
    Assert.Equal(StringValue("This is a test"), nestedBlock.Properties.["message"])

[<Fact>]
let ``Parse program`` () =
    let programText = """DESCRIBE {
    name: "Test Program"
    description: "A test program"
}

VARIABLE message {
    value: "Hello, World!"
}

ACTION {
    type: "log"
    message: "${message}"
}"""
    
    let program = parseProgram programText
    
    Assert.Equal(3, program.Blocks.Length)
    
    let describeBlock = program.Blocks.[0]
    Assert.Equal(BlockType.Describe, describeBlock.Type)
    
    let variableBlock = program.Blocks.[1]
    Assert.Equal(BlockType.Variable, variableBlock.Type)
    Assert.Equal(Some "message", variableBlock.Name)
    
    let actionBlock = program.Blocks.[2]
    Assert.Equal(BlockType.Action, actionBlock.Type)
    Assert.Equal(StringValue("log"), actionBlock.Properties.["type"])
    Assert.Equal(StringValue("${message}"), actionBlock.Properties.["message"])

[<Fact>]
let ``Execute program with variable substitution`` () =
    let programText = """VARIABLE message {
    value: "Hello, World!"
}

ACTION {
    type: "log"
    message: "${message}"
}"""
    
    let program = parseProgram programText
    let result = executeProgram program
    
    match result with
    | Success value ->
        match value with
        | StringValue message -> Assert.Contains("Hello, World!", message)
        | _ -> Assert.True(false, "Expected StringValue")
    | Error msg -> Assert.True(false, $"Expected Success but got Error: {msg}")

[<Fact>]
let ``Execute program with if condition true`` () =
    let programText = """VARIABLE message {
    value: "Hello, World!"
}

IF {
    condition: "${message == 'Hello, World!'}"
    
    ACTION {
        type: "log"
        message: "Condition is true"
    }
}"""
    
    let program = parseProgram programText
    let result = executeProgram program
    
    match result with
    | Success value ->
        match value with
        | StringValue message -> Assert.Contains("Condition is true", message)
        | _ -> Assert.True(false, "Expected StringValue")
    | Error msg -> Assert.True(false, $"Expected Success but got Error: {msg}")

[<Fact>]
let ``Execute program with if condition false`` () =
    let programText = """VARIABLE message {
    value: "Hello, World!"
}

IF {
    condition: "${message == 'Goodbye, World!'}"
    
    ACTION {
        type: "log"
        message: "Condition is true"
    }
}
ELSE {
    ACTION {
        type: "log"
        message: "Condition is false"
    }
}"""
    
    let program = parseProgram programText
    let result = executeProgram program
    
    match result with
    | Success value ->
        match value with
        | StringValue message -> Assert.Contains("Condition is false", message)
        | _ -> Assert.True(false, "Expected StringValue")
    | Error msg -> Assert.True(false, $"Expected Success but got Error: {msg}")

[<Fact>]
let ``Execute program with mcp_send action`` () =
    let programText = """VARIABLE target {
    value: "augment"
}

ACTION {
    type: "mcp_send"
    target: "${target}"
    action: "code_generation"
    result_variable: "response"
}

ACTION {
    type: "log"
    message: "${response}"
}"""
    
    let program = parseProgram programText
    let result = executeProgram program
    
    match result with
    | Success value ->
        match value with
        | StringValue message -> 
            Assert.Contains("MCP request to augment", message)
            Assert.Contains("action: code_generation", message)
        | _ -> Assert.True(false, "Expected StringValue")
    | Error msg -> Assert.True(false, $"Expected Success but got Error: {msg}")

[<Fact>]
let ``Execute program with mcp_receive action`` () =
    let programText = """ACTION {
    type: "mcp_receive"
    timeout: 10
    result_variable: "request"
}

ACTION {
    type: "log"
    message: "${request}"
}"""
    
    let program = parseProgram programText
    let result = executeProgram program
    
    match result with
    | Success value ->
        match value with
        | StringValue message -> 
            Assert.Contains("Received MCP request", message)
            Assert.Contains("timeout: 10", message)
        | _ -> Assert.True(false, "Expected StringValue")
    | Error msg -> Assert.True(false, $"Expected Success but got Error: {msg}")
