module TarsEngine.DSL.Tests

open System
open Xunit
open TarsEngine.DSL
open TarsEngine.DSL.Ast
open TarsEngine.DSL.Parser
open TarsEngine.DSL.Interpreter

[<Fact>]
let ``Parser can parse CONFIG block`` () =
    let code = """CONFIG {
        model: "llama3"
        temperature: 0.7
        max_tokens: 1000
    }"""

    let program = Parser.parse code

    Assert.Equal(1, program.Blocks.Length)
    Assert.Equal(BlockType.Config, program.Blocks.[0].Type)
    Assert.Equal(3, program.Blocks.[0].Properties.Count)

    match program.Blocks.[0].Properties.TryFind("model") with
    | Some(StringValue(model)) -> Assert.Equal("llama3", model)
    | _ -> Assert.True(false, "model property not found or not a string")

    match program.Blocks.[0].Properties.TryFind("temperature") with
    | Some(NumberValue(temp)) -> Assert.Equal(0.7, temp)
    | _ -> Assert.True(false, "temperature property not found or not a number")

    match program.Blocks.[0].Properties.TryFind("max_tokens") with
    | Some(NumberValue(maxTokens)) -> Assert.Equal(1000.0, maxTokens)
    | _ -> Assert.True(false, "max_tokens property not found or not a number")

[<Fact>]
let ``Parser can parse PROMPT block`` () =
    let code = """PROMPT {
        text: "What is the capital of France?"
        model: "llama3"
    }"""

    let program = Parser.parse code

    Assert.Equal(1, program.Blocks.Length)
    Assert.Equal(BlockType.Prompt, program.Blocks.[0].Type)
    Assert.Equal(2, program.Blocks.[0].Properties.Count)

    match program.Blocks.[0].Properties.TryFind("text") with
    | Some(StringValue(text)) -> Assert.Equal("What is the capital of France?", text)
    | _ -> Assert.True(false, "text property not found or not a string")

[<Fact>]
let ``Parser can parse multiple blocks`` () =
    let code = """CONFIG {
        model: "llama3"
        temperature: 0.7
    }

    PROMPT {
        text: "What is the capital of France?"
    }

    ACTION {
        type: "generate"
        model: "llama3"
    }"""

    let program = Parser.parse code

    Assert.Equal(3, program.Blocks.Length)
    Assert.Equal(BlockType.Config, program.Blocks.[0].Type)
    Assert.Equal(BlockType.Prompt, program.Blocks.[1].Type)
    Assert.Equal(BlockType.Action, program.Blocks.[2].Type)

[<Fact>]
let ``Interpreter can execute CONFIG block`` () =
    let code = """CONFIG {
        model: "llama3"
        temperature: 0.7
        max_tokens: 1000
    }"""

    let program = Parser.parse code
    let result = Interpreter.execute program

    match result with
    | Success(StringValue(msg)) -> Assert.Equal("Program executed successfully", msg)
    | _ -> Assert.True(false, "Execution failed or returned unexpected result")
