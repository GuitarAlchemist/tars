namespace Tars.Tests

open System
open System.IO
open Xunit
open Tars.Llm

module ConstrainedDecodingTests =

    // =========================================================================
    // Helpers
    // =========================================================================

    let private withTempGrammarDir (grammars: (string * string) list) (f: string -> 'a) =
        let dir = Path.Combine(Path.GetTempPath(), $"tars_test_grammars_{Guid.NewGuid():N}")
        Directory.CreateDirectory(dir) |> ignore
        try
            for (name, content) in grammars do
                File.WriteAllText(Path.Combine(dir, name), content)
            f dir
        finally
            Directory.Delete(dir, true)

    // =========================================================================
    // Grammar loading
    // =========================================================================

    [<Fact>]
    let ``loadEbnfGrammar loads existing grammar file`` () =
        withTempGrammarDir [ ("test.ebnf", "root ::= 'hello'") ] (fun dir ->
            let result = ConstrainedDecoding.loadEbnfGrammar dir "test"
            Assert.Equal(Ok "root ::= 'hello'", result)
        )

    [<Fact>]
    let ``loadEbnfGrammar loads with explicit .ebnf extension`` () =
        withTempGrammarDir [ ("test.ebnf", "root ::= 'world'") ] (fun dir ->
            let result = ConstrainedDecoding.loadEbnfGrammar dir "test.ebnf"
            Assert.Equal(Ok "root ::= 'world'", result)
        )

    [<Fact>]
    let ``loadEbnfGrammar returns Error for missing file`` () =
        withTempGrammarDir [] (fun dir ->
            let result = ConstrainedDecoding.loadEbnfGrammar dir "nonexistent"
            match result with
            | Error msg -> Assert.Contains("not found", msg)
            | Ok _ -> Assert.Fail("Expected Error")
        )

    [<Fact>]
    let ``loadJsonSchema loads existing schema file`` () =
        let path = Path.GetTempFileName()
        try
            File.WriteAllText(path, """{"type": "object"}""")
            let result = ConstrainedDecoding.loadJsonSchema path
            Assert.Equal(Ok """{"type": "object"}""", result)
        finally
            File.Delete(path)

    [<Fact>]
    let ``loadJsonSchema returns Error for missing file`` () =
        let result = ConstrainedDecoding.loadJsonSchema "/nonexistent/schema.json"
        match result with
        | Error msg -> Assert.Contains("not found", msg)
        | Ok _ -> Assert.Fail("Expected Error")

    [<Fact>]
    let ``listGrammars lists all .ebnf files`` () =
        withTempGrammarDir [ ("a.ebnf", ""); ("b.ebnf", ""); ("c.txt", "") ] (fun dir ->
            let grammars = ConstrainedDecoding.listGrammars dir |> List.sort
            Assert.Equal<string list>([ "a"; "b" ], grammars)
        )

    [<Fact>]
    let ``listGrammars returns empty for nonexistent directory`` () =
        let grammars = ConstrainedDecoding.listGrammars "/nonexistent/dir"
        Assert.Empty(grammars)

    // =========================================================================
    // Request construction
    // =========================================================================

    [<Fact>]
    let ``withEbnfGrammar sets Ebnf response format`` () =
        let req = ConstrainedDecoding.withEbnfGrammar "root ::= 'x'" LlmRequest.Default
        Assert.Equal(Some (ResponseFormat.Constrained (Grammar.Ebnf "root ::= 'x'")), req.ResponseFormat)

    [<Fact>]
    let ``withJsonSchema sets JsonSchema response format`` () =
        let schema = """{"type": "string"}"""
        let req = ConstrainedDecoding.withJsonSchema schema LlmRequest.Default
        Assert.Equal(Some (ResponseFormat.Constrained (Grammar.JsonSchema schema)), req.ResponseFormat)

    [<Fact>]
    let ``withRegex sets Regex response format`` () =
        let req = ConstrainedDecoding.withRegex "[0-9]+" LlmRequest.Default
        Assert.Equal(Some (ResponseFormat.Constrained (Grammar.Regex "[0-9]+")), req.ResponseFormat)

    [<Fact>]
    let ``withEbnfGrammar preserves other request fields`` () =
        let req = { LlmRequest.Default with MaxTokens = Some 500; Temperature = Some 0.7 }
        let constrained = ConstrainedDecoding.withEbnfGrammar "g" req
        Assert.Equal(Some 500, constrained.MaxTokens)
        Assert.Equal(Some 0.7, constrained.Temperature)

    [<Fact>]
    let ``withNamedGrammar loads and attaches grammar`` () =
        withTempGrammarDir [ ("mygrammar.ebnf", "root ::= digit+") ] (fun dir ->
            let result = ConstrainedDecoding.withNamedGrammar dir "mygrammar" LlmRequest.Default
            match result with
            | Ok req ->
                Assert.Equal(Some (ResponseFormat.Constrained (Grammar.Ebnf "root ::= digit+")), req.ResponseFormat)
            | Error err -> Assert.Fail($"Expected Ok, got Error: {err}")
        )

    [<Fact>]
    let ``withNamedGrammar returns Error for missing grammar`` () =
        withTempGrammarDir [] (fun dir ->
            let result = ConstrainedDecoding.withNamedGrammar dir "missing" LlmRequest.Default
            match result with
            | Error msg -> Assert.Contains("not found", msg)
            | Ok _ -> Assert.Fail("Expected Error")
        )

    // =========================================================================
    // Convenience constructors
    // =========================================================================

    [<Fact>]
    let ``jsonConstrained creates request with JsonSchema format`` () =
        let msgs = [ { Role = Role.User; Content = "test" } ]
        let schema = """{"type": "number"}"""
        let req = ConstrainedDecoding.jsonConstrained schema msgs
        Assert.Equal<LlmMessage list>(msgs, req.Messages)
        Assert.Equal(Some (ResponseFormat.Constrained (Grammar.JsonSchema schema)), req.ResponseFormat)

    [<Fact>]
    let ``ebnfConstrained creates request with Ebnf format`` () =
        let msgs = [ { Role = Role.User; Content = "generate" } ]
        let grammar = "root ::= [a-z]+"
        let req = ConstrainedDecoding.ebnfConstrained grammar msgs
        Assert.Equal<LlmMessage list>(msgs, req.Messages)
        Assert.Equal(Some (ResponseFormat.Constrained (Grammar.Ebnf grammar)), req.ResponseFormat)

    [<Fact>]
    let ``cortexConstrained loads cortex grammar`` () =
        withTempGrammarDir [ ("cortex.ebnf", "cortex ::= block+") ] (fun dir ->
            let msgs = [ { Role = Role.User; Content = "plan" } ]
            let result = ConstrainedDecoding.cortexConstrained dir msgs
            match result with
            | Ok req ->
                Assert.Equal<LlmMessage list>(msgs, req.Messages)
                Assert.Equal(Some (ResponseFormat.Constrained (Grammar.Ebnf "cortex ::= block+")), req.ResponseFormat)
            | Error err -> Assert.Fail($"Expected Ok: {err}")
        )

    [<Fact>]
    let ``cortexConstrained returns Error when cortex grammar missing`` () =
        withTempGrammarDir [] (fun dir ->
            let msgs = [ { Role = Role.User; Content = "plan" } ]
            let result = ConstrainedDecoding.cortexConstrained dir msgs
            match result with
            | Error msg -> Assert.Contains("not found", msg)
            | Ok _ -> Assert.Fail("Expected Error")
        )

    // =========================================================================
    // IR schemas
    // =========================================================================

    [<Fact>]
    let ``intentPlanSchema is valid JSON`` () =
        let parsed = System.Text.Json.JsonDocument.Parse(ConstrainedDecoding.intentPlanSchema)
        Assert.Equal("object", parsed.RootElement.GetProperty("type").GetString())
        Assert.Equal(4, parsed.RootElement.GetProperty("required").GetArrayLength())

    [<Fact>]
    let ``beliefUpdateSchema is valid JSON`` () =
        let parsed = System.Text.Json.JsonDocument.Parse(ConstrainedDecoding.beliefUpdateSchema)
        Assert.Equal("object", parsed.RootElement.GetProperty("type").GetString())
        Assert.Equal(4, parsed.RootElement.GetProperty("required").GetArrayLength())

    [<Fact>]
    let ``repairProposalSchema is valid JSON`` () =
        let parsed = System.Text.Json.JsonDocument.Parse(ConstrainedDecoding.repairProposalSchema)
        Assert.Equal("object", parsed.RootElement.GetProperty("type").GetString())
        Assert.Equal(4, parsed.RootElement.GetProperty("required").GetArrayLength())

    [<Fact>]
    let ``intentPlanSchema has strategy enum with 4 values`` () =
        let parsed = System.Text.Json.JsonDocument.Parse(ConstrainedDecoding.intentPlanSchema)
        let strategy = parsed.RootElement.GetProperty("properties").GetProperty("strategy")
        Assert.Equal(4, strategy.GetProperty("enum").GetArrayLength())

    [<Fact>]
    let ``beliefUpdateSchema has operation enum with 5 values`` () =
        let parsed = System.Text.Json.JsonDocument.Parse(ConstrainedDecoding.beliefUpdateSchema)
        let op = parsed.RootElement.GetProperty("properties").GetProperty("operation")
        Assert.Equal(5, op.GetProperty("enum").GetArrayLength())

    [<Fact>]
    let ``repairProposalSchema has repair_action enum with 5 values`` () =
        let parsed = System.Text.Json.JsonDocument.Parse(ConstrainedDecoding.repairProposalSchema)
        let action = parsed.RootElement.GetProperty("properties").GetProperty("repair_action")
        Assert.Equal(5, action.GetProperty("enum").GetArrayLength())

    // =========================================================================
    // Integration
    // =========================================================================

    [<Fact>]
    let ``jsonConstrained with intentPlanSchema produces valid constrained request`` () =
        let msgs = [ { Role = Role.User; Content = "What should I do?" } ]
        let req = ConstrainedDecoding.jsonConstrained ConstrainedDecoding.intentPlanSchema msgs
        match req.ResponseFormat with
        | Some (ResponseFormat.Constrained (Grammar.JsonSchema s)) ->
            Assert.Contains("intent", s)
            Assert.Contains("strategy", s)
        | _ -> Assert.Fail("Expected JsonSchema constrained format")

    [<Fact>]
    let ``multiple constraints - last one wins`` () =
        let req = LlmRequest.Default
        let r1 = ConstrainedDecoding.withEbnfGrammar "g1" req
        let r2 = ConstrainedDecoding.withJsonSchema """{"type":"string"}""" r1
        match r2.ResponseFormat with
        | Some (ResponseFormat.Constrained (Grammar.JsonSchema _)) -> ()
        | _ -> Assert.Fail("Expected JsonSchema to override Ebnf")
