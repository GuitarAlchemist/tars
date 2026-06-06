module Tars.Tests.ValidationTests

open Xunit
open Tars.Core.GrammarValidation

// ============================================================================
// GrammarValidation Tests
// ============================================================================

[<Fact>]
let ``validate: Simple capture works`` () =
    let grammar = "<answer>{text}</answer>"
    let output = "<answer>Hello World</answer>"

    match validate grammar output with
    | Valid captures ->
        Assert.Equal("Hello World", captures.["text"])
    | Invalid err ->
        Assert.Fail($"Expected valid, got: {err.Error}")

[<Fact>]
let ``validate: Multiple captures work`` () =
    let grammar = "<name>{name}</name><age>{age}</age>"
    let output = "<name>Alice</name><age>30</age>"

    match validate grammar output with
    | Valid captures ->
        Assert.Equal("Alice", captures.["name"])
        Assert.Equal("30", captures.["age"])
    | Invalid err ->
        Assert.Fail($"Expected valid, got: {err.Error}")

[<Fact>]
let ``validate: Greedy capture works`` () =
    let grammar = "<code>{content*}</code>"
    let output = "<code>line1\nline2\nline3</code>"

    match validate grammar output with
    | Valid captures ->
        Assert.Contains("line1", captures.["content"])
        Assert.Contains("line2", captures.["content"])
    | Invalid err ->
        Assert.Fail($"Expected valid, got: {err.Error}")

[<Fact>]
let ``validate: Invalid output returns error`` () =
    let grammar = "<answer>{text}</answer>"
    let output = "Just some plain text"

    match validate grammar output with
    | Valid _ ->
        Assert.Fail("Expected invalid")
    | Invalid err ->
        Assert.Contains("does not match", err.Error)

[<Fact>]
let ``matches: Returns true for matching output`` () =
    let grammar = "{\"result\": {value}}"
    let output = "{\"result\": 42}"
    Assert.True(matches grammar output)

[<Fact>]
let ``matches: Returns false for non-matching output`` () =
    let grammar = "{\"result\": {value}}"
    let output = "not json at all"
    Assert.False(matches grammar output)

[<Fact>]
let ``validateAny: Returns first matching grammar`` () =
    let grammars = [
        "<xml>{content}</xml>"
        "{\"json\": {value}}"
        "plain: {text}"
    ]
    let output = "{\"json\": 123}"

    match validateAny grammars output with
    | Valid captures ->
        Assert.Equal("123", captures.["value"])
    | Invalid _ ->
        Assert.Fail("Expected valid")

[<Fact>]
let ``extract: Gets specific capture`` () =
    let grammar = "<a>{first}</a><b>{second}</b>"
    let output = "<a>one</a><b>two</b>"
    let result = validate grammar output

    Assert.Equal(Some "one", extract "first" result)
    Assert.Equal(Some "two", extract "second" result)
    Assert.Equal(None, extract "third" result)

[<Fact>]
let ``buildHint: Creates prompt hint`` () =
    let grammar = "<answer>{text}</answer>"
    let hint = buildHint grammar

    Assert.Contains("exact format", hint)
    Assert.Contains(grammar, hint)


