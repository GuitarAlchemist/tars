namespace Tars.Engine.Grammar.Tests

open System
open Xunit
open Tars.Engine.Grammar
open Tars.Engine.Grammar.GrammarResolver

/// Tests that stress complex grammar generation scenarios
module ComplexGrammarTests =

    [<Fact>]
    let ``Generated grammar from HTTP request examples keeps structure`` () =
        // Arrange
        let examples =
            [ "GET /alpha HTTP/1.1"
              "POST /beta HTTP/2"
              "DELETE /gamma?id=1 HTTP/1.1" ]

        // Act
        let grammar = generateGrammarFromExamples "HttpRequests" examples
        let content = grammar.Content

        // Assert
        Assert.Contains("start =", content)
        Assert.Contains("httprequests_example_01", content)
        Assert.Contains("\"HTTP\"", content)
        Assert.Contains("\"DELETE\"", content)
        Assert.Contains("\"?\"", content)
        Assert.Contains("\"1\"", content)

    [<Fact>]
    let ``Generated grammar escapes quotes and braces`` () =
        // Arrange
        let examples =
            [ """{"op":"sum","values":[1,2,3]}"""
              """{"op":"average","values":[10,20]}""" ]

        // Act
        let grammar = generateGrammarFromExamples "JsonOps" examples

        // Assert
        let content = grammar.Content
        Assert.Contains("jsonops_example_01", content)
        Assert.Contains("\"{\"", content)
        Assert.Contains("\"}\"", content)
        Assert.Contains("\"values\"", content)
        Assert.Contains("\"\\\"\"", content)

    [<Fact>]
    let ``Grammar metadata is populated for complex examples`` () =
        // Arrange
        let examples =
            [ "fn add(x: i32, y: i32) -> i32 { return x + y; }"
              "fn mul(x: i32, y: i32) -> i32 { return x * y; }" ]

        // Act
        let grammar = generateGrammarFromExamples "RustFns" examples

        // Assert
        Assert.Equal("RustFns", grammar.Metadata.Name)
        Assert.Equal(Some "manual", grammar.Metadata.Source)
        Assert.True(grammar.Metadata.Hash.IsSome)
        Assert.True(grammar.Metadata.LastModified.IsSome)
        let content = grammar.Content
        Assert.Contains("\"fn\"", content)
        Assert.Contains("\"{\"", content)
        Assert.Contains("\"*\"", content)
