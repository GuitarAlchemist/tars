namespace Tars.Tests

open System
open System.IO
open System.Threading.Tasks
open System.Text
open Xunit

type FSharpToolsTests() =

    [<Fact>]
    member _.``explainError returns correct explanation for FS0001``() =
        task {
            let! result = Tars.Tools.Standard.FSharpTools.explainError "FS0001"
            match result with
            | Ok explanation ->
                Assert.Contains("Type Mismatch", explanation)
                Assert.Contains("Fixes:", explanation)
            | Error e -> Assert.Fail(e)
        }

    [<Fact>]
    member _.``suggestFix returns specific fix for string-int mismatch``() =
        task {
            let args = """{ "error": "FS0001: This expression was expected to have type 'int' but here has type 'string'", "code": "let x = \"123\"" }"""
            let! result = Tars.Tools.Standard.FSharpTools.suggestFix args
            match result with
            | Ok suggestion ->
                Assert.Contains("Convert string to int", suggestion)
            | Error e -> Assert.Fail(e)
        }

    [<Fact>]
    member _.``checkSyntax detects unbalanced parentheses``() =
        task {
             let args = """{ "code": "let x = (1 + 2" }"""
             let! result = Tars.Tools.Standard.FSharpTools.checkSyntax args
             match result with
             | Ok output ->
                 Assert.Contains("Unbalanced parentheses", output)
             | Error e -> Assert.Fail(e)
        }

    [<Fact>]
    member _.``analyzeStructure extracts F# structure``() =
        task {
            let tempFile = Path.GetTempFileName() + ".fs"
            try
                let content = 
                    """module TestModule

type MyType = { X: int }

let myFunction x = x + 1
"""
                File.WriteAllText(tempFile, content, Encoding.UTF8)

                let args = sprintf """{ "path": "%s" }""" (tempFile.Replace("\\", "\\\\"))
                let! result = Tars.Tools.Standard.FSharpTools.analyzeStructure args

                match result with
                | Ok output ->
                    try
                        Assert.Contains("module TestModule", output)
                        Assert.Contains("type MyType", output)
                        Assert.Contains("let myFunction", output)
                    with ex ->
                        Assert.Fail(sprintf "Assertion failed: %s\nOutput was:\n%s" ex.Message output)
                | Error e -> 
                    Assert.Fail(sprintf "Analysis tool returned Error: %s" e)
            finally
                if File.Exists(tempFile) then File.Delete(tempFile)
        }
