namespace TarsEngine.FSharp.Main.Tests.Metascripts

open System
open Xunit
open TarsEngine.FSharp.Main.Metascripts.Types

/// <summary>
/// Tests for the Metascripts.Types module.
/// </summary>
module TypesTests =
    [<Fact>]
    let ``Metascript should contain the name, code, and language`` () =
        let metascript = {
            Name = "test"
            Code = "code"
            Language = "language"
        }
        Assert.Equal("test", metascript.Name)
        Assert.Equal("code", metascript.Code)
        Assert.Equal("language", metascript.Language)
    
    [<Fact>]
    let ``MetascriptExecutionResult should contain the success, output, and errors`` () =
        let result = {
            Success = true
            Output = "output"
            Errors = ["error1"; "error2"]
        }
        Assert.True(result.Success)
        Assert.Equal("output", result.Output)
        Assert.Equal(2, result.Errors.Length)
        Assert.Equal("error1", result.Errors.[0])
        Assert.Equal("error2", result.Errors.[1])
