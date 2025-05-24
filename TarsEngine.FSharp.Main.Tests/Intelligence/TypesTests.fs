namespace TarsEngine.FSharp.Main.Tests.Intelligence

open System
open Xunit
open TarsEngine.FSharp.Main.Intelligence.Types

/// <summary>
/// Tests for the Intelligence.Types module.
/// </summary>
module TypesTests =
    [<Fact>]
    let ``Intelligence should contain the name, description, and capabilities`` () =
        let intelligence = {
            Name = "test"
            Description = "description"
            Capabilities = ["capability1"; "capability2"]
        }
        Assert.Equal("test", intelligence.Name)
        Assert.Equal("description", intelligence.Description)
        Assert.Equal(2, intelligence.Capabilities.Length)
        Assert.Equal("capability1", intelligence.Capabilities.[0])
        Assert.Equal("capability2", intelligence.Capabilities.[1])
    
    [<Fact>]
    let ``IntelligenceExecutionResult should contain the success, output, and errors`` () =
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
