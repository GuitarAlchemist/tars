namespace TarsEngine.FSharp.Main.Tests.SelfImprovement

open System
open Xunit
open TarsEngine.FSharp.Main.SelfImprovement.Types

/// <summary>
/// Tests for the SelfImprovement.Types module.
/// </summary>
module TypesTests =
    [<Fact>]
    let ``SelfImprovement should contain the name, description, and capabilities`` () =
        let selfImprovement = {
            Name = "test"
            Description = "description"
            Capabilities = ["capability1"; "capability2"]
        }
        Assert.Equal("test", selfImprovement.Name)
        Assert.Equal("description", selfImprovement.Description)
        Assert.Equal(2, selfImprovement.Capabilities.Length)
        Assert.Equal("capability1", selfImprovement.Capabilities.[0])
        Assert.Equal("capability2", selfImprovement.Capabilities.[1])
    
    [<Fact>]
    let ``SelfImprovementExecutionResult should contain the success, output, and errors`` () =
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
