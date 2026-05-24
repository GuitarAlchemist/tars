namespace TarsEngine.FSharp.Main.Tests.Utilities

open System
open Xunit
open TarsEngine.FSharp.Main.Utilities.Types

/// <summary>
/// Tests for the Utilities.Types module.
/// </summary>
module TypesTests =
    [<Fact>]
    let ``Utility should contain the name, description, and capabilities`` () =
        let utility = {
            Name = "test"
            Description = "description"
            Capabilities = ["capability1"; "capability2"]
        }
        Assert.Equal("test", utility.Name)
        Assert.Equal("description", utility.Description)
        Assert.Equal(2, utility.Capabilities.Length)
        Assert.Equal("capability1", utility.Capabilities.[0])
        Assert.Equal("capability2", utility.Capabilities.[1])
    
    [<Fact>]
    let ``UtilityExecutionResult should contain the success, output, and errors`` () =
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
