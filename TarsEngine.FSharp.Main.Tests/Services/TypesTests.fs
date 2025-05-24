namespace TarsEngine.FSharp.Main.Tests.Services

open System
open Xunit
open TarsEngine.FSharp.Main.Services.Types

/// <summary>
/// Tests for the Services.Types module.
/// </summary>
module TypesTests =
    [<Fact>]
    let ``Service should contain the name, description, and capabilities`` () =
        let service = {
            Name = "test"
            Description = "description"
            Capabilities = ["capability1"; "capability2"]
        }
        Assert.Equal("test", service.Name)
        Assert.Equal("description", service.Description)
        Assert.Equal(2, service.Capabilities.Length)
        Assert.Equal("capability1", service.Capabilities.[0])
        Assert.Equal("capability2", service.Capabilities.[1])
    
    [<Fact>]
    let ``ServiceExecutionResult should contain the success, output, and errors`` () =
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
