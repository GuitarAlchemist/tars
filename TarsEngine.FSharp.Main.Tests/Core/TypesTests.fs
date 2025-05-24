namespace TarsEngine.FSharp.Main.Tests.Core

open System
open Xunit
open TarsEngine.FSharp.Main.Core.Types

/// <summary>
/// Tests for the Core.Types module.
/// </summary>
module TypesTests =
    [<Fact>]
    let ``Result.Ok should contain the value`` () =
        let result = Ok 42
        match result with
        | Ok value -> Assert.Equal(42, value)
        | Error _ -> Assert.True(false, "Expected Ok, got Error")
    
    [<Fact>]
    let ``Result.Error should contain the error`` () =
        let result = Error "error"
        match result with
        | Ok _ -> Assert.True(false, "Expected Error, got Ok")
        | Error error -> Assert.Equal("error", error)
    
    [<Fact>]
    let ``Option.Some should contain the value`` () =
        let option = Some 42
        match option with
        | Some value -> Assert.Equal(42, value)
        | None -> Assert.True(false, "Expected Some, got None")
    
    [<Fact>]
    let ``Option.None should not contain a value`` () =
        let option = None
        match option with
        | Some _ -> Assert.True(false, "Expected None, got Some")
        | None -> Assert.True(true)
