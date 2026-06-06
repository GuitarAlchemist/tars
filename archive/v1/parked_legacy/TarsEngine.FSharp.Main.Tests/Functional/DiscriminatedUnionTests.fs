namespace TarsEngine.FSharp.Main.Tests.Functional

open System
open Xunit
open TarsEngine.FSharp.Main.Functional.DiscriminatedUnion

/// <summary>
/// Tests for the DiscriminatedUnion module
/// </summary>
module DiscriminatedUnionTests =
    [<Fact>]
    let ``unhandledCase should throw InvalidOperationException`` () =
        let ex = Assert.Throws<InvalidOperationException>(fun () -> unhandledCase<int> "test" |> ignore)
        Assert.Contains("Unhandled case", ex.Message)
        Assert.Contains("String", ex.Message)
