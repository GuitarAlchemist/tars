namespace TarsEngine.FSharp.Main.Tests.Monads

open System
open Xunit
open TarsEngine.FSharp.Main.Monads.Types

/// <summary>
/// Tests for the Monads.Types module.
/// </summary>
module TypesTests =
    [<Fact>]
    let ``Identity should contain the value`` () =
        let identity = Identity 42
        match identity with
        | Identity value -> Assert.Equal(42, value)
    
    [<Fact>]
    let ``Reader should contain the function`` () =
        let reader = Reader (fun env -> env + 1)
        match reader with
        | Reader f -> Assert.Equal(3, f 2)
    
    [<Fact>]
    let ``Writer should contain the value and the log`` () =
        let writer = Writer (42, "log")
        match writer with
        | Writer (value, log) ->
            Assert.Equal(42, value)
            Assert.Equal("log", log)
    
    [<Fact>]
    let ``State should contain the function`` () =
        let state = State (fun s -> (s + 1, s * 2))
        match state with
        | State f ->
            let (value, newState) = f 2
            Assert.Equal(3, value)
            Assert.Equal(4, newState)
