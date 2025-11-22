namespace TarsEngine.FSharp.Main.Tests.Functional

open System
open Xunit
open TarsEngine.FSharp.Main.Functional.Types

/// <summary>
/// Tests for the Functional.Types module.
/// </summary>
module TypesTests =
    [<Fact>]
    let ``compose should compose two functions`` () =
        let f x = x + 1
        let g x = x * 2
        let h = compose f g
        Assert.Equal(5, h 2)
    
    [<Fact>]
    let ``apply should apply a function to a value`` () =
        let f x = x + 1
        Assert.Equal(3, apply f 2)
    
    [<Fact>]
    let ``curry should curry a function`` () =
        let f (a, b) = a + b
        let g = curry f
        Assert.Equal(3, g 1 2)
    
    [<Fact>]
    let ``uncurry should uncurry a function`` () =
        let f a b = a + b
        let g = uncurry f
        Assert.Equal(3, g (1, 2))
