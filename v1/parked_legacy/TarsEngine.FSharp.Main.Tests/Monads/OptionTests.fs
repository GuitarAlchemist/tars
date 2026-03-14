namespace TarsEngine.FSharp.Main.Tests.Monads

open System
open Xunit
open TarsEngine.FSharp.Main.Monads

/// <summary>
/// Tests for the Option module
/// </summary>
module OptionTests =
    [<Fact>]
    let ``Option.valueOrDefault should return the value for Some`` () =
        let option = Some 42
        Assert.Equal(42, Option.valueOrDefault 0 option)
    
    [<Fact>]
    let ``Option.valueOrDefault should return the default value for None`` () =
        let option = None
        Assert.Equal(0, Option.valueOrDefault 0 option)
    
    [<Fact>]
    let ``Option.valueOr should return the value for Some`` () =
        let option = Some 42
        Assert.Equal(42, Option.valueOr (fun () -> 0) option)
    
    [<Fact>]
    let ``Option.valueOr should return the result of the function for None`` () =
        let option = None
        Assert.Equal(0, Option.valueOr (fun () -> 0) option)
    
    [<Fact>]
    let ``Option.map should transform the value for Some`` () =
        let option = Some 21
        let result = Option.map (fun i -> i * 2) option
        match result with
        | Some value -> Assert.Equal(42, value)
        | None -> Assert.True(false, "Expected Some, got None")
    
    [<Fact>]
    let ``Option.map should return None for None`` () =
        let option = None
        let result = Option.map (fun i -> i * 2) option
        match result with
        | Some _ -> Assert.True(false, "Expected None, got Some")
        | None -> Assert.True(true)
    
    [<Fact>]
    let ``Option.bind should apply the function to the value for Some`` () =
        let option = Some 21
        let result = Option.bind (fun i -> Some (i * 2)) option
        match result with
        | Some value -> Assert.Equal(42, value)
        | None -> Assert.True(false, "Expected Some, got None")
    
    [<Fact>]
    let ``Option.bind should return None for None`` () =
        let option = None
        let result = Option.bind (fun i -> Some (i * 2)) option
        match result with
        | Some _ -> Assert.True(false, "Expected None, got Some")
        | None -> Assert.True(true)
    
    [<Fact>]
    let ``Option.match' should apply the some function for Some`` () =
        let option = Some 21
        let result = Option.match' (fun i -> i * 2) (fun () -> 0) option
        Assert.Equal(42, result)
    
    [<Fact>]
    let ``Option.match' should apply the none function for None`` () =
        let option = None
        let result = Option.match' (fun i -> i * 2) (fun () -> 0) option
        Assert.Equal(0, result)
    
    [<Fact>]
    let ``Option.ifSome should perform the action for Some`` () =
        let option = Some 42
        let mutable value = 0
        let result = Option.ifSome (fun i -> value <- i) option
        Assert.Equal(42, value)
        Assert.Equal(option, result)
    
    [<Fact>]
    let ``Option.ifSome should not perform the action for None`` () =
        let option = None
        let mutable value = 0
        let result = Option.ifSome (fun i -> value <- i) option
        Assert.Equal(0, value)
        Assert.Equal(option, result)
    
    [<Fact>]
    let ``Option.ifNone should perform the action for None`` () =
        let option = None
        let mutable value = 0
        let result = Option.ifNone (fun () -> value <- 42) option
        Assert.Equal(42, value)
        Assert.Equal(option, result)
    
    [<Fact>]
    let ``Option.ifNone should not perform the action for Some`` () =
        let option = Some 42
        let mutable value = 0
        let result = Option.ifNone (fun () -> value <- 42) option
        Assert.Equal(0, value)
        Assert.Equal(option, result)
    
    [<Fact>]
    let ``Option.ofObj should return Some for non-null value`` () =
        let value = "test"
        let result = Option.ofObj value
        match result with
        | Some v -> Assert.Equal("test", v)
        | None -> Assert.True(false, "Expected Some, got None")
    
    [<Fact>]
    let ``Option.ofObj should return None for null value`` () =
        let value = null
        let result = Option.ofObj value
        match result with
        | Some _ -> Assert.True(false, "Expected None, got Some")
        | None -> Assert.True(true)
    
    [<Fact>]
    let ``Option.ofNullable should return Some for value with HasValue true`` () =
        let value = Nullable<int>(42)
        let result = Option.ofNullable value
        match result with
        | Some v -> Assert.Equal(42, v)
        | None -> Assert.True(false, "Expected Some, got None")
    
    [<Fact>]
    let ``Option.ofNullable should return None for value with HasValue false`` () =
        let value = Nullable<int>()
        let result = Option.ofNullable value
        match result with
        | Some _ -> Assert.True(false, "Expected None, got Some")
        | None -> Assert.True(true)
