namespace TarsEngine.FSharp.Main.Tests.Monads

open System
open Xunit
open TarsEngine.FSharp.Main.Monads

/// <summary>
/// Tests for the Result module
/// </summary>
module ResultTests =
    [<Fact>]
    let ``Result.success should create an Ok result`` () =
        let result = Result.success 42
        match result with
        | Ok value -> Assert.Equal(42, value)
        | Error _ -> Assert.True(false, "Expected Ok, got Error")
    
    [<Fact>]
    let ``Result.error should create an Error result`` () =
        let result = Result.error "error"
        match result with
        | Ok _ -> Assert.True(false, "Expected Error, got Ok")
        | Error value -> Assert.Equal("error", value)
    
    [<Fact>]
    let ``Result.isOk should return true for Ok`` () =
        let result = Ok 42
        Assert.True(Result.isOk result)
    
    [<Fact>]
    let ``Result.isOk should return false for Error`` () =
        let result = Error "error"
        Assert.False(Result.isOk result)
    
    [<Fact>]
    let ``Result.isError should return true for Error`` () =
        let result = Error "error"
        Assert.True(Result.isError result)
    
    [<Fact>]
    let ``Result.isError should return false for Ok`` () =
        let result = Ok 42
        Assert.False(Result.isError result)
    
    [<Fact>]
    let ``Result.getValue should return the value for Ok`` () =
        let result = Ok 42
        Assert.Equal(42, Result.getValue result)
    
    [<Fact>]
    let ``Result.getValue should throw for Error`` () =
        let result = Error "error"
        Assert.Throws<InvalidOperationException>(fun () -> Result.getValue result |> ignore)
    
    [<Fact>]
    let ``Result.getError should return the error for Error`` () =
        let result = Error "error"
        Assert.Equal("error", Result.getError result)
    
    [<Fact>]
    let ``Result.getError should throw for Ok`` () =
        let result = Ok 42
        Assert.Throws<InvalidOperationException>(fun () -> Result.getError result |> ignore)
    
    [<Fact>]
    let ``Result.valueOrDefault should return the value for Ok`` () =
        let result = Ok 42
        Assert.Equal(42, Result.valueOrDefault 0 result)
    
    [<Fact>]
    let ``Result.valueOrDefault should return the default value for Error`` () =
        let result = Error "error"
        Assert.Equal(0, Result.valueOrDefault 0 result)
    
    [<Fact>]
    let ``Result.valueOr should return the value for Ok`` () =
        let result = Ok 42
        Assert.Equal(42, Result.valueOr (fun _ -> 0) result)
    
    [<Fact>]
    let ``Result.valueOr should return the result of the function for Error`` () =
        let result = Error "error"
        Assert.Equal(5, Result.valueOr (fun s -> s.Length) result)
    
    [<Fact>]
    let ``Result.map should transform the value for Ok`` () =
        let result = Ok 21
        let mapped = Result.map (fun i -> i * 2) result
        match mapped with
        | Ok value -> Assert.Equal(42, value)
        | Error _ -> Assert.True(false, "Expected Ok, got Error")
    
    [<Fact>]
    let ``Result.map should not transform the error for Error`` () =
        let result = Error "error"
        let mapped = Result.map (fun i -> i * 2) result
        match mapped with
        | Ok _ -> Assert.True(false, "Expected Error, got Ok")
        | Error value -> Assert.Equal("error", value)
    
    [<Fact>]
    let ``Result.mapError should transform the error for Error`` () =
        let result = Error "error"
        let mapped = Result.mapError (fun s -> s.Length) result
        match mapped with
        | Ok _ -> Assert.True(false, "Expected Error, got Ok")
        | Error value -> Assert.Equal(5, value)
    
    [<Fact>]
    let ``Result.mapError should not transform the value for Ok`` () =
        let result = Ok 42
        let mapped = Result.mapError (fun s -> s.Length) result
        match mapped with
        | Ok value -> Assert.Equal(42, value)
        | Error _ -> Assert.True(false, "Expected Ok, got Error")
    
    [<Fact>]
    let ``Result.bind should apply the function to the value for Ok`` () =
        let result = Ok 21
        let bound = Result.bind (fun i -> Ok (i * 2)) result
        match bound with
        | Ok value -> Assert.Equal(42, value)
        | Error _ -> Assert.True(false, "Expected Ok, got Error")
    
    [<Fact>]
    let ``Result.bind should not apply the function to the error for Error`` () =
        let result = Error "error"
        let bound = Result.bind (fun i -> Ok (i * 2)) result
        match bound with
        | Ok _ -> Assert.True(false, "Expected Error, got Ok")
        | Error value -> Assert.Equal("error", value)
    
    [<Fact>]
    let ``Result.match' should apply the ok function for Ok`` () =
        let result = Ok 21
        let matched = Result.match' (fun i -> i * 2) (fun s -> s.Length) result
        Assert.Equal(42, matched)
    
    [<Fact>]
    let ``Result.match' should apply the error function for Error`` () =
        let result = Error "error"
        let matched = Result.match' (fun i -> i * 2) (fun s -> s.Length) result
        Assert.Equal(5, matched)
    
    [<Fact>]
    let ``Result.ifOk should perform the action for Ok`` () =
        let result = Ok 42
        let mutable value = 0
        let returned = Result.ifOk (fun i -> value <- i) result
        Assert.Equal(42, value)
        Assert.Equal(result, returned)
    
    [<Fact>]
    let ``Result.ifOk should not perform the action for Error`` () =
        let result = Error "error"
        let mutable value = 0
        let returned = Result.ifOk (fun i -> value <- i) result
        Assert.Equal(0, value)
        Assert.Equal(result, returned)
    
    [<Fact>]
    let ``Result.ifError should perform the action for Error`` () =
        let result = Error "error"
        let mutable value = ""
        let returned = Result.ifError (fun s -> value <- s) result
        Assert.Equal("error", value)
        Assert.Equal(result, returned)
    
    [<Fact>]
    let ``Result.ifError should not perform the action for Ok`` () =
        let result = Ok 42
        let mutable value = ""
        let returned = Result.ifError (fun s -> value <- s) result
        Assert.Equal("", value)
        Assert.Equal(result, returned)
    
    [<Fact>]
    let ``Result.tryFunc should return Ok for successful function`` () =
        let result = Result.tryFunc (fun () -> 42)
        match result with
        | Ok value -> Assert.Equal(42, value)
        | Error _ -> Assert.True(false, "Expected Ok, got Error")
    
    [<Fact>]
    let ``Result.tryFunc should return Error for failing function`` () =
        let result = Result.tryFunc (fun () -> raise (InvalidOperationException("error")))
        match result with
        | Ok _ -> Assert.True(false, "Expected Error, got Ok")
        | Error ex -> Assert.IsType<InvalidOperationException>(ex)
    
    [<Fact>]
    let ``Result.ofOption should return Ok for Some`` () =
        let option = Some 42
        let result = Result.ofOption "error" option
        match result with
        | Ok value -> Assert.Equal(42, value)
        | Error _ -> Assert.True(false, "Expected Ok, got Error")
    
    [<Fact>]
    let ``Result.ofOption should return Error for None`` () =
        let option = None
        let result = Result.ofOption "error" option
        match result with
        | Ok _ -> Assert.True(false, "Expected Error, got Ok")
        | Error value -> Assert.Equal("error", value)
    
    [<Fact>]
    let ``Result.toOption should return Some for Ok`` () =
        let result = Ok 42
        let option = Result.toOption result
        match option with
        | Some value -> Assert.Equal(42, value)
        | None -> Assert.True(false, "Expected Some, got None")
    
    [<Fact>]
    let ``Result.toOption should return None for Error`` () =
        let result = Error "error"
        let option = Result.toOption result
        match option with
        | Some _ -> Assert.True(false, "Expected None, got Some")
        | None -> Assert.True(true)
