namespace TarsEngine.FSharp.Main.Tests.Monads

open System
open System.Threading.Tasks
open Xunit
open TarsEngine.FSharp.Main.Monads

/// <summary>
/// Tests for the AsyncResult module
/// </summary>
module AsyncResultTests =
    [<Fact>]
    let ``AsyncResult.success should create a successful async result`` () =
        let asyncResult = AsyncResult.success 42
        let result = Async.RunSynchronously asyncResult
        match result with
        | Ok value -> Assert.Equal(42, value)
        | Error _ -> Assert.True(false, "Expected Ok, got Error")
    
    [<Fact>]
    let ``AsyncResult.error should create a failed async result`` () =
        let asyncResult = AsyncResult.error "error"
        let result = Async.RunSynchronously asyncResult
        match result with
        | Ok _ -> Assert.True(false, "Expected Error, got Ok")
        | Error value -> Assert.Equal("error", value)
    
    [<Fact>]
    let ``AsyncResult.ofResult should convert a Result to an async Result`` () =
        let result = Ok 42
        let asyncResult = AsyncResult.ofResult result
        let unwrapped = Async.RunSynchronously asyncResult
        match unwrapped with
        | Ok value -> Assert.Equal(42, value)
        | Error _ -> Assert.True(false, "Expected Ok, got Error")
    
    [<Fact>]
    let ``AsyncResult.ofTask should convert a successful Task to an async Result`` () =
        let task = Task.FromResult(42)
        let asyncResult = AsyncResult.ofTask task
        let result = Async.RunSynchronously asyncResult
        match result with
        | Ok value -> Assert.Equal(42, value)
        | Error _ -> Assert.True(false, "Expected Ok, got Error")
    
    [<Fact>]
    let ``AsyncResult.ofTask should convert a failing Task to an async Result`` () =
        let task = Task.FromException<int>(InvalidOperationException("error"))
        let asyncResult = AsyncResult.ofTask task
        let result = Async.RunSynchronously asyncResult
        match result with
        | Ok _ -> Assert.True(false, "Expected Error, got Ok")
        | Error ex -> Assert.IsType<InvalidOperationException>(ex)
    
    [<Fact>]
    let ``AsyncResult.ofTaskWithError should convert a successful Task to an async Result`` () =
        let task = Task.FromResult(42)
        let asyncResult = AsyncResult.ofTaskWithError (fun ex -> ex.Message) task
        let result = Async.RunSynchronously asyncResult
        match result with
        | Ok value -> Assert.Equal(42, value)
        | Error _ -> Assert.True(false, "Expected Ok, got Error")
    
    [<Fact>]
    let ``AsyncResult.ofTaskWithError should convert a failing Task to an async Result with mapped error`` () =
        let task = Task.FromException<int>(InvalidOperationException("error"))
        let asyncResult = AsyncResult.ofTaskWithError (fun ex -> ex.Message) task
        let result = Async.RunSynchronously asyncResult
        match result with
        | Ok _ -> Assert.True(false, "Expected Error, got Ok")
        | Error message -> Assert.Equal("error", message)
    
    [<Fact>]
    let ``AsyncResult.map should transform the value for Ok`` () =
        let asyncResult = AsyncResult.success 21
        let mapped = AsyncResult.map (fun i -> i * 2) asyncResult
        let result = Async.RunSynchronously mapped
        match result with
        | Ok value -> Assert.Equal(42, value)
        | Error _ -> Assert.True(false, "Expected Ok, got Error")
    
    [<Fact>]
    let ``AsyncResult.map should not transform the error for Error`` () =
        let asyncResult = AsyncResult.error "error"
        let mapped = AsyncResult.map (fun i -> i * 2) asyncResult
        let result = Async.RunSynchronously mapped
        match result with
        | Ok _ -> Assert.True(false, "Expected Error, got Ok")
        | Error value -> Assert.Equal("error", value)
    
    [<Fact>]
    let ``AsyncResult.mapError should transform the error for Error`` () =
        let asyncResult = AsyncResult.error "error"
        let mapped = AsyncResult.mapError (fun s -> s.Length) asyncResult
        let result = Async.RunSynchronously mapped
        match result with
        | Ok _ -> Assert.True(false, "Expected Error, got Ok")
        | Error value -> Assert.Equal(5, value)
    
    [<Fact>]
    let ``AsyncResult.mapError should not transform the value for Ok`` () =
        let asyncResult = AsyncResult.success 42
        let mapped = AsyncResult.mapError (fun s -> s.Length) asyncResult
        let result = Async.RunSynchronously mapped
        match result with
        | Ok value -> Assert.Equal(42, value)
        | Error _ -> Assert.True(false, "Expected Ok, got Error")
    
    [<Fact>]
    let ``AsyncResult.bind should apply the function to the value for Ok`` () =
        let asyncResult = AsyncResult.success 21
        let bound = AsyncResult.bind (fun i -> Ok (i * 2)) asyncResult
        let result = Async.RunSynchronously bound
        match result with
        | Ok value -> Assert.Equal(42, value)
        | Error _ -> Assert.True(false, "Expected Ok, got Error")
    
    [<Fact>]
    let ``AsyncResult.bind should not apply the function to the error for Error`` () =
        let asyncResult = AsyncResult.error "error"
        let bound = AsyncResult.bind (fun i -> Ok (i * 2)) asyncResult
        let result = Async.RunSynchronously bound
        match result with
        | Ok _ -> Assert.True(false, "Expected Error, got Ok")
        | Error value -> Assert.Equal("error", value)
    
    [<Fact>]
    let ``AsyncResult.bindAsync should apply the function to the value for Ok`` () =
        let asyncResult = AsyncResult.success 21
        let bound = AsyncResult.bindAsync (fun i -> AsyncResult.success (i * 2)) asyncResult
        let result = Async.RunSynchronously bound
        match result with
        | Ok value -> Assert.Equal(42, value)
        | Error _ -> Assert.True(false, "Expected Ok, got Error")
    
    [<Fact>]
    let ``AsyncResult.bindAsync should not apply the function to the error for Error`` () =
        let asyncResult = AsyncResult.error "error"
        let bound = AsyncResult.bindAsync (fun i -> AsyncResult.success (i * 2)) asyncResult
        let result = Async.RunSynchronously bound
        match result with
        | Ok _ -> Assert.True(false, "Expected Error, got Ok")
        | Error value -> Assert.Equal("error", value)
    
    [<Fact>]
    let ``AsyncResult.match' should apply the ok function for Ok`` () =
        let asyncResult = AsyncResult.success 21
        let matched = AsyncResult.match' (fun i -> i * 2) (fun s -> s.Length) asyncResult
        let result = Async.RunSynchronously matched
        Assert.Equal(42, result)
    
    [<Fact>]
    let ``AsyncResult.match' should apply the error function for Error`` () =
        let asyncResult = AsyncResult.error "error"
        let matched = AsyncResult.match' (fun i -> i * 2) (fun s -> s.Length) asyncResult
        let result = Async.RunSynchronously matched
        Assert.Equal(5, result)
    
    [<Fact>]
    let ``AsyncResult.ifOk should perform the action for Ok`` () =
        let asyncResult = AsyncResult.success 42
        let mutable value = 0
        let returned = AsyncResult.ifOk (fun i -> value <- i) asyncResult
        let result = Async.RunSynchronously returned
        Assert.Equal(42, value)
        match result with
        | Ok v -> Assert.Equal(42, v)
        | Error _ -> Assert.True(false, "Expected Ok, got Error")
    
    [<Fact>]
    let ``AsyncResult.ifOk should not perform the action for Error`` () =
        let asyncResult = AsyncResult.error "error"
        let mutable value = 0
        let returned = AsyncResult.ifOk (fun i -> value <- i) asyncResult
        let result = Async.RunSynchronously returned
        Assert.Equal(0, value)
        match result with
        | Ok _ -> Assert.True(false, "Expected Error, got Ok")
        | Error v -> Assert.Equal("error", v)
    
    [<Fact>]
    let ``AsyncResult.ifError should perform the action for Error`` () =
        let asyncResult = AsyncResult.error "error"
        let mutable value = ""
        let returned = AsyncResult.ifError (fun s -> value <- s) asyncResult
        let result = Async.RunSynchronously returned
        Assert.Equal("error", value)
        match result with
        | Ok _ -> Assert.True(false, "Expected Error, got Ok")
        | Error v -> Assert.Equal("error", v)
    
    [<Fact>]
    let ``AsyncResult.ifError should not perform the action for Ok`` () =
        let asyncResult = AsyncResult.success 42
        let mutable value = ""
        let returned = AsyncResult.ifError (fun s -> value <- s) asyncResult
        let result = Async.RunSynchronously returned
        Assert.Equal("", value)
        match result with
        | Ok v -> Assert.Equal(42, v)
        | Error _ -> Assert.True(false, "Expected Ok, got Error")
    
    [<Fact>]
    let ``AsyncResult.tryAsync should return Ok for successful function`` () =
        let asyncResult = AsyncResult.tryAsync (fun () -> async { return 42 })
        let result = Async.RunSynchronously asyncResult
        match result with
        | Ok value -> Assert.Equal(42, value)
        | Error _ -> Assert.True(false, "Expected Ok, got Error")
    
    [<Fact>]
    let ``AsyncResult.tryAsync should return Error for failing function`` () =
        let asyncResult = AsyncResult.tryAsync (fun () -> async { return raise (InvalidOperationException("error")) })
        let result = Async.RunSynchronously asyncResult
        match result with
        | Ok _ -> Assert.True(false, "Expected Error, got Ok")
        | Error ex -> Assert.IsType<InvalidOperationException>(ex)
    
    [<Fact>]
    let ``AsyncResult.ofAsyncOption should return Ok for Some`` () =
        let asyncOption = async { return Some 42 }
        let asyncResult = AsyncResult.ofAsyncOption "error" asyncOption
        let result = Async.RunSynchronously asyncResult
        match result with
        | Ok value -> Assert.Equal(42, value)
        | Error _ -> Assert.True(false, "Expected Ok, got Error")
    
    [<Fact>]
    let ``AsyncResult.ofAsyncOption should return Error for None`` () =
        let asyncOption = async { return None }
        let asyncResult = AsyncResult.ofAsyncOption "error" asyncOption
        let result = Async.RunSynchronously asyncResult
        match result with
        | Ok _ -> Assert.True(false, "Expected Error, got Ok")
        | Error value -> Assert.Equal("error", value)
