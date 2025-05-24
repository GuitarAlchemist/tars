module TarsEngine.FSharp.Core.Tests.Core.AsyncResultTests

open System
open System.Threading.Tasks
open Xunit
open TarsEngine.FSharp.Core

/// <summary>
/// Tests for the AsyncResult module.
/// </summary>
type AsyncResultTests() =
    /// <summary>
    /// Test that AsyncResult.map applies the function to the value in the Ok case.
    /// </summary>
    [<Fact>]
    member _.``AsyncResult.map applies function to Ok value``() =
        // Arrange
        let asyncResult = async { return Ok 42 }
        let f x = x * 2
        
        // Act
        let mappedAsyncResult = AsyncResult.map f asyncResult
        let result = mappedAsyncResult |> Async.RunSynchronously
        
        // Assert
        match result with
        | Ok value -> Assert.Equal(84, value)
        | Error _ -> Assert.True(false, "Expected Ok but got Error")
    
    /// <summary>
    /// Test that AsyncResult.map does not apply the function in the Error case.
    /// </summary>
    [<Fact>]
    member _.``AsyncResult.map does not apply function to Error value``() =
        // Arrange
        let asyncResult = async { return Error "error" }
        let f x = x * 2
        
        // Act
        let mappedAsyncResult = AsyncResult.map f asyncResult
        let result = mappedAsyncResult |> Async.RunSynchronously
        
        // Assert
        match result with
        | Ok _ -> Assert.True(false, "Expected Error but got Ok")
        | Error e -> Assert.Equal("error", e)
    
    /// <summary>
    /// Test that AsyncResult.bind applies the function to the value in the Ok case.
    /// </summary>
    [<Fact>]
    member _.``AsyncResult.bind applies function to Ok value``() =
        // Arrange
        let asyncResult = async { return Ok 42 }
        let f x = async { return Ok (x * 2) }
        
        // Act
        let boundAsyncResult = AsyncResult.bind f asyncResult
        let result = boundAsyncResult |> Async.RunSynchronously
        
        // Assert
        match result with
        | Ok value -> Assert.Equal(84, value)
        | Error _ -> Assert.True(false, "Expected Ok but got Error")
    
    /// <summary>
    /// Test that AsyncResult.bind does not apply the function in the Error case.
    /// </summary>
    [<Fact>]
    member _.``AsyncResult.bind does not apply function to Error value``() =
        // Arrange
        let asyncResult = async { return Error "error" }
        let f x = async { return Ok (x * 2) }
        
        // Act
        let boundAsyncResult = AsyncResult.bind f asyncResult
        let result = boundAsyncResult |> Async.RunSynchronously
        
        // Assert
        match result with
        | Ok _ -> Assert.True(false, "Expected Error but got Ok")
        | Error e -> Assert.Equal("error", e)
