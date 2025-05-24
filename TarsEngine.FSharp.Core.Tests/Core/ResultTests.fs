module TarsEngine.FSharp.Core.Tests.Core.ResultTests

open System
open Xunit
open TarsEngine.FSharp.Core.Core

/// <summary>
/// Tests for the Result module.
/// </summary>
type ResultTests() =
    /// <summary>
    /// Test that Result.map applies the function to the value in the Ok case.
    /// </summary>
    [<Fact>]
    member _.``Result.map applies function to Ok value``() =
        // Arrange
        let result = Ok 42
        let f x = x * 2
        
        // Act
        let mappedResult = Result.map f result
        
        // Assert
        match mappedResult with
        | Ok value -> Assert.Equal(84, value)
        | Error _ -> Assert.True(false, "Expected Ok but got Error")
    
    /// <summary>
    /// Test that Result.map does not apply the function in the Error case.
    /// </summary>
    [<Fact>]
    member _.``Result.map does not apply function to Error value``() =
        // Arrange
        let result = Error "error"
        let f x = x * 2
        
        // Act
        let mappedResult = Result.map f result
        
        // Assert
        match mappedResult with
        | Ok _ -> Assert.True(false, "Expected Error but got Ok")
        | Error e -> Assert.Equal("error", e)
    
    /// <summary>
    /// Test that Result.bind applies the function to the value in the Ok case.
    /// </summary>
    [<Fact>]
    member _.``Result.bind applies function to Ok value``() =
        // Arrange
        let result = Ok 42
        let f x = Ok (x * 2)
        
        // Act
        let boundResult = Result.bind f result
        
        // Assert
        match boundResult with
        | Ok value -> Assert.Equal(84, value)
        | Error _ -> Assert.True(false, "Expected Ok but got Error")
    
    /// <summary>
    /// Test that Result.bind does not apply the function in the Error case.
    /// </summary>
    [<Fact>]
    member _.``Result.bind does not apply function to Error value``() =
        // Arrange
        let result = Error "error"
        let f x = Ok (x * 2)
        
        // Act
        let boundResult = Result.bind f result
        
        // Assert
        match boundResult with
        | Ok _ -> Assert.True(false, "Expected Error but got Ok")
        | Error e -> Assert.Equal("error", e)
