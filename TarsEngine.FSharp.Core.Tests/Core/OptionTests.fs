module TarsEngine.FSharp.Core.Tests.Core.OptionTests

open System
open Xunit
open TarsEngine.FSharp.Core

/// <summary>
/// Tests for the Option module.
/// </summary>
type OptionTests() =
    /// <summary>
    /// Test that Option.map applies the function to the value in the Some case.
    /// </summary>
    [<Fact>]
    member _.``Option.map applies function to Some value``() =
        // Arrange
        let option = Some 42
        let f x = x * 2
        
        // Act
        let mappedOption = Option.map f option
        
        // Assert
        match mappedOption with
        | Some value -> Assert.Equal(84, value)
        | None -> Assert.True(false, "Expected Some but got None")
    
    /// <summary>
    /// Test that Option.map does not apply the function in the None case.
    /// </summary>
    [<Fact>]
    member _.``Option.map does not apply function to None value``() =
        // Arrange
        let option = None
        let f x = x * 2
        
        // Act
        let mappedOption = Option.map f option
        
        // Assert
        match mappedOption with
        | Some _ -> Assert.True(false, "Expected None but got Some")
        | None -> Assert.True(true)
    
    /// <summary>
    /// Test that Option.bind applies the function to the value in the Some case.
    /// </summary>
    [<Fact>]
    member _.``Option.bind applies function to Some value``() =
        // Arrange
        let option = Some 42
        let f x = Some (x * 2)
        
        // Act
        let boundOption = Option.bind f option
        
        // Assert
        match boundOption with
        | Some value -> Assert.Equal(84, value)
        | None -> Assert.True(false, "Expected Some but got None")
    
    /// <summary>
    /// Test that Option.bind does not apply the function in the None case.
    /// </summary>
    [<Fact>]
    member _.``Option.bind does not apply function to None value``() =
        // Arrange
        let option = None
        let f x = Some (x * 2)
        
        // Act
        let boundOption = Option.bind f option
        
        // Assert
        match boundOption with
        | Some _ -> Assert.True(false, "Expected None but got Some")
        | None -> Assert.True(true)
