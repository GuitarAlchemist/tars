module TarsEngine.FSharp.Core.Tests.Core.CollectionsTests

open System
open Xunit
open TarsEngine.FSharp.Core

/// <summary>
/// Tests for the Collections module.
/// </summary>
type CollectionsTests() =
    /// <summary>
    /// Test that Collections.tryFind returns Some when the element is found.
    /// </summary>
    [<Fact>]
    member _.``Collections.tryFind returns Some when element is found``() =
        // Arrange
        let list = [1; 2; 3; 4; 5]
        let predicate x = x = 3
        
        // Act
        let result = Collections.tryFind predicate list
        
        // Assert
        match result with
        | Some value -> Assert.Equal(3, value)
        | None -> Assert.True(false, "Expected Some but got None")
    
    /// <summary>
    /// Test that Collections.tryFind returns None when the element is not found.
    /// </summary>
    [<Fact>]
    member _.``Collections.tryFind returns None when element is not found``() =
        // Arrange
        let list = [1; 2; 3; 4; 5]
        let predicate x = x = 6
        
        // Act
        let result = Collections.tryFind predicate list
        
        // Assert
        match result with
        | Some _ -> Assert.True(false, "Expected None but got Some")
        | None -> Assert.True(true)
    
    /// <summary>
    /// Test that Collections.tryFindIndex returns Some when the element is found.
    /// </summary>
    [<Fact>]
    member _.``Collections.tryFindIndex returns Some when element is found``() =
        // Arrange
        let list = [1; 2; 3; 4; 5]
        let predicate x = x = 3
        
        // Act
        let result = Collections.tryFindIndex predicate list
        
        // Assert
        match result with
        | Some index -> Assert.Equal(2, index)
        | None -> Assert.True(false, "Expected Some but got None")
    
    /// <summary>
    /// Test that Collections.tryFindIndex returns None when the element is not found.
    /// </summary>
    [<Fact>]
    member _.``Collections.tryFindIndex returns None when element is not found``() =
        // Arrange
        let list = [1; 2; 3; 4; 5]
        let predicate x = x = 6
        
        // Act
        let result = Collections.tryFindIndex predicate list
        
        // Assert
        match result with
        | Some _ -> Assert.True(false, "Expected None but got Some")
        | None -> Assert.True(true)
