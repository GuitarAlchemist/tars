namespace TarsEngine.FSharp.Core.Tests

open Xunit

module BasicTests =
    
    [<Fact>]
    let ``Basic test should pass``() =
        // Arrange
        let expected = 42
        let actual = 21 * 2
        
        // Assert
        Assert.Equal(expected, actual)
    
    [<Fact>]
    let ``String test should pass``() =
        // Arrange
        let expected = "Hello, TARS!"
        let actual = "Hello, " + "TARS!"
        
        // Assert
        Assert.Equal(expected, actual)
    
    [<Fact>]
    let ``List test should pass``() =
        // Arrange
        let numbers = [1; 2; 3; 4; 5]
        let expected = 15
        let actual = List.sum numbers
        
        // Assert
        Assert.Equal(expected, actual)
    
    [<Fact>]
    let ``Option test should pass``() =
        // Arrange
        let value = Some 42
        let expected = true
        let actual = Option.isSome value
        
        // Assert
        Assert.Equal(expected, actual)
