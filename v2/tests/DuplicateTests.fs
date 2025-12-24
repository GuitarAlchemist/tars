module ArrayUtils

open NUnit.Framework
open FsUnit

[<TestFixture>]
 type DuplicateTests()
     [<Test>]
     member x.``findDuplicates should work correctly`` () =
         // Arrange
         let input = [|3; 4; 3; 2; 2; 1|] // Example input with duplicates
         
         // Act
         let result = findDuplicates input
         
         // Assert
         result |> should equal [3; 2]
     
     [<Test>]
     member x.``findDuplicates handles edge cases`` () =
         // Test edge cases like empty input, null, etc.
         let emptyInput = [||]
         let result = findDuplicates emptyInput
         result |> should equal []
         
         // Add more edge cases as needed
