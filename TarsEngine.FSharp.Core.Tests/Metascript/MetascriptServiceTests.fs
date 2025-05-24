module TarsEngine.FSharp.Core.Tests.Metascript.MetascriptServiceTests

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open Xunit
open TarsEngine.FSharp.Core.Metascript
open TarsEngine.FSharp.Core.Metascript.Services

/// <summary>
/// Mock logger for testing.
/// </summary>
type MockLogger<'T>() =
    interface ILogger<'T> with
        member _.BeginScope<'TState>(state: 'TState) = { new IDisposable with member _.Dispose() = () }
        member _.IsEnabled(logLevel: LogLevel) = true
        member _.Log<'TState>(logLevel: LogLevel, eventId: EventId, state: 'TState, ex: exn, formatter: Func<'TState, exn, string>) = ()

/// <summary>
/// Tests for the MetascriptService class.
/// </summary>
type MetascriptServiceTests() =
    let executor = new MetascriptExecutor(MockLogger<MetascriptExecutor>() :> ILogger<MetascriptExecutor>)
    let service = new MetascriptService(MockLogger<MetascriptService>() :> ILogger<MetascriptService>, executor)
    
    /// <summary>
    /// Test that MetascriptService can parse a metascript.
    /// </summary>
    [<Fact>]
    member _.``MetascriptService can parse a metascript``() =
        // Arrange
        let text = """
# Test Metascript

```yaml
name: Test
description: A test metascript
author: TARS
version: 1.0.0
```

## F# Block

```fsharp
let greeting = "Hello, World!"
printfn "%s" greeting
```
"""
        
        // Act
        let metascript = service.ParseMetascriptAsync(text, "Test").Result
        
        // Assert
        Assert.NotNull(metascript)
        Assert.Equal("Test", metascript.Name)
        Assert.Equal(Some "A test metascript", metascript.Description)
        Assert.Equal(Some "TARS", metascript.Author)
        Assert.Equal(Some "1.0.0", metascript.Version)
        Assert.Equal(2, metascript.Blocks.Length)
    
    /// <summary>
    /// Test that MetascriptService can execute a metascript.
    /// </summary>
    [<Fact>]
    member _.``MetascriptService can execute a metascript``() =
        // Arrange
        let text = """
# Test Metascript

```fsharp
let greeting = "Hello, World!"
printfn "%s" greeting
greeting
```
"""
        
        // Act
        let result = service.ExecuteMetascriptTextAsync(text, "Test").Result
        
        // Assert
        Assert.NotNull(result)
        Assert.Equal(MetascriptExecutionStatus.Success, result.Status)
        Assert.Contains("Hello, World!", result.Output)
        Assert.Equal("Hello, World!", result.ReturnValue |> Option.defaultValue "")
    
    /// <summary>
    /// Test that MetascriptService can execute a metascript with multiple blocks.
    /// </summary>
    [<Fact>]
    member _.``MetascriptService can execute a metascript with multiple blocks``() =
        // Arrange
        let text = """
# Test Metascript

```fsharp
let x = 2
let y = 3
let sum = x + y
printfn "%d + %d = %d" x y sum
sum
```

```fsharp
let product = sum * 2
printfn "%d * 2 = %d" sum product
product
```
"""
        
        // Act
        let result = service.ExecuteMetascriptTextAsync(text, "Test").Result
        
        // Assert
        Assert.NotNull(result)
        Assert.Equal(MetascriptExecutionStatus.Success, result.Status)
        Assert.Contains("2 + 3 = 5", result.Output)
        Assert.Contains("5 * 2 = 10", result.Output)
        Assert.Equal(10, result.ReturnValue |> Option.defaultValue 0)
    
    /// <summary>
    /// Test that MetascriptService can execute a command block.
    /// </summary>
    [<Fact>]
    member _.``MetascriptService can execute a command block``() =
        // Arrange
        let text = """
# Test Metascript

```command
echo "Hello from the command line!"
```
"""
        
        // Act
        let result = service.ExecuteMetascriptTextAsync(text, "Test").Result
        
        // Assert
        Assert.NotNull(result)
        Assert.Equal(MetascriptExecutionStatus.Success, result.Status)
        Assert.Contains("Hello from the command line!", result.Output)
    
    /// <summary>
    /// Test that MetascriptService can validate a metascript.
    /// </summary>
    [<Fact>]
    member _.``MetascriptService can validate a metascript``() =
        // Arrange
        let text = """
# Test Metascript

```fsharp
let greeting = "Hello, World!"
printfn "%s" greeting
```
"""
        
        // Act
        let metascript = service.ParseMetascriptAsync(text, "Test").Result
        let (isValid, errors) = service.ValidateMetascriptAsync(metascript).Result
        
        // Assert
        Assert.True(isValid)
        Assert.Empty(errors)
    
    /// <summary>
    /// Test that MetascriptService can detect invalid blocks.
    /// </summary>
    [<Fact>]
    member _.``MetascriptService can detect invalid blocks``() =
        // Arrange
        let text = """
# Test Metascript

```invalid
This is an invalid block type.
```
"""
        
        // Act
        let metascript = service.ParseMetascriptAsync(text, "Test").Result
        
        // Assert
        Assert.NotNull(metascript)
        Assert.Equal(1, metascript.Blocks.Length)
        Assert.Equal(MetascriptBlockType.Unknown, metascript.Blocks.[0].Type)
    
    /// <summary>
    /// Test that MetascriptService can parse block parameters.
    /// </summary>
    [<Fact>]
    member _.``MetascriptService can parse block parameters``() =
        // Arrange
        let text = """
# Test Metascript

```fsharp name="test" description="A test block"
let greeting = "Hello, World!"
printfn "%s" greeting
```
"""
        
        // Act
        let metascript = service.ParseMetascriptAsync(text, "Test").Result
        
        // Assert
        Assert.NotNull(metascript)
        Assert.Equal(1, metascript.Blocks.Length)
        
        let block = metascript.Blocks.[0]
        Assert.Equal(2, block.Parameters.Length)
        
        let nameParam = block.Parameters |> List.find (fun p -> p.Name = "name")
        Assert.Equal("test", nameParam.Value)
        
        let descParam = block.Parameters |> List.find (fun p -> p.Name = "description")
        Assert.Equal("A test block", descParam.Value)
