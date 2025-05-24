module TarsEngine.FSharp.Core.Tests.Metascript.MetascriptExecutorTests

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
/// Tests for the MetascriptExecutor class.
/// </summary>
type MetascriptExecutorTests() =
    let executor = new MetascriptExecutor(MockLogger<MetascriptExecutor>() :> ILogger<MetascriptExecutor>)
    
    /// <summary>
    /// Test that MetascriptExecutor can execute an F# block.
    /// </summary>
    [<Fact>]
    member _.``MetascriptExecutor can execute an F# block``() =
        // Arrange
        let block = {
            Type = MetascriptBlockType.FSharp
            Content = """
let greeting = "Hello, World!"
printfn "%s" greeting
greeting
"""
            LineNumber = 1
            ColumnNumber = 0
            Parameters = []
            Id = Guid.NewGuid().ToString()
            ParentId = None
            Metadata = Map.empty
        }
        
        let! context = executor.CreateContextAsync()
        
        // Act
        let result = executor.ExecuteBlockAsync(block, context).Result
        
        // Assert
        Assert.NotNull(result)
        Assert.Equal(MetascriptExecutionStatus.Success, result.Status)
        Assert.Contains("Hello, World!", result.Output)
        Assert.Equal("Hello, World!", result.ReturnValue |> Option.defaultValue "")
    
    /// <summary>
    /// Test that MetascriptExecutor can execute a command block.
    /// </summary>
    [<Fact>]
    member _.``MetascriptExecutor can execute a command block``() =
        // Arrange
        let block = {
            Type = MetascriptBlockType.Command
            Content = """echo "Hello from the command line!""""
            LineNumber = 1
            ColumnNumber = 0
            Parameters = []
            Id = Guid.NewGuid().ToString()
            ParentId = None
            Metadata = Map.empty
        }
        
        let! context = executor.CreateContextAsync()
        
        // Act
        let result = executor.ExecuteBlockAsync(block, context).Result
        
        // Assert
        Assert.NotNull(result)
        Assert.Equal(MetascriptExecutionStatus.Success, result.Status)
        Assert.Contains("Hello from the command line!", result.Output)
    
    /// <summary>
    /// Test that MetascriptExecutor can execute a metascript.
    /// </summary>
    [<Fact>]
    member _.``MetascriptExecutor can execute a metascript``() =
        // Arrange
        let metascript = {
            Name = "Test"
            Blocks = [
                {
                    Type = MetascriptBlockType.FSharp
                    Content = """
let x = 2
let y = 3
let sum = x + y
printfn "%d + %d = %d" x y sum
sum
"""
                    LineNumber = 1
                    ColumnNumber = 0
                    Parameters = []
                    Id = Guid.NewGuid().ToString()
                    ParentId = None
                    Metadata = Map.empty
                }
                {
                    Type = MetascriptBlockType.FSharp
                    Content = """
let product = sum * 2
printfn "%d * 2 = %d" sum product
product
"""
                    LineNumber = 8
                    ColumnNumber = 0
                    Parameters = []
                    Id = Guid.NewGuid().ToString()
                    ParentId = None
                    Metadata = Map.empty
                }
            ]
            FilePath = None
            CreationTime = DateTime.UtcNow
            LastModificationTime = None
            Description = None
            Author = None
            Version = None
            Dependencies = []
            Imports = []
            Metadata = Map.empty
        }
        
        // Act
        let result = executor.ExecuteAsync(metascript).Result
        
        // Assert
        Assert.NotNull(result)
        Assert.Equal(MetascriptExecutionStatus.Success, result.Status)
        Assert.Contains("2 + 3 = 5", result.Output)
        Assert.Contains("5 * 2 = 10", result.Output)
        Assert.Equal(10, result.ReturnValue |> Option.defaultValue 0)
    
    /// <summary>
    /// Test that MetascriptExecutor can create a context.
    /// </summary>
    [<Fact>]
    member _.``MetascriptExecutor can create a context``() =
        // Act
        let context = executor.CreateContextAsync().Result
        
        // Assert
        Assert.NotNull(context)
        Assert.Equal(Directory.GetCurrentDirectory(), context.WorkingDirectory)
        Assert.Empty(context.Variables)
        Assert.None(context.Parent)
        Assert.None(context.CurrentMetascript)
        Assert.None(context.CurrentBlock)
    
    /// <summary>
    /// Test that MetascriptExecutor can create a context with variables.
    /// </summary>
    [<Fact>]
    member _.``MetascriptExecutor can create a context with variables``() =
        // Arrange
        let variables = Map.ofList [
            ("x", {
                Name = "x"
                Value = 42
                Type = typeof<int>
                IsReadOnly = false
                Metadata = Map.empty
            })
            ("greeting", {
                Name = "greeting"
                Value = "Hello, World!"
                Type = typeof<string>
                IsReadOnly = false
                Metadata = Map.empty
            })
        ]
        
        // Act
        let context = executor.CreateContextAsync(variables = variables).Result
        
        // Assert
        Assert.NotNull(context)
        Assert.Equal(2, context.Variables.Count)
        Assert.Equal(42, context.Variables.["x"].Value)
        Assert.Equal("Hello, World!", context.Variables.["greeting"].Value)
    
    /// <summary>
    /// Test that MetascriptExecutor can create a context with a parent.
    /// </summary>
    [<Fact>]
    member _.``MetascriptExecutor can create a context with a parent``() =
        // Arrange
        let parentVariables = Map.ofList [
            ("x", {
                Name = "x"
                Value = 42
                Type = typeof<int>
                IsReadOnly = false
                Metadata = Map.empty
            })
        ]
        
        let parentContext = executor.CreateContextAsync(variables = parentVariables).Result
        
        let childVariables = Map.ofList [
            ("y", {
                Name = "y"
                Value = 24
                Type = typeof<int>
                IsReadOnly = false
                Metadata = Map.empty
            })
        ]
        
        // Act
        let childContext = executor.CreateContextAsync(variables = childVariables, parent = parentContext).Result
        
        // Assert
        Assert.NotNull(childContext)
        Assert.Equal(2, childContext.Variables.Count)
        Assert.Equal(42, childContext.Variables.["x"].Value)
        Assert.Equal(24, childContext.Variables.["y"].Value)
        Assert.Equal(Some parentContext, childContext.Parent)
    
    /// <summary>
    /// Test that MetascriptExecutor supports the expected block types.
    /// </summary>
    [<Fact>]
    member _.``MetascriptExecutor supports the expected block types``() =
        // Act
        let supportedTypes = executor.GetSupportedBlockTypes()
        
        // Assert
        Assert.NotEmpty(supportedTypes)
        Assert.Contains(MetascriptBlockType.Text, supportedTypes)
        Assert.Contains(MetascriptBlockType.FSharp, supportedTypes)
        Assert.Contains(MetascriptBlockType.Command, supportedTypes)
        
        // Check specific block types
        Assert.True(executor.IsBlockTypeSupported(MetascriptBlockType.Text))
        Assert.True(executor.IsBlockTypeSupported(MetascriptBlockType.FSharp))
        Assert.True(executor.IsBlockTypeSupported(MetascriptBlockType.Command))
        Assert.False(executor.IsBlockTypeSupported(MetascriptBlockType.Unknown))
