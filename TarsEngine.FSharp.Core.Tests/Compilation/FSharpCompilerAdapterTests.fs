namespace TarsEngine.FSharp.Core.Tests.Compilation

open System
open System.IO
open System.Threading.Tasks
open Microsoft.CodeAnalysis.Scripting
open Microsoft.Extensions.Logging
open Xunit
open TarsEngine.FSharp.Core.Compilation

/// <summary>
/// Tests for the FSharpCompilerAdapter class.
/// </summary>
module FSharpCompilerAdapterTests =
    
    /// <summary>
    /// Mock logger for testing.
    /// </summary>
    type MockLogger<'T>() =
        interface ILogger<'T> with
            member _.Log<'TState>(logLevel, eventId, state, ex, formatter) =
                // Do nothing
                ()
            
            member _.IsEnabled(logLevel) = true
            
            member _.BeginScope<'TState>(state) =
                { new IDisposable with
                    member _.Dispose() = ()
                }
    
    /// <summary>
    /// Test that the adapter can compile simple F# code.
    /// </summary>
    [<Fact>]
    let ``CompileAsync with simple F# code should succeed``() = task {
        // Arrange
        let logger = MockLogger<FSharpCompilerAdapter>() :> ILogger<FSharpCompilerAdapter>
        let adapter = FSharpCompilerAdapter(logger)
        let code = """
            module Test

            let add x y = x + y
            """
        
        // Act
        let! result = adapter.CompileAsync(code)
        
        // Assert
        Assert.True(result.Success)
        Assert.Empty(result.Errors)
    }
    
    /// <summary>
    /// Test that the adapter can compile F# code with references.
    /// </summary>
    [<Fact>]
    let ``CompileAsync with references should succeed``() = task {
        // Arrange
        let logger = MockLogger<FSharpCompilerAdapter>() :> ILogger<FSharpCompilerAdapter>
        let adapter = FSharpCompilerAdapter(logger)
        let code = """
            module Test

            open System.Text.RegularExpressions

            let isValidEmail email =
                Regex.IsMatch(email, @"^[^@\s]+@[^@\s]+\.[^@\s]+$")
            """
        
        let references = [|
            typeof<System.Text.RegularExpressions.Regex>.Assembly.Location
        |]
        
        // Act
        let! result = adapter.CompileAsync(code, references)
        
        // Assert
        Assert.True(result.Success)
        Assert.Empty(result.Errors)
    }
    
    /// <summary>
    /// Test that the adapter can compile F# code with output path.
    /// </summary>
    [<Fact>]
    let ``CompileAsync with output path should succeed``() = task {
        // Arrange
        let logger = MockLogger<FSharpCompilerAdapter>() :> ILogger<FSharpCompilerAdapter>
        let adapter = FSharpCompilerAdapter(logger)
        let code = """
            module Test

            let multiply x y = x * y
            """
        
        let outputPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}.dll")
        
        // Act
        let! result = adapter.CompileAsync(code, [||], outputPath)
        
        // Assert
        Assert.True(result.Success)
        Assert.Empty(result.Errors)
        Assert.True(File.Exists(outputPath))
        
        // Cleanup
        if File.Exists(outputPath) then
            File.Delete(outputPath)
    }
    
    /// <summary>
    /// Test that the adapter can compile F# code with executable flag.
    /// </summary>
    [<Fact>]
    let ``CompileAsync with executable flag should succeed``() = task {
        // Arrange
        let logger = MockLogger<FSharpCompilerAdapter>() :> ILogger<FSharpCompilerAdapter>
        let adapter = FSharpCompilerAdapter(logger)
        let code = """
            module Program

            [<EntryPoint>]
            let main argv =
                printfn "Hello, world!"
                0
            """
        
        let outputPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}.exe")
        
        // Act
        let! result = adapter.CompileAsync(code, [||], outputPath, true)
        
        // Assert
        Assert.True(result.Success)
        Assert.Empty(result.Errors)
        Assert.True(File.Exists(outputPath))
        
        // Cleanup
        if File.Exists(outputPath) then
            File.Delete(outputPath)
    }
    
    /// <summary>
    /// Test that the adapter can compile F# code with defines.
    /// </summary>
    [<Fact>]
    let ``CompileAsync with defines should succeed``() = task {
        // Arrange
        let logger = MockLogger<FSharpCompilerAdapter>() :> ILogger<FSharpCompilerAdapter>
        let adapter = FSharpCompilerAdapter(logger)
        let code = """
            module Test

            #if DEBUG
            let isDebug = true
            #else
            let isDebug = false
            #endif
            """
        
        let outputPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}.dll")
        let defines = [| "DEBUG" |]
        
        // Act
        let! result = adapter.CompileAsync(code, [||], outputPath, false, defines)
        
        // Assert
        Assert.True(result.Success)
        Assert.Empty(result.Errors)
        Assert.True(File.Exists(outputPath))
        
        // Cleanup
        if File.Exists(outputPath) then
            File.Delete(outputPath)
    }
    
    /// <summary>
    /// Test that the adapter can compile F# code with source files.
    /// </summary>
    [<Fact>]
    let ``CompileAsync with source files should succeed``() = task {
        // Arrange
        let logger = MockLogger<FSharpCompilerAdapter>() :> ILogger<FSharpCompilerAdapter>
        let adapter = FSharpCompilerAdapter(logger)
        let code = """
            module Main

            let result = Helper.add 2 3
            """
        
        let helperCode = """
            module Helper

            let add x y = x + y
            """
        
        let helperFilePath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}.fs")
        File.WriteAllText(helperFilePath, helperCode)
        
        let outputPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}.dll")
        let sourceFiles = [| helperFilePath |]
        
        // Act
        let! result = adapter.CompileAsync(code, [||], outputPath, false, [||], sourceFiles)
        
        // Assert
        Assert.True(result.Success)
        Assert.Empty(result.Errors)
        Assert.True(File.Exists(outputPath))
        
        // Cleanup
        if File.Exists(outputPath) then
            File.Delete(outputPath)
        if File.Exists(helperFilePath) then
            File.Delete(helperFilePath)
    }
    
    /// <summary>
    /// Test that the adapter can compile F# code with resources.
    /// </summary>
    [<Fact>]
    let ``CompileAsync with resources should succeed``() = task {
        // Arrange
        let logger = MockLogger<FSharpCompilerAdapter>() :> ILogger<FSharpCompilerAdapter>
        let adapter = FSharpCompilerAdapter(logger)
        let code = """
            module Test

            let getResource() =
                let assembly = System.Reflection.Assembly.GetExecutingAssembly()
                use stream = assembly.GetManifestResourceStream("Test.Resource.txt")
                use reader = new System.IO.StreamReader(stream)
                reader.ReadToEnd()
            """
        
        let resourceFilePath = Path.Combine(Path.GetTempPath(), "Resource.txt")
        File.WriteAllText(resourceFilePath, "Hello, world!")
        
        let outputPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}.dll")
        let resources = [| $"{resourceFilePath},Test.Resource.txt" |]
        
        // Act
        let! result = adapter.CompileAsync(code, [||], outputPath, false, [||], [||], resources)
        
        // Assert
        Assert.True(result.Success)
        Assert.Empty(result.Errors)
        Assert.True(File.Exists(outputPath))
        
        // Cleanup
        if File.Exists(outputPath) then
            File.Delete(outputPath)
        if File.Exists(resourceFilePath) then
            File.Delete(resourceFilePath)
    }
    
    /// <summary>
    /// Test that the adapter can compile F# code with other flags.
    /// </summary>
    [<Fact>]
    let ``CompileAsync with other flags should succeed``() = task {
        // Arrange
        let logger = MockLogger<FSharpCompilerAdapter>() :> ILogger<FSharpCompilerAdapter>
        let adapter = FSharpCompilerAdapter(logger)
        let code = """
            module Test

            let divide x y = x / y
            """
        
        let outputPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}.dll")
        let otherFlags = [| "--optimize+" |]
        
        // Act
        let! result = adapter.CompileAsync(code, [||], outputPath, false, [||], [||], [||], otherFlags)
        
        // Assert
        Assert.True(result.Success)
        Assert.Empty(result.Errors)
        Assert.True(File.Exists(outputPath))
        
        // Cleanup
        if File.Exists(outputPath) then
            File.Delete(outputPath)
    }
    
    /// <summary>
    /// Test that the adapter handles compilation errors correctly.
    /// </summary>
    [<Fact>]
    let ``CompileAsync with invalid F# code should fail``() = task {
        // Arrange
        let logger = MockLogger<FSharpCompilerAdapter>() :> ILogger<FSharpCompilerAdapter>
        let adapter = FSharpCompilerAdapter(logger)
        let code = """
            module Test

            let x = 1 +
            """
        
        // Act
        let! result = adapter.CompileAsync(code)
        
        // Assert
        Assert.False(result.Success)
        Assert.NotEmpty(result.Errors)
    }
    
    /// <summary>
    /// Test that the adapter can compile F# code with script options.
    /// </summary>
    [<Fact>]
    let ``CompileAsync with script options should succeed``() = task {
        // Arrange
        let logger = MockLogger<FSharpCompilerAdapter>() :> ILogger<FSharpCompilerAdapter>
        let adapter = FSharpCompilerAdapter(logger)
        let code = """
            module Test

            open System.Text.RegularExpressions

            let isValidEmail email =
                Regex.IsMatch(email, @"^[^@\s]+@[^@\s]+\.[^@\s]+$")
            """
        
        let options =
            ScriptOptions.Default
                .AddReferences(typeof<System.Text.RegularExpressions.Regex>.Assembly)
        
        // Act
        let! result = adapter.CompileAsync(code, options)
        
        // Assert
        Assert.True(result.Success)
        Assert.Empty(result.Errors)
    }
