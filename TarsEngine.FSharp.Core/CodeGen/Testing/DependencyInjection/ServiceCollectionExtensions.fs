namespace TarsEngine.FSharp.Core.CodeGen.Testing.DependencyInjection

open System
open Microsoft.Extensions.DependencyInjection
open TarsEngine.FSharp.Core.CodeGen.Testing
open TarsEngine.FSharp.Core.CodeGen.Testing.Assertions
open TarsEngine.FSharp.Core.CodeGen.Testing.Generators
open TarsEngine.FSharp.Core.CodeGen.Testing.Interfaces

/// <summary>
/// Extension methods for IServiceCollection to register TarsEngine.FSharp.Core.CodeGen.Testing services.
/// </summary>
module ServiceCollectionExtensions =
    /// <summary>
    /// Adds TarsEngine.FSharp.Core.CodeGen.Testing services to the service collection.
    /// </summary>
    /// <param name="services">The service collection.</param>
    /// <returns>The service collection.</returns>
    let addTarsEngineFSharpCodeGenTesting (services: IServiceCollection) =
        // Register assertion formatters
        services.AddSingleton<IAssertionFormatter, XUnitAssertionFormatter>() |> ignore
        
        // Register test value generator
        services.AddSingleton<TestValueGenerator>() |> ignore
        
        // Register assertion generators
        services.AddSingleton<PrimitiveAssertionGenerator>() |> ignore
        
        // Register test code analyzer
        services.AddSingleton<TestCodeAnalyzer>() |> ignore
        
        // Register test template manager
        services.AddSingleton<TestTemplateManager>() |> ignore
        
        // Register test case generator
        services.AddSingleton<ITestCaseGenerator, BasicTestCaseGenerator>() |> ignore
        
        // Register test generator
        services.AddSingleton<ITestGenerator, TestGenerator>() |> ignore
        
        // Return the service collection
        services
    
    /// <summary>
    /// Extension method for IServiceCollection to add TarsEngine.FSharp.Core.CodeGen.Testing services.
    /// </summary>
    type IServiceCollection with
        /// <summary>
        /// Adds TarsEngine.FSharp.Core.CodeGen.Testing services to the service collection.
        /// </summary>
        /// <returns>The service collection.</returns>
        member this.AddTarsEngineFSharpCodeGenTesting() =
            addTarsEngineFSharpCodeGenTesting this
