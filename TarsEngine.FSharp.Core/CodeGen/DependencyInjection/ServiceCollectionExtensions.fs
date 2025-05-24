namespace TarsEngine.FSharp.Core.CodeGen.DependencyInjection

open System
open Microsoft.Extensions.DependencyInjection
open TarsEngine.FSharp.Core.CodeGen
open TarsEngine.FSharp.Core.CodeGen.Documentation
open TarsEngine.FSharp.Core.CodeGen.Testing
open TarsEngine.FSharp.Core.CodeGen.Testing.Assertions
open TarsEngine.FSharp.Core.CodeGen.Testing.Coverage
open TarsEngine.FSharp.Core.CodeGen.Testing.DependencyInjection
open TarsEngine.FSharp.Core.CodeGen.Testing.Generators
open TarsEngine.FSharp.Core.CodeGen.Testing.Interfaces
open TarsEngine.FSharp.Core.CodeGen.Testing.Runners
open TarsEngine.FSharp.Core.CodeGen.Workflow

/// <summary>
/// Extension methods for IServiceCollection to register TarsEngine.FSharp.Core.CodeGen services.
/// </summary>
module ServiceCollectionExtensions =
    /// <summary>
    /// Adds TarsEngine.FSharp.Core.CodeGen services to the service collection.
    /// </summary>
    /// <param name="services">The service collection.</param>
    /// <returns>The service collection.</returns>
    let addTarsEngineFSharpCodeGen (services: IServiceCollection) =
        // Add code generation services
        services.AddSingleton<ICodeGenerator, CodeGenerator>() |> ignore
        services.AddSingleton<IRefactorer, Refactorer>() |> ignore
        
        // Add testing services
        services.AddTarsEngineFSharpCodeGenTesting() |> ignore
        
        // Add documentation services
        services.AddSingleton<IDocumentationGenerator, MarkdownDocumentationGenerator>() |> ignore
        
        // Add workflow services
        services.AddSingleton<IWorkflowCoordinator, WorkflowCoordinator>() |> ignore
        
        // Add test runners
        services.AddSingleton<ITestRunner, XUnitTestRunner>() |> ignore
        services.AddSingleton<ITestRunner, NUnitTestRunner>() |> ignore
        services.AddSingleton<ITestRunner, MSTestRunner>() |> ignore
        
        // Add test runner factory
        services.AddSingleton<TestRunnerFactory>() |> ignore
        
        // Add test coverage analyzers
        services.AddSingleton<ITestCoverageAnalyzer, CoverletCoverageAnalyzer>() |> ignore
        
        // Return the service collection
        services
    
    /// <summary>
    /// Extension method for IServiceCollection to add TarsEngine.FSharp.Core.CodeGen services.
    /// </summary>
    type IServiceCollection with
        /// <summary>
        /// Adds TarsEngine.FSharp.Core.CodeGen services to the service collection.
        /// </summary>
        /// <returns>The service collection.</returns>
        member this.AddTarsEngineFSharpCodeGen() =
            addTarsEngineFSharpCodeGen this
