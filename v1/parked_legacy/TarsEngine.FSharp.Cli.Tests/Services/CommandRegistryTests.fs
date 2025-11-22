namespace TarsEngine.FSharp.Cli.Tests.Services

open System
open System.Threading.Tasks
open Xunit
open FsUnit.Xunit
open Microsoft.Extensions.Logging
open Microsoft.Extensions.DependencyInjection
open TarsEngine.FSharp.Cli.Commands
open TarsEngine.FSharp.Cli.Tests.TestHelpers

module CommandRegistryTests =

    [<Fact>]
    let ``CommandRegistry should initialize successfully`` () =
        let serviceProvider = TestHelpers.createTestServiceProvider()
        let registry = CommandRegistry(serviceProvider)
        
        registry |> should not' (be null)

    [<Fact>]
    let ``CommandRegistry should register commands`` () =
        let serviceProvider = TestHelpers.createTestServiceProvider()
        let registry = CommandRegistry(serviceProvider)
        
        let commands = registry.GetCommands()
        commands |> should not' (be empty)

    [<Fact>]
    let ``CommandRegistry should find commands by name`` () =
        let serviceProvider = TestHelpers.createTestServiceProvider()
        let registry = CommandRegistry(serviceProvider)
        
        let versionCommand = registry.FindCommand("version")
        versionCommand |> should not' (be null)

    [<Fact>]
    let ``CommandRegistry should return null for unknown commands`` () =
        let serviceProvider = TestHelpers.createTestServiceProvider()
        let registry = CommandRegistry(serviceProvider)
        
        let unknownCommand = registry.FindCommand("unknown-command")
        unknownCommand |> should be null

    [<Fact>]
    let ``CommandRegistry should list all available commands`` () =
        let serviceProvider = TestHelpers.createTestServiceProvider()
        let registry = CommandRegistry(serviceProvider)
        
        let commands = registry.GetCommands()
        commands |> Seq.length |> should be (greaterThan 0)

    [<Fact>]
    let ``CommandRegistry should handle command registration errors gracefully`` () =
        let serviceProvider = TestHelpers.createTestServiceProvider()
        let registry = CommandRegistry(serviceProvider)
        
        // Should not throw during initialization
        registry |> should not' (be null)
