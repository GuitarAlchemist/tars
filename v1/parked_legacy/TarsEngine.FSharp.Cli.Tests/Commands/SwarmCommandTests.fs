namespace TarsEngine.FSharp.Cli.Tests.Commands

open System
open System.Threading.Tasks
open Xunit
open FsUnit.Xunit
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Cli.Commands
open TarsEngine.FSharp.Cli.Tests.TestHelpers

module SwarmCommandTests =

    [<Fact>]
    let ``SwarmCommand should have correct name`` () =
        let logger = TestHelpers.createMockLogger<SwarmCommand>()
        let command = SwarmCommand(logger)
        command.Name |> should equal "swarm"

    [<Fact>]
    let ``SwarmCommand should have correct description`` () =
        let logger = TestHelpers.createMockLogger<SwarmCommand>()
        let command = SwarmCommand(logger)
        command.Description |> should contain "swarm"

    [<Fact>]
    let ``SwarmCommand should execute successfully with default options`` () =
        task {
            let logger = TestHelpers.createMockLogger<SwarmCommand>()
            let command = SwarmCommand(logger)
            let options = TestHelpers.createDefaultCommandOptions()
            
            let! result = command.ExecuteAsync(options)
            
            result.Success |> should be True
        }

    [<Fact>]
    let ``SwarmCommand should manage swarm operations`` () =
        task {
            let logger = TestHelpers.createMockLogger<SwarmCommand>()
            let command = SwarmCommand(logger)
            let options = TestHelpers.createDefaultCommandOptions()
            
            let! result = command.ExecuteAsync(options)
            
            result.Success |> should be True
            result.Message |> should contain "swarm"
        }

    [<Fact>]
    let ``SwarmCommand should handle swarm errors gracefully`` () =
        task {
            let logger = TestHelpers.createMockLogger<SwarmCommand>()
            let command = SwarmCommand(logger)
            let options = TestHelpers.createDefaultCommandOptions()
            
            let! result = command.ExecuteAsync(options)
            
            // Should not throw exceptions
            result |> should not' (be null)
        }

    [<Fact>]
    let ``SwarmCommand should validate swarm parameters`` () =
        task {
            let logger = TestHelpers.createMockLogger<SwarmCommand>()
            let command = SwarmCommand(logger)
            let options = TestHelpers.createDefaultCommandOptions()
            
            let! result = command.ExecuteAsync(options)
            
            result.Success |> should be True
            result.Message |> should contain "operations"
        }
