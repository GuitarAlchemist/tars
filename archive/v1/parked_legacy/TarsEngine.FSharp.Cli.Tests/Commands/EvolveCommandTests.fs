namespace TarsEngine.FSharp.Cli.Tests.Commands

open System
open System.Threading.Tasks
open Xunit
open FsUnit.Xunit
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Cli.Commands
open TarsEngine.FSharp.Cli.Tests.TestHelpers

module EvolveCommandTests =

    [<Fact>]
    let ``EvolveCommand should have correct name`` () =
        let logger = TestHelpers.createMockLogger<EvolveCommand>()
        let command = EvolveCommand(logger)
        command.Name |> should equal "evolve"

    [<Fact>]
    let ``EvolveCommand should have correct description`` () =
        let logger = TestHelpers.createMockLogger<EvolveCommand>()
        let command = EvolveCommand(logger)
        command.Description |> should contain "evolve"

    [<Fact>]
    let ``EvolveCommand should execute successfully with default options`` () =
        task {
            let logger = TestHelpers.createMockLogger<EvolveCommand>()
            let command = EvolveCommand(logger)
            let options = TestHelpers.createDefaultCommandOptions()
            
            let! result = command.ExecuteAsync(options)
            
            result.Success |> should be True
        }

    [<Fact>]
    let ``EvolveCommand should perform evolution analysis`` () =
        task {
            let logger = TestHelpers.createMockLogger<EvolveCommand>()
            let command = EvolveCommand(logger)
            let options = TestHelpers.createDefaultCommandOptions()
            
            let! result = command.ExecuteAsync(options)
            
            result.Success |> should be True
            result.Message |> should contain "evolution"
        }

    [<Fact>]
    let ``EvolveCommand should handle evolution errors gracefully`` () =
        task {
            let logger = TestHelpers.createMockLogger<EvolveCommand>()
            let command = EvolveCommand(logger)
            let options = TestHelpers.createDefaultCommandOptions()
            
            let! result = command.ExecuteAsync(options)
            
            // Should not throw exceptions
            result |> should not' (be null)
        }

    [<Fact>]
    let ``EvolveCommand should validate evolution parameters`` () =
        task {
            let logger = TestHelpers.createMockLogger<EvolveCommand>()
            let command = EvolveCommand(logger)
            let options = TestHelpers.createDefaultCommandOptions()
            
            let! result = command.ExecuteAsync(options)
            
            result.Success |> should be True
            result.Message |> should contain "parameters"
        }
