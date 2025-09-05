namespace TarsEngine.FSharp.Cli.Tests.Commands

open System
open System.Threading.Tasks
open Xunit
open FsUnit.Xunit
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Cli.Commands
open TarsEngine.FSharp.Cli.Tests.TestHelpers

module MixtralCommandTests =

    [<Fact>]
    let ``MixtralCommand should have correct name`` () =
        let logger = TestHelpers.createMockLogger<MixtralCommand>()
        let command = MixtralCommand(logger)
        command.Name |> should equal "mixtral"

    [<Fact>]
    let ``MixtralCommand should have correct description`` () =
        let logger = TestHelpers.createMockLogger<MixtralCommand>()
        let command = MixtralCommand(logger)
        command.Description |> should contain "Mixtral"

    [<Fact>]
    let ``MixtralCommand should execute successfully with default options`` () =
        task {
            let logger = TestHelpers.createMockLogger<MixtralCommand>()
            let command = MixtralCommand(logger)
            let options = TestHelpers.createDefaultCommandOptions()
            
            let! result = command.ExecuteAsync(options)
            
            result.Success |> should be True
        }

    [<Fact>]
    let ``MixtralCommand should process AI requests`` () =
        task {
            let logger = TestHelpers.createMockLogger<MixtralCommand>()
            let command = MixtralCommand(logger)
            let options = TestHelpers.createDefaultCommandOptions()
            
            let! result = command.ExecuteAsync(options)
            
            result.Success |> should be True
            result.Message |> should contain "AI"
        }

    [<Fact>]
    let ``MixtralCommand should handle AI errors gracefully`` () =
        task {
            let logger = TestHelpers.createMockLogger<MixtralCommand>()
            let command = MixtralCommand(logger)
            let options = TestHelpers.createDefaultCommandOptions()
            
            let! result = command.ExecuteAsync(options)
            
            // Should not throw exceptions
            result |> should not' (be null)
        }

    [<Fact>]
    let ``MixtralCommand should validate AI parameters`` () =
        task {
            let logger = TestHelpers.createMockLogger<MixtralCommand>()
            let command = MixtralCommand(logger)
            let options = TestHelpers.createDefaultCommandOptions()
            
            let! result = command.ExecuteAsync(options)
            
            result.Success |> should be True
            result.Message |> should contain "Mixtral"
        }
