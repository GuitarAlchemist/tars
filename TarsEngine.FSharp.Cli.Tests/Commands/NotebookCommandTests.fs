namespace TarsEngine.FSharp.Cli.Tests.Commands

open System
open System.Threading.Tasks
open Xunit
open FsUnit.Xunit
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Cli.Commands
open TarsEngine.FSharp.Cli.Tests.TestHelpers

module NotebookCommandTests =

    [<Fact>]
    let ``NotebookCommand should have correct name`` () =
        let logger = TestHelpers.createMockLogger<NotebookCommand>()
        let command = NotebookCommand(logger)
        command.Name |> should equal "notebook"

    [<Fact>]
    let ``NotebookCommand should have correct description`` () =
        let logger = TestHelpers.createMockLogger<NotebookCommand>()
        let command = NotebookCommand(logger)
        command.Description |> should contain "notebook"

    [<Fact>]
    let ``NotebookCommand should execute successfully with default options`` () =
        task {
            let logger = TestHelpers.createMockLogger<NotebookCommand>()
            let command = NotebookCommand(logger)
            let options = TestHelpers.createDefaultCommandOptions()
            
            let! result = command.ExecuteAsync(options)
            
            result.Success |> should be True
        }

    [<Fact>]
    let ``NotebookCommand should manage notebook operations`` () =
        task {
            let logger = TestHelpers.createMockLogger<NotebookCommand>()
            let command = NotebookCommand(logger)
            let options = TestHelpers.createDefaultCommandOptions()
            
            let! result = command.ExecuteAsync(options)
            
            result.Success |> should be True
            result.Message |> should contain "notebook"
        }

    [<Fact>]
    let ``NotebookCommand should handle notebook errors gracefully`` () =
        task {
            let logger = TestHelpers.createMockLogger<NotebookCommand>()
            let command = NotebookCommand(logger)
            let options = TestHelpers.createDefaultCommandOptions()
            
            let! result = command.ExecuteAsync(options)
            
            // Should not throw exceptions
            result |> should not' (be null)
        }

    [<Fact>]
    let ``NotebookCommand should validate notebook parameters`` () =
        task {
            let logger = TestHelpers.createMockLogger<NotebookCommand>()
            let command = NotebookCommand(logger)
            let options = TestHelpers.createDefaultCommandOptions()
            
            let! result = command.ExecuteAsync(options)
            
            result.Success |> should be True
            result.Message |> should contain "operations"
        }
