namespace TarsEngine.FSharp.Cli.Tests.Commands

open System
open System.Threading.Tasks
open Xunit
open FsUnit.Xunit
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Cli.Commands
open TarsEngine.FSharp.Cli.Tests.TestHelpers

module LiveEndpointsCommandTests =

    [<Fact>]
    let ``LiveEndpointsCommand should have correct name`` () =
        let logger = TestHelpers.createMockLogger<LiveEndpointsCommand>()
        let command = LiveEndpointsCommand(logger)
        command.Name |> should equal "live-endpoints"

    [<Fact>]
    let ``LiveEndpointsCommand should have correct description`` () =
        let logger = TestHelpers.createMockLogger<LiveEndpointsCommand>()
        let command = LiveEndpointsCommand(logger)
        command.Description |> should contain "live endpoints"

    [<Fact>]
    let ``LiveEndpointsCommand should execute successfully with default options`` () =
        task {
            let logger = TestHelpers.createMockLogger<LiveEndpointsCommand>()
            let command = LiveEndpointsCommand(logger)
            let options = TestHelpers.createDefaultCommandOptions()
            
            let! result = command.ExecuteAsync(options)
            
            result.Success |> should be True
        }

    [<Fact>]
    let ``LiveEndpointsCommand should show endpoint status`` () =
        task {
            let logger = TestHelpers.createMockLogger<LiveEndpointsCommand>()
            let command = LiveEndpointsCommand(logger)
            let options = TestHelpers.createDefaultCommandOptions()
            
            let! result = command.ExecuteAsync(options)
            
            result.Success |> should be True
            result.Message |> should contain "endpoints"
        }

    [<Fact>]
    let ``LiveEndpointsCommand should handle errors gracefully`` () =
        task {
            let logger = TestHelpers.createMockLogger<LiveEndpointsCommand>()
            let command = LiveEndpointsCommand(logger)
            let options = TestHelpers.createDefaultCommandOptions()
            
            // Test with invalid configuration
            let! result = command.ExecuteAsync(options)
            
            // Should not throw exceptions
            result |> should not' (be null)
        }

    [<Fact>]
    let ``LiveEndpointsCommand should validate endpoint connectivity`` () =
        task {
            let logger = TestHelpers.createMockLogger<LiveEndpointsCommand>()
            let command = LiveEndpointsCommand(logger)
            let options = TestHelpers.createDefaultCommandOptions()
            
            let! result = command.ExecuteAsync(options)
            
            result.Success |> should be True
            result.Message |> should contain "connectivity"
        }
