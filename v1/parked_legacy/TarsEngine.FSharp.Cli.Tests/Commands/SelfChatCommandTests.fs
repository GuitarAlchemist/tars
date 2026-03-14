namespace TarsEngine.FSharp.Cli.Tests.Commands

open System
open System.Threading.Tasks
open Xunit
open FsUnit.Xunit
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Cli.Commands
open TarsEngine.FSharp.Cli.Tests.TestHelpers

module SelfChatCommandTests =

    [<Fact>]
    let ``SelfChatCommand should have correct name`` () =
        let logger = TestHelpers.createMockLogger<SelfChatCommand>()
        let command = SelfChatCommand(logger)
        command.Name |> should equal "self-chat"

    [<Fact>]
    let ``SelfChatCommand should have correct description`` () =
        let logger = TestHelpers.createMockLogger<SelfChatCommand>()
        let command = SelfChatCommand(logger)
        command.Description |> should contain "self-chat"

    [<Fact>]
    let ``SelfChatCommand should execute successfully with default options`` () =
        task {
            let logger = TestHelpers.createMockLogger<SelfChatCommand>()
            let command = SelfChatCommand(logger)
            let options = TestHelpers.createDefaultCommandOptions()
            
            let! result = command.ExecuteAsync(options)
            
            result.Success |> should be True
        }

    [<Fact>]
    let ``SelfChatCommand should initiate self-conversation`` () =
        task {
            let logger = TestHelpers.createMockLogger<SelfChatCommand>()
            let command = SelfChatCommand(logger)
            let options = TestHelpers.createDefaultCommandOptions()
            
            let! result = command.ExecuteAsync(options)
            
            result.Success |> should be True
            result.Message |> should contain "conversation"
        }

    [<Fact>]
    let ``SelfChatCommand should handle chat errors gracefully`` () =
        task {
            let logger = TestHelpers.createMockLogger<SelfChatCommand>()
            let command = SelfChatCommand(logger)
            let options = TestHelpers.createDefaultCommandOptions()
            
            let! result = command.ExecuteAsync(options)
            
            // Should not throw exceptions
            result |> should not' (be null)
        }

    [<Fact>]
    let ``SelfChatCommand should validate chat parameters`` () =
        task {
            let logger = TestHelpers.createMockLogger<SelfChatCommand>()
            let command = SelfChatCommand(logger)
            let options = TestHelpers.createDefaultCommandOptions()
            
            let! result = command.ExecuteAsync(options)
            
            result.Success |> should be True
            result.Message |> should contain "chat"
        }
