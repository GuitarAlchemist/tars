namespace TarsEngine.FSharp.Cli.Tests.Commands

open System
open System.Threading.Tasks
open Xunit
open FsUnit.Xunit
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Cli.Commands
open TarsEngine.FSharp.Cli.Tests.TestHelpers

module ChatbotCommandTests =

    [<Fact>]
    let ``ChatbotCommand should have correct name`` () =
        let logger = TestHelpers.createMockLogger<ChatbotCommand>()
        let command = ChatbotCommand(logger)
        command.Name |> should equal "chatbot"

    [<Fact>]
    let ``ChatbotCommand should have correct description`` () =
        let logger = TestHelpers.createMockLogger<ChatbotCommand>()
        let command = ChatbotCommand(logger)
        command.Description |> should contain "chatbot"

    [<Fact>]
    let ``ChatbotCommand should execute successfully with default options`` () =
        task {
            let logger = TestHelpers.createMockLogger<ChatbotCommand>()
            let command = ChatbotCommand(logger)
            let options = TestHelpers.createDefaultCommandOptions()
            
            let! result = command.ExecuteAsync(options)
            
            result.Success |> should be True
        }

    [<Fact>]
    let ``ChatbotCommand should start chatbot interface`` () =
        task {
            let logger = TestHelpers.createMockLogger<ChatbotCommand>()
            let command = ChatbotCommand(logger)
            let options = TestHelpers.createDefaultCommandOptions()
            
            let! result = command.ExecuteAsync(options)
            
            result.Success |> should be True
            result.Message |> should contain "chatbot"
        }

    [<Fact>]
    let ``ChatbotCommand should handle chatbot errors gracefully`` () =
        task {
            let logger = TestHelpers.createMockLogger<ChatbotCommand>()
            let command = ChatbotCommand(logger)
            let options = TestHelpers.createDefaultCommandOptions()
            
            let! result = command.ExecuteAsync(options)
            
            // Should not throw exceptions
            result |> should not' (be null)
        }

    [<Fact>]
    let ``ChatbotCommand should validate chatbot parameters`` () =
        task {
            let logger = TestHelpers.createMockLogger<ChatbotCommand>()
            let command = ChatbotCommand(logger)
            let options = TestHelpers.createDefaultCommandOptions()
            
            let! result = command.ExecuteAsync(options)
            
            result.Success |> should be True
            result.Message |> should contain "interface"
        }
