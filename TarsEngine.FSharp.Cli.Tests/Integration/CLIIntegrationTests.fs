namespace TarsEngine.FSharp.Cli.Tests.Integration

open System
open System.Threading.Tasks
open Xunit
open FsUnit.Xunit
open TarsEngine.FSharp.Cli.Tests.TestHelpers

module CLIIntegrationTests =

    [<Fact>]
    let ``CLI should execute version command successfully`` () =
        task {
            let! output = TestHelpers.assertCliCommandSucceeds ["version"]
            output |> should contain "TARS CLI"
        }

    [<Fact>]
    let ``CLI should execute help command successfully`` () =
        task {
            let! output = TestHelpers.assertCliCommandSucceeds ["help"]
            output |> should contain "Commands"
        }

    [<Fact>]
    let ``CLI should execute diagnose command successfully`` () =
        task {
            let! output = TestHelpers.assertCliCommandSucceeds ["diagnose"]
            output |> should contain "Diagnostics"
        }

    [<Fact>]
    let ``CLI should handle invalid commands gracefully`` () =
        task {
            let! stderr = TestHelpers.assertCliCommandFails ["invalid-command"]
            stderr |> should contain "Unknown command"
        }

    [<Fact>]
    let ``CLI should execute swarm status command successfully`` () =
        task {
            let! output = TestHelpers.assertCliCommandSucceeds ["swarm"; "status"]
            output |> should contain "swarm"
        }

    [<Fact>]
    let ``CLI should execute tars-llm status command successfully`` () =
        task {
            let! output = TestHelpers.assertCliCommandSucceeds ["tars-llm"; "status"]
            output |> should contain "LLM"
        }

    [<Fact>]
    let ``CLI should execute notebook create command successfully`` () =
        task {
            let! output = TestHelpers.assertCliCommandSucceeds ["notebook"; "create"; "test-notebook"]
            output |> should contain "notebook"
        }

    [<Fact>]
    let ``CLI should handle command execution errors gracefully`` () =
        task {
            // Test with potentially problematic command
            let! (exitCode, stdout, stderr) = TestHelpers.runCliCommand ["diagnose"; "--invalid-flag"]
            
            // Should not crash the CLI
            exitCode |> should not' (equal -1)
        }
