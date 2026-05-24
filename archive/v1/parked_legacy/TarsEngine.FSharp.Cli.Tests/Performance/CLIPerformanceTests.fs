namespace TarsEngine.FSharp.Cli.Tests.Performance

open System
open System.Diagnostics
open System.Threading.Tasks
open Xunit
open FsUnit.Xunit
open TarsEngine.FSharp.Cli.Tests.TestHelpers

module CLIPerformanceTests =

    [<Fact>]
    let ``CLI version command should execute within reasonable time`` () =
        task {
            let stopwatch = Stopwatch.StartNew()
            let! output = TestHelpers.assertCliCommandSucceeds ["version"]
            stopwatch.Stop()
            
            output |> should contain "TARS CLI"
            stopwatch.ElapsedMilliseconds |> should be (lessThan 5000L) // 5 seconds max
        }

    [<Fact>]
    let ``CLI help command should execute within reasonable time`` () =
        task {
            let stopwatch = Stopwatch.StartNew()
            let! output = TestHelpers.assertCliCommandSucceeds ["help"]
            stopwatch.Stop()
            
            output |> should contain "Commands"
            stopwatch.ElapsedMilliseconds |> should be (lessThan 5000L) // 5 seconds max
        }

    [<Fact>]
    let ``CLI diagnose command should execute within reasonable time`` () =
        task {
            let stopwatch = Stopwatch.StartNew()
            let! output = TestHelpers.assertCliCommandSucceeds ["diagnose"]
            stopwatch.Stop()
            
            output |> should contain "Diagnostics"
            stopwatch.ElapsedMilliseconds |> should be (lessThan 30000L) // 30 seconds max for diagnostics
        }

    [<Fact>]
    let ``CLI should handle multiple concurrent commands`` () =
        task {
            let tasks = [
                TestHelpers.assertCliCommandSucceeds ["version"]
                TestHelpers.assertCliCommandSucceeds ["help"]
                TestHelpers.assertCliCommandSucceeds ["swarm"; "status"]
            ]
            
            let! results = Task.WhenAll(tasks)
            
            results |> Array.length |> should equal 3
            results |> Array.iter (fun result -> result |> should not' (be null))
        }

    [<Fact>]
    let ``CLI memory usage should be reasonable`` () =
        task {
            let initialMemory = GC.GetTotalMemory(true)
            
            let! output = TestHelpers.assertCliCommandSucceeds ["version"]
            
            let finalMemory = GC.GetTotalMemory(true)
            let memoryIncrease = finalMemory - initialMemory
            
            output |> should contain "TARS CLI"
            memoryIncrease |> should be (lessThan 100_000_000L) // 100MB max increase
        }
