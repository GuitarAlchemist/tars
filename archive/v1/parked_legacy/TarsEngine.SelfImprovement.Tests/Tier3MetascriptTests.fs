namespace TarsEngine.SelfImprovement.Tests

open System
open System.IO
open Microsoft.Extensions.Logging.Abstractions
open Xunit
open TarsEngine.FSharp.SelfImprovement

module Tier3MetascriptTests =

    let private createTempDirectory prefix =
        let path = Path.Combine(Path.GetTempPath(), $"{prefix}-{Guid.NewGuid():N}")
        Directory.CreateDirectory(path) |> ignore
        path

    let private writeScript (directory: string) (fileName: string) (content: string) =
        let path = Path.Combine(directory, fileName)
        File.WriteAllText(path, content)

    let private runTier3 baseDir =
        SelfImprovementHelpers.runTier3Metascripts NullLoggerFactory.Instance (Some baseDir)
        |> Async.RunSynchronously

    let private successScript =
        """```text
All systems nominal.
```"""

    let private failingScript =
        """```fsharp
failwith "analysis failed"
```"""

    [<Fact>]
    let ``runTier3Metascripts captures success metrics`` () =
        let root = createTempDirectory "tier3-success"
        let tier3Dir = Path.Combine(root, "tier3")
        Directory.CreateDirectory(tier3Dir) |> ignore

        try
            writeScript tier3Dir "safety-review.trsx" successScript
            writeScript tier3Dir "performance-review.trsx" successScript
            writeScript tier3Dir "critic-federation.trsx" successScript

            let outcomes, metrics = runTier3 tier3Dir
            Assert.Equal(3, outcomes.Length)

            let status = metrics["tier3.agent.tier3-safety.status"] :?> string
            Assert.Equal("success", status)

            let totalBlocks = metrics["tier3.agent.tier3-safety.blocks.total"] :?> int
            Assert.Equal(1, totalBlocks)

            let confidence = metrics["tier3.agent.tier3-safety.confidence"] :?> float
            Assert.True(confidence >= 0.89 && confidence <= 0.91)
        finally
            try
                Directory.Delete(root, true)
            with _ -> ()

    [<Fact>]
    let ``runTier3Metascripts records partial failures`` () =
        let root = createTempDirectory "tier3-partial"
        let tier3Dir = Path.Combine(root, "tier3")
        Directory.CreateDirectory(tier3Dir) |> ignore

        try
            writeScript tier3Dir "safety-review.trsx" successScript
            writeScript tier3Dir "performance-review.trsx" successScript
            writeScript tier3Dir "critic-federation.trsx" failingScript

            let _, metrics = runTier3 tier3Dir

            let criticStatus = metrics["tier3.agent.tier3-critic.status"] :?> string
            Assert.Equal("partial", criticStatus)

            let failureBlocks =
                metrics
                |> Map.tryFind "tier3.agent.tier3-critic.blocks.failure"
                |> Option.map (fun value -> value :?> int)
                |> Option.defaultValue 0

            Assert.Equal(1, failureBlocks)
        finally
            try
                Directory.Delete(root, true)
            with _ -> ()

    [<Fact>]
    let ``runTier3Metascripts reports missing metascripts`` () =
        let root = createTempDirectory "tier3-missing"
        let tier3Dir = Path.Combine(root, "tier3")
        Directory.CreateDirectory(tier3Dir) |> ignore

        try
            writeScript tier3Dir "safety-review.trsx" successScript
            writeScript tier3Dir "critic-federation.trsx" successScript
            // intentionally omit performance metascript

            let _, metrics = runTier3 tier3Dir

            let status = metrics["tier3.agent.tier3-performance.status"] :?> string
            Assert.Equal("missing", status)

            let totalBlocks = metrics["tier3.agent.tier3-performance.blocks.total"] :?> int
            Assert.Equal(0, totalBlocks)
        finally
            try
                Directory.Delete(root, true)
            with _ -> ()
