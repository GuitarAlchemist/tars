namespace Tars.Tests

open System
open System.IO
open Xunit
open Tars.Core
open Tars.Core.Metrics

type MetricsTests() =

    [<Fact>]
    member _.``Metrics can record and flush to csv``() =
        Metrics.clear()
        Metrics.recordSimple "agent.bind" "success" None None None
        Metrics.record "budget.check" "ok" 1.5 None (Map [ "remaining", "100" ])

        let path = Path.Combine(Path.GetTempPath(), $"tars-metrics-{Guid.NewGuid()}.csv")
        Metrics.dumpCsv path

        Assert.True(File.Exists path)
        let lines = File.ReadAllLines path
        Assert.True(lines.Length >= 2)
        Assert.Contains("agent.bind", String.Join("", lines))
        Assert.Contains("budget.check", String.Join("", lines))
