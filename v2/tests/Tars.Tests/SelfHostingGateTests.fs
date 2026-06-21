namespace Tars.Tests

open Xunit
open Tars.Evolution
open Tars.Evolution.SelfHostingGate

/// Unit tests for the pure (git/dotnet-independent) decision logic of the
/// self-hosting gate: the hermetic boundary, TRX parsing, and the accept/reject
/// rules — especially the anti-gaming (Goodhart) rejections.
module SelfHostingGateTests =

    // ── isTestFile (hermetic boundary) ────────────────────────────────────────

    [<Theory>]
    [<InlineData("v2/tests/Tars.Tests/Foo.fs", true)>]
    [<InlineData("tests/X.fs", true)>]
    [<InlineData("v2/src/Tars.Evolution/SelfImprovementTests.fs", true)>]
    [<InlineData("v2\\tests\\Tars.Tests\\Bar.fs", true)>]
    [<InlineData("v2/src/Tars.Evolution/GrammarMeshBridge.fs", false)>]
    [<InlineData("v2/src/Tars.Core/IxSkill.fs", false)>]
    let ``isTestFile flags only test files`` (path: string) (expected: bool) =
        Assert.Equal(expected, isTestFile path)

    // ── parseTrx ──────────────────────────────────────────────────────────────

    let private trx (rows: (string * string) list) =
        let body =
            rows
            |> List.map (fun (n, o) -> sprintf "<UnitTestResult testName=\"%s\" outcome=\"%s\" />" n o)
            |> String.concat ""
        sprintf
            "<TestRun xmlns=\"http://microsoft.com/schemas/VisualStudio/TeamTest/2010\"><Results>%s</Results></TestRun>"
            body

    [<Fact>]
    let ``parseTrx extracts testName -> outcome`` () =
        let m = parseTrx (trx [ "A.t1", "Passed"; "A.t2", "Failed" ])
        Assert.Equal(2, m.Count)
        Assert.Equal("Passed", m.["A.t1"])
        Assert.Equal("Failed", m.["A.t2"])

    [<Fact>]
    let ``parseTrx returns empty on malformed xml`` () =
        Assert.True((parseTrx "not xml").IsEmpty)

    // ── decide — accept path ──────────────────────────────────────────────────

    [<Fact>]
    let ``decide accepts a clean target flip with zero regressions`` () =
        let baseline = Map [ "T.target", "Failed"; "T.other", "Passed" ]
        let variant = Map [ "T.target", "Passed"; "T.other", "Passed" ]
        match decide "T.target" baseline variant with
        | Accept _ -> ()
        | Reject r -> failwithf "expected Accept, got Reject: %s" r

    // ── decide — anti-gaming rejections ───────────────────────────────────────

    [<Fact>]
    let ``decide rejects a regression`` () =
        // target flips to pass, but an unrelated test was broken
        let baseline = Map [ "T.target", "Failed"; "T.other", "Passed" ]
        let variant = Map [ "T.target", "Passed"; "T.other", "Failed" ]
        match decide "T.target" baseline variant with
        | Reject r -> Assert.Contains("regression", r)
        | Accept _ -> failwith "expected Reject (regression)"

    [<Fact>]
    let ``decide rejects a dropped test (skip/delete gaming)`` () =
        // target passes, no regression — but a test vanished from the set
        let baseline = Map [ "T.target", "Failed"; "T.other", "Passed" ]
        let variant = Map [ "T.target", "Passed" ]
        match decide "T.target" baseline variant with
        | Reject r -> Assert.Contains("hermetic", r)
        | Accept _ -> failwith "expected Reject (test set changed)"

    [<Fact>]
    let ``decide rejects when target already passed (no improvement)`` () =
        let baseline = Map [ "T.target", "Passed"; "T.other", "Passed" ]
        let variant = Map [ "T.target", "Passed"; "T.other", "Passed" ]
        match decide "T.target" baseline variant with
        | Reject r -> Assert.Contains("already passed", r)
        | Accept _ -> failwith "expected Reject (no improvement)"

    [<Fact>]
    let ``decide rejects when target does not pass in variant`` () =
        let baseline = Map [ "T.target", "Failed"; "T.other", "Passed" ]
        let variant = Map [ "T.target", "Failed"; "T.other", "Passed" ]
        match decide "T.target" baseline variant with
        | Reject r -> Assert.Contains("does not pass", r)
        | Accept _ -> failwith "expected Reject (target still failing)"

    [<Fact>]
    let ``decide rejects empty results (build failure)`` () =
        match decide "T.target" Map.empty Map.empty with
        | Reject r -> Assert.Contains("no test results", r)
        | Accept _ -> failwith "expected Reject (no results)"
