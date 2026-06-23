namespace Tars.Tests

open Xunit
open Tars.Evolution
open Tars.Evolution.SelfImproveBacklog

/// Unit tests for the self-improve backlog loader (ADR 0002 D5): JSON → typed
/// entries, tolerant of missing optional fields, strict on required ones.
module SelfImproveBacklogTests =

    [<Fact>]
    let ``parse reads a well-formed entry`` () =
        let json =
            """[{"id":"search","target_test":"Search is a first-class agent skill",
                 "target_file":"v2/src/Tars.Core/AgentDefinition.fs",
                 "test_project":"v2/tests/Tars.Tests/Tars.Tests.fsproj",
                 "rationale":"add a Search skill"}]"""
        match parse json with
        | Result.Ok [ e ] ->
            Assert.Equal("search", e.Id)
            Assert.Equal("Search is a first-class agent skill", e.TargetTest)
            Assert.Equal("v2/src/Tars.Core/AgentDefinition.fs", e.TargetFile)
            Assert.Equal("add a Search skill", e.Rationale)
        | other -> failwithf "expected one entry, got %A" other

    [<Fact>]
    let ``parse defaults a missing rationale to empty`` () =
        let json =
            """[{"id":"x","target_test":"t","target_file":"f.fs","test_project":"p.fsproj"}]"""
        match parse json with
        | Result.Ok [ e ] -> Assert.Equal("", e.Rationale)
        | other -> failwithf "expected one entry, got %A" other

    [<Fact>]
    let ``parse skips entries missing a required field`` () =
        // First entry lacks target_file → dropped; second is complete → kept.
        let json =
            """[{"id":"bad","target_test":"t","test_project":"p.fsproj"},
                {"id":"good","target_test":"t","target_file":"f.fs","test_project":"p.fsproj"}]"""
        match parse json with
        | Result.Ok [ e ] -> Assert.Equal("good", e.Id)
        | other -> failwithf "expected only the complete entry, got %A" other

    [<Fact>]
    let ``parse errors when the root is not an array`` () =
        match parse """{"id":"x"}""" with
        | Result.Error _ -> ()
        | Result.Ok _ -> failwith "expected Error for a non-array root"

    [<Fact>]
    let ``parse errors on malformed JSON`` () =
        match parse "not json" with
        | Result.Error _ -> ()
        | Result.Ok _ -> failwith "expected Error for malformed JSON"
