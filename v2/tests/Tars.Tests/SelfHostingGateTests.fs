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

    // ── applyEditPure (EOL reconciliation — the Windows CRLF guard) ───────────

    [<Fact>]
    let ``applyEditPure lands an LF edit into a CRLF file`` () =
        // The model emits LF in old_text; the file is CRLF. Without EOL
        // reconciliation this misses entirely (the live self-improve bug).
        let content = "let a = 1\r\nlet b = 2\r\nlet c = 3\r\n"
        let result = applyEditPure content "let a = 1\nlet b = 2" "let a = 1\nlet b = 20"
        match result with
        | Some updated ->
            Assert.Equal("let a = 1\r\nlet b = 20\r\nlet c = 3\r\n", updated)
        | None -> failwith "expected the LF edit to apply to the CRLF file"

    [<Fact>]
    let ``applyEditPure preserves LF when the file is LF`` () =
        let content = "let a = 1\nlet b = 2\n"
        match applyEditPure content "let b = 2" "let b = 22" with
        | Some updated -> Assert.Equal("let a = 1\nlet b = 22\n", updated)
        | None -> failwith "expected the edit to apply"

    [<Fact>]
    let ``applyEditPure refuses a non-unique match`` () =
        let content = "x\r\nx\r\n"
        Assert.True((applyEditPure content "x" "y").IsNone)

    [<Fact>]
    let ``applyEditPure refuses a missing match`` () =
        Assert.True((applyEditPure "let a = 1\r\n" "let z = 9" "let z = 0").IsNone)

    // ── viableProposals (best-of-N pre-filter, ADR 0002 D5) ───────────────────

    let private edit old new' = { OldText = old; NewText = new' }

    let private mkTask old new' =
        { TargetTest = "t"; TargetFile = "Lib.fs"; Edits = [ edit old new' ]; Rationale = "r" }

    [<Fact>]
    let ``viableProposals keeps only applicable, de-duplicated edits in order`` () =
        let content = "let a = 1\r\nlet b = 2\r\nlet c = 3\r\n"
        let proposals =
            [ mkTask "let b = 2" "let b = 20" // applies
              mkTask "let z = 9" "let z = 0" // missing -> dropped
              mkTask "let b = 2" "let b = 20" // duplicate -> dropped
              mkTask "let c = 3" "let c = 30" ] // applies
        let viable = viableProposals content proposals
        Assert.Equal(2, List.length viable)
        Assert.Equal("let b = 2", viable.[0].Edits.Head.OldText)
        Assert.Equal("let c = 3", viable.[1].Edits.Head.OldText)

    [<Fact>]
    let ``viableProposals returns empty when nothing applies`` () =
        let content = "let a = 1\r\n"
        Assert.Empty(viableProposals content [ mkTask "nope" "x"; mkTask "also nope" "y" ])

    [<Fact>]
    let ``viableProposals keeps a multi-edit proposal whose edits all apply`` () =
        let content = "let a = 1\r\nlet b = 2\r\n"
        let multi =
            { TargetTest = "t"
              TargetFile = "Lib.fs"
              Edits = [ edit "let a = 1" "let a = 10"; edit "let b = 2" "let b = 20" ]
              Rationale = "r" }
        Assert.Equal(1, List.length (viableProposals content [ multi ]))

    // ── applyEditsPure (multi-edit fold, ADR 0002 D5) ─────────────────────────

    [<Fact>]
    let ``applyEditsPure applies two edits in sequence`` () =
        // The unlock: a fix spanning two locations (a DU case AND its parse arm) —
        // impossible with one contiguous replacement.
        let content = "type T = A | B\r\nlet parse c = match c with 'a' -> A | _ -> B\r\n"
        let edits =
            [ edit "type T = A | B" "type T = A | B | C"
              edit "'a' -> A | _ -> B" "'a' -> A | 'c' -> C | _ -> B" ]
        match applyEditsPure content edits with
        | Some updated ->
            Assert.Contains("A | B | C", updated)
            Assert.Contains("'c' -> C", updated)
        | None -> failwith "expected both edits to apply"

    [<Fact>]
    let ``applyEditsPure fails when any edit does not apply (nothing written)`` () =
        let content = "let a = 1\r\n"
        Assert.True((applyEditsPure content [ edit "let a = 1" "let a = 2"; edit "nope" "x" ]).IsNone)

    [<Fact>]
    let ``applyEditsPure rejects an empty edit list`` () =
        Assert.True((applyEditsPure "let a = 1\r\n" []).IsNone)

    // ── buildSftExample (ADR 0003 coupling) ───────────────────────────────────

    [<Fact>]
    let ``buildSftExample emits a valid messages line with the verified edit as target`` () =
        let task =
            { TargetTest = "answer is 42"
              TargetFile = "Lib.fs"
              Edits = [ { OldText = "answer () = 0"; NewText = "answer () = 42" } ]
              Rationale = "fix answer" }
        let line = buildSftExample task
        use doc = System.Text.Json.JsonDocument.Parse(line)
        let msgs = doc.RootElement.GetProperty("messages")
        Assert.Equal(3, msgs.GetArrayLength())
        let roles = [ for m in msgs.EnumerateArray() -> m.GetProperty("role").GetString() ]
        Assert.Equal("system", List.item 0 roles)
        Assert.Equal("user", List.item 1 roles)
        Assert.Equal("assistant", List.item 2 roles)
        // The assistant target is the mutation JSON carrying the verified new_text —
        // the exact thing the generator must learn to emit (ADR 0003 D2).
        let assistant =
            (msgs.EnumerateArray() |> Seq.last).GetProperty("content").GetString()
        Assert.Contains("new_text", assistant)
        Assert.Contains("answer () = 42", assistant)

    // ── parseProposal (LLM generation; self-driving) ──────────────────────────

    [<Fact>]
    let ``parseProposal extracts a clean single mutation`` () =
        let r = parseProposal """{"rationale":"fix","old_text":"a","new_text":"b"}"""
        match r with
        | Ok(rat, edits) ->
            Assert.Equal("fix", rat)
            Assert.Equal(1, List.length edits)
            Assert.Equal("a", edits.Head.OldText)
            Assert.Equal("b", edits.Head.NewText)
        | Error e -> failwithf "expected Ok, got %s" e

    [<Fact>]
    let ``parseProposal extracts a multi-edit array in order`` () =
        let r =
            parseProposal
                """{"rationale":"two spots","edits":[{"old_text":"a","new_text":"b"},{"old_text":"c","new_text":"d"}]}"""
        match r with
        | Ok(rat, edits) ->
            Assert.Equal("two spots", rat)
            Assert.Equal(2, List.length edits)
            Assert.Equal("a", edits.[0].OldText)
            Assert.Equal("d", edits.[1].NewText)
        | Error e -> failwithf "expected Ok, got %s" e

    [<Fact>]
    let ``parseProposal tolerates prose around the JSON and old/new fallback keys`` () =
        let r = parseProposal "Sure! Here is the fix:\n{\"old\":\"x\",\"new\":\"y\"}\nDone."
        match r with
        | Ok(_, edits) ->
            Assert.Equal("x", edits.Head.OldText)
            Assert.Equal("y", edits.Head.NewText)
        | Error e -> failwithf "expected Ok, got %s" e

    [<Fact>]
    let ``parseProposal errors on missing fields and on no JSON`` () =
        Assert.True(
            (match parseProposal """{"rationale":"x"}""" with
             | Error _ -> true
             | Ok _ -> false))
        Assert.True(
            (match parseProposal "no json here" with
             | Error _ -> true
             | Ok _ -> false))

    [<Fact>]
    let ``buildProposePrompt includes the test, file, and source`` () =
        let p = buildProposePrompt "MyTest" "Lib.fs" "let x = 1"
        Assert.Contains("MyTest", p)
        Assert.Contains("Lib.fs", p)
        Assert.Contains("let x = 1", p)

    // ── buildRepairPrompt (D5 repair tail) ────────────────────────────────────

    [<Fact>]
    let ``buildRepairPrompt carries the failing variant's error and prior edit`` () =
        let failed =
            { TargetTest = "MyTest"
              TargetFile = "Lib.fs"
              Edits = [ { OldText = "let x = 1"; NewText = "let x = 2" } ]
              Rationale = "bump x" }
        let error = "1 regression(s): T.other"
        let p = buildRepairPrompt "MyTest" "Lib.fs" "let x = 1\nlet y = 9" failed error
        // The rejection reason is the signal that makes this a repair, not a re-roll.
        Assert.Contains(error, p)
        // The model also sees the edit it tried, so it can correct rather than repeat.
        Assert.Contains("let x = 1", p)
        Assert.Contains("let x = 2", p)
        Assert.Contains("MyTest", p)

    [<Fact>]
    let ``repairRank ranks build/run failures above apply/build failures`` () =
        // A variant that ran and regressed teaches more than one that never ran.
        Assert.True(repairRank "1 regression(s): T.other" > repairRank "no test results parsed (build or run failure)")
        Assert.True(repairRank "target test 'T' does not pass in the variant" > repairRank "edit precondition failed")
