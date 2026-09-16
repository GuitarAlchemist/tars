namespace Tars.Tests

open System
open System.IO
open System.Threading.Tasks
open Xunit
open Tars.Llm
open Tars.Evolution.SelfHostingGate

/// Orchestration tests for the self-hosting gate (#245): worktree -> apply -> test ->
/// decide -> promote, driven through GateEffects with a scripted git/dotnet instead of
/// real processes, so no worktree, build or ~/.tars write happens.
module SelfHostingGateOrchestrationTests =

    let private targetFile = "src/Calc.fs"
    let private testProject = "tests/Calc.Tests/Calc.Tests.fsproj"
    let private targetTest = "Calc.adds"

    /// A replying stub: every completion returns `reply`.
    type private ReplyLlm(reply: string) =
        interface ILlmService with
            member _.CompleteAsync(_) =
                Task.FromResult
                    { Text = reply
                      FinishReason = Some "stop"
                      Usage = None
                      Raw = None }

            member _.CompleteStreamAsync(_, _) = failwith "not used"
            member _.EmbedAsync(_) = failwith "not used"
            member _.RouteAsync(_) = failwith "not used"

    let private proposal (newText: string) =
        sprintf """{"rationale":"fix add","edits":[{"old_text":"a - b","new_text":"%s"}]}""" newText

    /// Scripted git + dotnet over a throwaway repo directory. `worktree add` copies the
    /// repo into the worktree path; `dotnet test` writes a TRX in which the target test
    /// passes iff the worktree's Calc.fs adds.
    type private FakeProcesses(repo: string) =
        let calls = ResizeArray<string * string>()
        let worktrees = ResizeArray<string>()
        let wins = ResizeArray<GateTask>()
        let quoted (args: string) = args.Split('"').[1]

        member _.Calls = List.ofSeq calls
        member _.Worktrees = List.ofSeq worktrees
        member _.Wins = List.ofSeq wins

        member _.Effects(?failTests: bool) : GateEffects =
            let failTests = defaultArg failTests false

            { Run =
                fun workdir exe args ->
                    lock calls (fun () -> calls.Add((exe, args)))

                    match exe with
                    | "git" when args.StartsWith "worktree add" ->
                        let wt = quoted args
                        lock worktrees (fun () -> worktrees.Add wt)
                        let src = Path.Combine(wt, "src")
                        Directory.CreateDirectory src |> ignore
                        File.Copy(Path.Combine(repo, targetFile), Path.Combine(wt, targetFile))
                        0, "", ""
                    | "git" when args.StartsWith "worktree remove" ->
                        Directory.Delete(quoted args, true)
                        0, "", ""
                    | "git" -> 0, "", ""
                    | "dotnet" when failTests -> failwith "dotnet crashed"
                    | "dotnet" ->
                        let trxName = args.Substring(args.IndexOf("LogFileName=") + "LogFileName=".Length).TrimEnd('"')
                        let adds = File.ReadAllText(Path.Combine(workdir, targetFile)).Contains "a + b"
                        let outcome = if adds then "Passed" else "Failed"

                        let trx =
                            sprintf
                                """<TestRun xmlns="http://microsoft.com/schemas/VisualStudio/TeamTest/2010"><Results><UnitTestResult testName="%s" outcome="%s" /><UnitTestResult testName="Calc.subtracts" outcome="Passed" /></Results></TestRun>"""
                                targetTest
                                outcome

                        let dir = Path.Combine(workdir, "tests", "Calc.Tests", "TestResults")
                        Directory.CreateDirectory dir |> ignore
                        File.WriteAllText(Path.Combine(dir, trxName), trx)
                        0, "", ""
                    | _ -> failwithf "unexpected process %s %s" exe args
              RecordWin = fun task -> lock wins (fun () -> wins.Add task) }

    let private withRepo (f: string -> unit) =
        let repo = Path.Combine(Path.GetTempPath(), $"tars-gate-repo-{Guid.NewGuid():N}")
        Directory.CreateDirectory(Path.Combine(repo, "src")) |> ignore
        File.WriteAllText(Path.Combine(repo, targetFile), "let add a b = a - b\n")

        try
            f repo
        finally
            Directory.Delete(repo, true)

    let private runGate (fx: GateEffects) repo reply =
        runGateBestOfNWith fx (ReplyLlm reply) repo testProject targetTest targetFile 2 2
        |> Async.RunSynchronously

    let private assertCleanedUp (fake: FakeProcesses) repo =
        Assert.NotEmpty fake.Worktrees

        for wt in fake.Worktrees do
            Assert.False(Directory.Exists wt, $"worktree left behind: {wt}")

        let removes = fake.Calls |> List.filter (fun (_, a) -> a.StartsWith "worktree remove")
        Assert.Equal(fake.Worktrees.Length, removes.Length)
        // The live tree is never edited; only worktrees are.
        Assert.Equal("let add a b = a - b\n", File.ReadAllText(Path.Combine(repo, targetFile)))

    [<Fact>]
    let ``a fixing proposal is promoted to a self-improve branch and recorded as a win`` () =
        withRepo (fun repo ->
            let fake = FakeProcesses repo

            match runGate (fake.Effects()) repo (proposal "a + b") with
            | Promoted(branch, _) -> Assert.StartsWith("self-improve/", branch)
            | Rejected reason -> failwithf "expected promotion, got: %s" reason

            Assert.Contains(fake.Calls, fun (exe, a) -> exe = "git" && a.StartsWith "commit")
            let win = Assert.Single fake.Wins
            Assert.Equal("a + b", (List.exactlyOne win.Edits).NewText)
            assertCleanedUp fake repo)

    [<Fact>]
    let ``a proposal that does not fix the test is rejected without commit or win`` () =
        withRepo (fun repo ->
            let fake = FakeProcesses repo

            match runGate (fake.Effects()) repo (proposal "a * b") with
            | Rejected reason -> Assert.Contains("does not pass", reason)
            | Promoted(branch, _) -> failwithf "expected rejection, got promotion to %s" branch

            Assert.DoesNotContain(fake.Calls, fun (exe, a) -> exe = "git" && a.StartsWith "commit")
            Assert.Empty fake.Wins
            assertCleanedUp fake repo)

    [<Fact>]
    let ``a crashing test run still removes the worktree`` () =
        withRepo (fun repo ->
            let fake = FakeProcesses repo

            let ex =
                Assert.ThrowsAny<exn>(fun () -> runGate (fake.Effects(failTests = true)) repo (proposal "a + b") |> ignore)

            Assert.Contains("dotnet crashed", ex.ToString())
            Assert.Empty fake.Wins
            assertCleanedUp fake repo)
