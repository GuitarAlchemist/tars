module Tars.Tests.TarsReviewerTests

open System
open System.IO
open System.Threading.Tasks
open Xunit
open Tars.Core
open Tars.Llm
open Tars.Core.WorkflowOfThought
open Tars.Interface.Cli.Commands

// TARS's vote in the cross-review (`tars review`). Each test is written from a
// way the vote could claim a review that did not happen.

/// Answers with whatever `answer` returns, and counts the calls.
type private ScriptedLlm(answer: unit -> string) =
    let mutable calls = 0
    member _.Calls = calls

    interface ILlmService with
        member _.CompleteAsync(_) =
            calls <- calls + 1

            Task.FromResult
                { Text = answer ()
                  FinishReason = Some "stop"
                  Usage = None
                  Raw = None }

        member this.CompleteStreamAsync(req, _) = (this :> ILlmService).CompleteAsync req
        member _.EmbedAsync(_) = Task.FromResult([| 0.0f |])

        member _.RouteAsync(_) =
            Task.FromResult(
                ({ Backend = Tars.Llm.LlmBackend.Ollama "stub"
                   Endpoint = Uri "http://localhost:11434"
                   ApiKey = None }
                : Tars.Llm.Routing.RoutedBackend)
            )

let private agent: AgentConfig option =
    Some
        { Role = "CodeReviewer"
          SystemPrompt = "Review the diff."
          ModelHint = Some "code"
          Temperature = Some 0.1
          Description = None }

let private sha = "abcdef1234567890abcdef1234567890abcdef12"
let private diff = "diff --git a/A.fs b/A.fs\n@@ -1 +1 @@\n-let x = 1\n+let x = 2\n"

let private reviewWith (llm: ScriptedLlm) (agent: AgentConfig option) (diff: string) =
    (ReviewCommand.review llm agent sha (Some "qwen2.5-coder:7b") diff).Result

[<Fact>]
let ``A clean answer becomes an advisory clean vote on the commit`` () =
    let vote = reviewWith (ScriptedLlm(fun () -> "VOTE: clean")) agent diff
    Assert.StartsWith($"Cross-review vote (TARS): clean @ {sha}", vote)
    Assert.Contains("Advisory", vote)

[<Fact>]
let ``An answer without a vote line is not counted as a review`` () =
    let vote = reviewWith (ScriptedLlm(fun () -> "Looks good to me!")) agent diff
    Assert.StartsWith($"Cross-review vote (TARS): not-reviewed @ {sha}", vote)

[<Fact>]
let ``Only the vote and the finding lines are kept from the answer`` () =
    let answer =
        "Sure! Here is my review.\nVOTE: to-fix\n- [P2] v2/src/A.fs:12 - drops the error\nHope this helps"

    let vote = reviewWith (ScriptedLlm(fun () -> answer)) agent diff
    Assert.StartsWith($"Cross-review vote (TARS): to-fix @ {sha}", vote)
    Assert.Contains("- [P2] v2/src/A.fs:12 - drops the error", vote)
    Assert.DoesNotContain("Hope this helps", vote)
    Assert.DoesNotContain("Sure!", vote)

[<Fact>]
let ``A diff too large to read is not reviewed, and the model is not asked`` () =
    let llm = ScriptedLlm(fun () -> "VOTE: clean")
    let vote = reviewWith llm agent (String('x', ReviewCommand.contextWindow))
    Assert.StartsWith($"Cross-review vote (TARS): not-reviewed @ {sha}", vote)
    Assert.Equal(0, llm.Calls)

[<Fact>]
let ``A diff of few characters but many bytes is measured in bytes, not characters`` () =
    // 12,000 characters, 36,000 UTF-8 bytes: it could need more tokens than the context holds.
    let llm = ScriptedLlm(fun () -> "VOTE: clean")
    let vote = reviewWith llm agent (String('語', 12_000))
    Assert.StartsWith($"Cross-review vote (TARS): not-reviewed @ {sha}", vote)
    Assert.Equal(0, llm.Calls)

[<Fact>]
let ``A model that cannot be reached yields not-reviewed, not a vote`` () =
    let vote =
        reviewWith (ScriptedLlm(fun () -> raise (Net.Http.HttpRequestException "connection refused"))) agent diff

    Assert.StartsWith($"Cross-review vote (TARS): not-reviewed @ {sha}", vote)
    Assert.Contains("connection refused", vote)

[<Fact>]
let ``A missing agent definition yields not-reviewed`` () =
    let llm = ScriptedLlm(fun () -> "VOTE: clean")
    let vote = reviewWith llm None diff
    Assert.StartsWith($"Cross-review vote (TARS): not-reviewed @ {sha}", vote)
    Assert.Equal(0, llm.Calls)

[<Fact>]
let ``The declared code-reviewer agent asks for the vote format the command reads`` () =
    let rec v2Root (dir: DirectoryInfo) =
        if isNull dir then
            failwith "Could not locate v2/ (no Tars.sln above the test directory)"
        elif dir.GetFiles("Tars.sln").Length > 0 then
            dir.FullName
        else
            v2Root dir.Parent

    let path =
        Path.Combine(v2Root (DirectoryInfo(Directory.GetCurrentDirectory())), "agents", "code-reviewer.md")

    match AgentDefinitionParser.loadFile path with
    | Result.Ok def ->
        Assert.Equal("CodeReviewer", def.Role)
        Assert.Equal(Some "code", def.ModelHint)
        Assert.Contains("VOTE: blocking", def.SystemPrompt)
        Assert.Contains("- [P1] path/to/file.fs:123", def.SystemPrompt)
    | Result.Error e -> Assert.Fail($"code-reviewer.md did not parse: {e}")
