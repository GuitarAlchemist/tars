module Tars.Interface.Cli.Commands.ReviewCommand

open System
open System.IO
open System.Text.RegularExpressions
open System.Threading.Tasks
open Tars.Llm
open Tars.Core.WorkflowOfThought
open Tars.Interface.Cli

// `tars review <diff-file> --sha <commit> --out <file> [--model <name>]`
//
// TARS's vote in the cross-review (.github/scripts/cross_review.py): the declared
// agent v2/agents/code-reviewer.md reads a pull request diff and answers with a
// vote. It has no tools, so the diff can only be read, never acted on.
//
// Every run writes a vote. When nothing was reviewed (no agent, a diff too large
// to read, a model that could not be reached, an answer with no vote), the vote
// says "not-reviewed" and why, so it is never mistaken for a clean review.

/// The most diff this reviewer reads. A larger diff is not reviewed at all:
/// reviewing its first part only would read as a vote on the whole PR.
let maxDiffChars = 60_000

let private voteLine =
    System.Text.RegularExpressions.Regex(@"^\s*VOTE:\s*(blocking|to-fix|clean)\b", RegexOptions.IgnoreCase ||| RegexOptions.Multiline)

let private findingLine = System.Text.RegularExpressions.Regex(@"^- \[P[0-3]\] .+?:\d+.*$", RegexOptions.Multiline)

let private footer (model: string) =
    $"\n\nTARS agent `code-reviewer` on `{model}`. Advisory: this vote is shown in the cross-review but not counted in its verdict."

let notReviewed (sha: string) (model: string) (reason: string) =
    $"Cross-review vote (TARS): not-reviewed @ {sha}\n\n{reason}" + footer model

/// The comment to post for the model's answer. Only the VOTE line and the
/// finding lines are kept; an answer without a VOTE line is not a vote.
let formatVote (sha: string) (model: string) (answer: string) =
    let m = voteLine.Match(answer)

    if not m.Success then
        notReviewed sha model "The model's answer had no VOTE line, so it is not counted as a review."
    else
        let findings =
            findingLine.Matches(answer) |> Seq.map (fun f -> f.Value.Trim()) |> Seq.toList

        let body =
            if findings.IsEmpty then
                ""
            else
                "\n\n" + String.Join("\n", findings)

        $"Cross-review vote (TARS): {m.Groups.[1].Value.ToLowerInvariant()} @ {sha}"
        + body
        + footer model

/// Ask the agent for a vote on the diff.
let review
    (llm: ILlmService)
    (agent: AgentConfig option)
    (sha: string)
    (model: string option)
    (diff: string)
    : Task<string> =
    task {
        let shown = model |> Option.defaultValue "the configured model"

        match agent with
        | None ->
            return notReviewed sha shown "The agent definition `code-reviewer` was not found, so nothing was reviewed."
        | Some _ when String.IsNullOrWhiteSpace diff ->
            return notReviewed sha shown "The diff is empty, so there was nothing to review."
        | Some _ when diff.Length > maxDiffChars ->
            return
                notReviewed
                    sha
                    shown
                    $"The diff is {diff.Length} characters, more than the {maxDiffChars} this reviewer reads, so none of it was reviewed."
        | Some agent ->
            let request =
                { LlmRequest.Default with
                    ModelHint = agent.ModelHint
                    Model = model
                    SystemPrompt = Some agent.SystemPrompt
                    Temperature = agent.Temperature |> Option.orElse (Some 0.1)
                    MaxTokens = Some 2048
                    // Ollama's default context would cut the diff off without saying so.
                    ContextWindow = Some 32768
                    Messages =
                        [ { Role = Role.User
                            Content = $"Commit: {sha}\n\n```diff\n{diff}\n```" } ] }

            try
                let! response = llm.CompleteAsync request

                if response.FinishReason = Some "parse_error" then
                    return notReviewed sha shown "The model's response could not be parsed, so nothing was reviewed."
                else
                    return formatVote sha shown response.Text
            with ex ->
                return notReviewed sha shown $"The model could not be reached, so nothing was reviewed: {ex.Message}"
    }

let private usage () =
    eprintfn "Usage: tars review <diff-file> --sha <commit> --out <file> [--model <name>]"
    2

let run (rt: ITarsRuntime) (args: string list) =
    task {
        let option name =
            args
            |> List.pairwise
            |> List.tryPick (fun (k, v) -> if k = name then Some v else None)

        match args, option "--sha", option "--out" with
        | diffPath :: _, Some sha, Some outPath when not (diffPath.StartsWith "--") ->
            let model = option "--model"

            let llm =
                rt.Llm(
                    match model with
                    | Some m -> ModelChoice.Model m
                    | None -> ModelChoice.Default
                )

            let! vote = review llm (AgentRegistry.get "CodeReviewer") sha model (File.ReadAllText diffPath)
            File.WriteAllText(outPath, vote + "\n")
            printfn "%s" vote
            return 0
        | _ -> return usage ()
    }
