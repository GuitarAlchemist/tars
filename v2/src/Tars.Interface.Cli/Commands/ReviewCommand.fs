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

/// The model's context, in tokens. Ollama's default is far smaller and would
/// cut the diff off without saying so, so every request sets it.
let contextWindow = 32768

let private maxAnswerTokens = 2048

/// Room for the chat template's own tokens around the messages.
let private templateMargin = 256

/// Whether the whole request is sure to fit in the context. A token always
/// covers at least one byte, so text of N UTF-8 bytes is at most N tokens.
/// Counting characters would let dense text (CJK, minified code) overflow the
/// context, and the model would then vote on a diff it only partly read. A
/// diff that may not fit is not reviewed at all.
let fitsContext (systemPrompt: string) (userMessage: string) =
    Text.Encoding.UTF8.GetByteCount systemPrompt
    + Text.Encoding.UTF8.GetByteCount userMessage
    + maxAnswerTokens
    + templateMargin
    <= contextWindow

let private voteLine =
    System.Text.RegularExpressions.Regex(
        @"^\s*VOTE:\s*(blocking|to-fix|clean)\b",
        RegexOptions.IgnoreCase ||| RegexOptions.Multiline
    )

let private findingLine =
    System.Text.RegularExpressions.Regex(@"^- \[P[0-3]\] .+?:\d+.*$", RegexOptions.Multiline)

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
        | Some agent ->
            let message = $"Commit: {sha}\n\n```diff\n{diff}\n```"

            if not (fitsContext agent.SystemPrompt message) then
                return
                    notReviewed
                        sha
                        shown
                        $"The diff is {Text.Encoding.UTF8.GetByteCount diff} bytes. With the prompt and room for the answer it may not fit in the model's {contextWindow}-token context, so none of it was reviewed."
            else
                let request =
                    { LlmRequest.Default with
                        ModelHint = agent.ModelHint
                        Model = model
                        SystemPrompt = Some agent.SystemPrompt
                        Temperature = agent.Temperature |> Option.orElse (Some 0.1)
                        MaxTokens = Some maxAnswerTokens
                        ContextWindow = Some contextWindow
                        Messages = [ { Role = Role.User; Content = message } ] }

                try
                    let! response = llm.CompleteAsync request

                    if response.FinishReason = Some "parse_error" then
                        return
                            notReviewed sha shown "The model's response could not be parsed, so nothing was reviewed."
                    else
                        return formatVote sha shown response.Text
                with ex ->
                    return
                        notReviewed sha shown $"The model could not be reached, so nothing was reviewed: {ex.Message}"
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
