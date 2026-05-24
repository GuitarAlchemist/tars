namespace TarsEngine.FSharp.SelfImprovement

open System
open System.IO
open System.Text.Json
open TarsEngine.FSharp.Core.Services.ReasoningTrace

module ReasoningFeedback =

    let private verdictToString verdict =
        match verdict with
        | Some CriticVerdict.Accept -> "accept"
        | Some (CriticVerdict.NeedsReview message) -> $"needs_review:{message}"
        | Some (CriticVerdict.Reject message) -> $"reject:{message}"
        | None -> "none"

    /// Creates a feedback sink that appends reasoning traces and verdicts to a file as JSON lines.
    let createFileSink (path: string) =
        let fullPath = Path.GetFullPath(path)
        let directory = Path.GetDirectoryName(fullPath)
        if not (String.IsNullOrWhiteSpace(directory)) then
            Directory.CreateDirectory(directory) |> ignore

        fun (traces: ReasoningTrace list) (verdict: CriticVerdict option) ->
            let payload =
                {| timestamp = DateTime.UtcNow
                   verdict = verdictToString verdict
                   traces = traces |}

            let json = JsonSerializer.Serialize(payload)
            File.AppendAllText(fullPath, json + Environment.NewLine)
