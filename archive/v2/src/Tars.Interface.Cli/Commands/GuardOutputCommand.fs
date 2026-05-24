module Tars.Interface.Cli.Commands.GuardOutputCommand

open System
open System.IO
open Microsoft.Extensions.Configuration
open Tars.Core

let run (config: IConfiguration) (path: string) (fieldsArg: string option) (requireCitations: bool) (allowExtra: bool) =
    if not (File.Exists path) then
        printfn $"File not found: %s{path}"
        1
    else
        let text = File.ReadAllText path
        let fields =
            fieldsArg
            |> Option.map (fun s ->
                s.Split(',', StringSplitOptions.RemoveEmptyEntries)
                |> Array.map (fun x -> x.Trim())
                |> Array.toList)

        // Build analyzer from env/secrets
        let ollama = config["OLLAMA_BASE_URL"] |> Option.ofObj
        let model = config["DEFAULT_OLLAMA_MODEL"] |> Option.ofObj

        match ollama, model with
        | None, _ ->
            printfn "Missing OLLAMA_BASE_URL (set via user secrets or env)."
            1
        | _, None ->
            printfn "Missing DEFAULT_OLLAMA_MODEL (set via user secrets or env)."
            1
        | Some ollamaUrl, Some modelName ->
            let analyzer = Tars.Cortex.OutputGuardAnalyzerFactory.createOllamaAnalyzer (Uri ollamaUrl) modelName
            let guard = OutputGuard.withAnalyzer OutputGuard.defaultGuard (Some analyzer)

            let input: GuardInput =
                { ResponseText = text
                  Grammar = None
                  ExpectedJsonFields = fields
                  RequireCitations = requireCitations
                  Citations = None
                  AllowExtraFields = allowExtra
                  Metadata = Map.empty }

            let result = guard.Evaluate input |> Async.RunSynchronously

            printfn $"Risk: %.2f{result.Risk}"
            printfn $"Action: %A{result.Action}"
            if result.Messages.Length > 0 then
                printfn "Messages:"
                result.Messages |> List.iter (fun m -> printfn $"  - %s{m}")

            match result.Action with
            | GuardAction.Accept -> 0
            | GuardAction.AskForEvidence _ -> 1
            | GuardAction.RetryWithHint _ -> 2
            | GuardAction.Fallback _ -> 3
            | GuardAction.Reject _ -> 4
