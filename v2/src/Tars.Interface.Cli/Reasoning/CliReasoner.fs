namespace Tars.Interface.Cli.Reasoning

open System.IO
open System.Text.Json
open Serilog
open Spectre.Console
open Tars.Core.WorkflowOfThought
open Tars.Llm

/// Settings for the CLI reasoner
type ReasonerSettings =
    { Model: string option
      ModelHint: string option
      Temperature: float option
      MaxTokens: int option
      Deterministic: bool
      Seed: int option
      ContextWindow: int option
      AgentHint: string option // Phase 17.2: Agent role hint
      GrammarConstraint: ResponseFormat option } // Constrained decoding format

    static member Default =
        { Model = None
          ModelHint = Some "reasoning"
          Temperature = None
          MaxTokens = None
          Deterministic = false
          Seed = None
          ContextWindow = None
          AgentHint = None
          GrammarConstraint = None }

module private Journal =
    let ensureDir (path: string) =
        Directory.CreateDirectory(path) |> ignore

    let writeText (path: string) (text: string) = File.WriteAllText(path, text)

    let writeJson<'T> (path: string) (value: 'T) =
        let opts = JsonSerializerOptions(WriteIndented = true)
        opts.Converters.Add(System.Text.Json.Serialization.JsonFSharpConverter())
        let json = JsonSerializer.Serialize(value, opts)
        File.WriteAllText(path, json)

module private Prompt =
    let private truncate (maxLen: int) (s: string) =
        if isNull s then ""
        elif s.Length <= maxLen then s
        else s.Substring(0, maxLen) + "\n...[truncated]"

    let build (ctx: ExecContext) (goal: string option) (instruction: string option) : string =
        let maxVarLen = 20000

        let inputs =
            ctx.Inputs
            |> Map.toList
            |> List.sortBy fst
            |> List.map (fun (k, v) -> $"- {k}: {v}")
            |> String.concat "\n"

        let varsStringOnly =
            ctx.Vars
            |> Map.toList
            |> List.choose (fun (k, v) ->
                match v with
                | :? string as s -> Some(k, truncate maxVarLen s)
                | _ -> None)
            |> List.sortBy fst
            |> List.map (fun (k, v) -> $"- {k}: {v}")
            |> String.concat "\n"

        $"""# WoT Reason Step

Goal: {defaultArg goal "<none>"}
Instruction: {defaultArg instruction "<none>"}

## Inputs
{inputs}

## Vars (string only)
{varsStringOnly}
"""

/// CLI Reasoner that wraps ILlmService and implements IReasoner with journaling
type CliReasoner(llm: ILlmService, runDir: string, settings: ReasonerSettings, logger: ILogger) =

    let reasonDir = Path.Combine(runDir, "reason")

    let mkRequest (prompt: string) (agent: string option) : LlmRequest =
        // Resolve agent configuration: per-step choice first, then global reasoner setting, then default
        let role = agent |> Option.orElse settings.AgentHint |> Option.defaultValue "Default"
        let config = AgentRegistry.getOrDefault role

        let temp =
            if settings.Deterministic then
                Some 0.0
            else
                // Use agent-specific temperature if no explicit setting
                settings.Temperature |> Option.orElse config.Temperature

        let seed =
            if settings.Deterministic then
                settings.Seed |> Option.orElse (Some 42)
            else
                settings.Seed
        
        // Use agent-specific model hint if no explicit setting
        let modelHint = 
            settings.ModelHint |> Option.orElse config.ModelHint

        { LlmRequest.Default with
            ModelHint = modelHint
            Model = settings.Model
            SystemPrompt = Some config.SystemPrompt // Agent-specific system prompt!
            MaxTokens = settings.MaxTokens
            Temperature = temp
            Seed = seed
            ContextWindow = settings.ContextWindow
            ResponseFormat = settings.GrammarConstraint
            Messages = [ { Role = Role.User; Content = prompt } ] }


    interface IReasoner with
        member _.Reason(stepId, ctx, goalOpt, instrOpt, agentOpt) : Async<Result<ReasoningResult, string>> =
            async {
                try
                    Journal.ensureDir reasonDir

                    let prompt = Prompt.build ctx goalOpt instrOpt

                    let promptPath = Path.Combine(reasonDir, $"{stepId}.prompt.txt")
                    let respPath = Path.Combine(reasonDir, $"{stepId}.response.txt")
                    let metaPath = Path.Combine(reasonDir, $"{stepId}.meta.json")

                    // Write prompt BEFORE call (so we have it even if call fails)
                    Journal.writeText promptPath prompt

                    let req = mkRequest prompt agentOpt

                    // Identify backend information
                    let! routed = llm.RouteAsync req |> Async.AwaitTask

                    let backendStr =
                        match routed.Backend with
                        | Ollama m -> $"Ollama ({m})"
                        | LlamaCpp(m, _) -> $"LlamaCpp ({m})"
                        | OpenAI m -> $"OpenAI ({m})"
                        | GoogleGemini m -> $"Gemini ({m})"
                        | Anthropic m -> $"Anthropic ({m})"
                        | DockerModelRunner m -> $"Docker ({m})"
                        | Vllm m -> $"vLLM ({m})"
                        | LlamaSharp p -> $"LlamaSharp ({System.IO.Path.GetFileName p})"


                    let ctxStr =
                        match settings.ContextWindow with
                        | Some c -> $", CW: {c}"
                        | None -> ""

                    logger.Information(
                        "Reason step {StepId}: calling LLM ({Backend}{Cw})...",
                        stepId,
                        backendStr,
                        ctxStr
                    )

                    // Capture GPU metrics around LLM call
                    // Capture GPU metrics around LLM call with streaming
                    let! (resp, gpuMetrics) =
                        GpuMonitor.withMetrics
                            logger
                            (async {
                                let onToken (t: string) = AnsiConsole.Markup(Markup.Escape(t))
                                let! res = llm.CompleteStreamAsync(req, onToken) |> Async.AwaitTask
                                AnsiConsole.WriteLine() // Newline after stream
                                return res
                            })

                    // Write response AFTER call
                    Journal.writeText respPath resp.Text

                    // Write meta with GPU metrics AFTER call
                    Journal.writeJson
                        metaPath
                        {| StepId = stepId
                           Model = req.Model
                           ModelHint = req.ModelHint
                           Temperature = req.Temperature
                           MaxTokens = req.MaxTokens
                           Seed = req.Seed
                           Deterministic = settings.Deterministic
                           Gpu =
                            {| BeforeVramMiB = gpuMetrics.Before.VramUsedMiB
                               AfterVramMiB = gpuMetrics.After.VramUsedMiB
                               VramDeltaMiB = gpuMetrics.VramDeltaMiB
                               GpuUtilPercent = gpuMetrics.After.GpuUtilPercent
                               ModelFullyOnGpu = gpuMetrics.ModelFullyOnGpu
                               InferenceDurationMs = gpuMetrics.DurationMs |} |}

                    // Calculate speed
                    let speedStr =
                        match resp.Usage with
                        | Some u when gpuMetrics.DurationMs > 0L ->
                            let spd = float u.CompletionTokens / (float gpuMetrics.DurationMs / 1000.0)
                            $", %.1f{spd} tok/s"
                        | _ -> ""

                    // Log GPU status
                    match gpuMetrics.ModelFullyOnGpu with
                    | Some true ->
                        logger.Information(
                            "Reason step {StepId}: completed (GPU 100%, {VramMiB} MiB, {DurationMs}ms{Speed})",
                            stepId,
                            gpuMetrics.After.VramUsedMiB |> Option.defaultValue 0,
                            gpuMetrics.DurationMs,
                            speedStr
                        )
                    | Some false ->
                        logger.Warning(
                            "Reason step {StepId}: completed with CPU fallback ({DurationMs}ms{Speed})",
                            stepId,
                            gpuMetrics.DurationMs,
                            speedStr
                        )
                    | None ->
                        logger.Information(
                            "Reason step {StepId}: completed ({DurationMs}ms{Speed})",
                            stepId,
                            gpuMetrics.DurationMs,
                            speedStr
                        )

                    let usage =
                        resp.Usage
                        |> Option.map (fun u ->
                            { Prompt = u.PromptTokens
                              Completion = u.CompletionTokens })

                    let res: ReasoningResult = { Content = resp.Text; Usage = usage }
                    return Microsoft.FSharp.Core.Result.Ok res
                with ex ->
                    logger.Error(ex, "Reason step {StepId} failed", stepId)
                    return Microsoft.FSharp.Core.Result.Error ex.Message
            }
