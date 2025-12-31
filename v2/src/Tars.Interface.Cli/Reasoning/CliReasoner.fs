namespace Tars.Interface.Cli.Reasoning

open System
open System.IO
open System.Text.Json
open Serilog
open Tars.Core.WorkflowOfThought
open Tars.Llm

/// Settings for the CLI reasoner
type ReasonerSettings =
    { Model: string option
      ModelHint: string option
      Temperature: float option
      MaxTokens: int option
      Deterministic: bool
      Seed: int option }
    
    static member Default =
        { Model = None
          ModelHint = Some "reasoning"
          Temperature = None
          MaxTokens = None
          Deterministic = false
          Seed = None }

module private Journal =
    let ensureDir (path: string) =
        Directory.CreateDirectory(path) |> ignore

    let writeText (path: string) (text: string) =
        File.WriteAllText(path, text)

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
            |> List.map (fun (k,v) -> $"- {k}: {v}")
            |> String.concat "\n"

        let varsStringOnly =
            ctx.Vars
            |> Map.toList
            |> List.choose (fun (k,v) ->
                match v with
                | :? string as s -> Some (k, truncate maxVarLen s)
                | _ -> None)
            |> List.sortBy fst
            |> List.map (fun (k,v) -> $"- {k}: {v}")
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

    let mkRequest (prompt: string) : LlmRequest =
        let temp =
            if settings.Deterministic then Some 0.0 else settings.Temperature
        let seed =
            if settings.Deterministic then settings.Seed |> Option.orElse (Some 42) else settings.Seed

        { LlmRequest.Default with
            ModelHint = settings.ModelHint
            Model = settings.Model
            SystemPrompt = Some "You are executing a Workflow-of-Thought reason step. Be precise and concise."
            MaxTokens = settings.MaxTokens
            Temperature = temp
            Seed = seed
            Messages = [ { Role = Role.User; Content = prompt } ] }

    interface IReasoner with
        member _.Reason(stepId, ctx, goalOpt, instrOpt) = async {
            try
                Journal.ensureDir reasonDir

                let prompt = Prompt.build ctx goalOpt instrOpt

                let promptPath = Path.Combine(reasonDir, $"{stepId}.prompt.txt")
                let respPath   = Path.Combine(reasonDir, $"{stepId}.response.txt")
                let metaPath   = Path.Combine(reasonDir, $"{stepId}.meta.json")

                // Write prompt BEFORE call (so we have it even if call fails)
                Journal.writeText promptPath prompt

                let req = mkRequest prompt

                logger.Information("Reason step {StepId}: calling LLM...", stepId)

                // Capture GPU metrics around LLM call
                let! (resp, gpuMetrics) = 
                    GpuMonitor.withMetrics logger (async {
                        return! llm.CompleteAsync(req) |> Async.AwaitTask
                    })

                // Write response AFTER call
                Journal.writeText respPath resp.Text

                // Write meta with GPU metrics AFTER call
                Journal.writeJson metaPath
                    {| StepId = stepId
                       Model = req.Model
                       ModelHint = req.ModelHint
                       Temperature = req.Temperature
                       MaxTokens = req.MaxTokens
                       Seed = req.Seed
                       Deterministic = settings.Deterministic
                       Gpu = {|
                           BeforeVramMiB = gpuMetrics.Before.VramUsedMiB
                           AfterVramMiB = gpuMetrics.After.VramUsedMiB
                           VramDeltaMiB = gpuMetrics.VramDeltaMiB
                           GpuUtilPercent = gpuMetrics.After.GpuUtilPercent
                           ModelFullyOnGpu = gpuMetrics.ModelFullyOnGpu
                           InferenceDurationMs = gpuMetrics.DurationMs
                       |}
                    |}

                // Log GPU status
                match gpuMetrics.ModelFullyOnGpu with
                | Some true -> 
                    logger.Information("Reason step {StepId}: completed (GPU 100%, {VramMiB} MiB, {DurationMs}ms)", 
                        stepId, gpuMetrics.After.VramUsedMiB |> Option.defaultValue 0, gpuMetrics.DurationMs)
                | Some false ->
                    logger.Warning("Reason step {StepId}: completed with CPU fallback ({DurationMs}ms)", 
                        stepId, gpuMetrics.DurationMs)
                | None ->
                    logger.Information("Reason step {StepId}: completed ({DurationMs}ms)", 
                        stepId, gpuMetrics.DurationMs)

                return Result.Ok resp.Text
            with ex ->
                logger.Error(ex, "Reason step {StepId} failed", stepId)
                return Result.Error ex.Message
        }
