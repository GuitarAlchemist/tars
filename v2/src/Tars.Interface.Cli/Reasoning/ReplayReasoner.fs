namespace Tars.Interface.Cli.Reasoning

open System.IO
open Serilog
open Tars.Core.WorkflowOfThought

/// Reasoner that replays previously journaled responses for CI without GPU
type ReplayReasoner(replayRunDir: string, logger: ILogger) =

    let reasonDir = Path.Combine(replayRunDir, "reason")

    interface IReasoner with
        member _.Reason(stepId, _ctx, _goalOpt, _instrOpt) = async {
            try
                let responsePath = Path.Combine(reasonDir, $"{stepId}.response.txt")
                
                if not (File.Exists responsePath) then
                    logger.Warning("Replay: response file not found for step {StepId} at {Path}", stepId, responsePath)
                    return Result.Error $"Replay file not found: {responsePath}"
                else
                    let response = File.ReadAllText(responsePath)
                    logger.Information("Replay step {StepId}: loaded {Chars} chars from journal", stepId, response.Length)
                    return Result.Ok response
            with ex ->
                logger.Error(ex, "Replay step {StepId} failed", stepId)
                return Result.Error ex.Message
        }
