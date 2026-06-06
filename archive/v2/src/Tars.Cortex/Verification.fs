namespace Tars.Cortex

open System
open System.Text.RegularExpressions
open Tars.Cortex.WoTTypes

/// <summary>
/// Implements structured verification logic (Phase 15.1 Verifier++).
/// </summary>
module Verification =

    /// <summary>
    /// Verify content against a structured operation.
    /// </summary>
    let verify
        (content: string)
        (op: VerificationOp)
        (executeTool: string -> Map<string, obj> -> Async<Result<string, string>>)
        : Async<Result<bool, string>> =
        async {
            try
                match op with
                | Contains substring ->
                    let passed = content.Contains(substring, StringComparison.OrdinalIgnoreCase)
                    return Result.Ok passed

                | Regex pattern ->
                    let passed = Regex.IsMatch(content, pattern, RegexOptions.IgnoreCase)
                    return Result.Ok passed

                | JsonPath path ->
                    // Heuristic check for now. Will be enhanced with real JSON parsing later.
                    let passed = content.Contains(path) || content.Contains($"\"{path}\"")
                    return Result.Ok passed

                | Schema schema ->
                    // Placeholder for Schema validation
                    return Result.Ok true

                | ToolCheck(toolName, args) ->
                    let! result = executeTool toolName args

                    match result with
                    | Result.Ok _ -> return Result.Ok true
                    | Result.Error err -> return Result.Ok false

                | CustomOp name ->
                    // Placeholder for custom ops
                    return Result.Ok true
            with ex ->
                return Result.Error ex.Message
        }
