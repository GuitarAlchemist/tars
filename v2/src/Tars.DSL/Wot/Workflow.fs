namespace Tars.DSL.Wot

/// Why a `.wot.trsx` file could not become a runnable plan.
type WorkflowLoadError =
    | ParseFailed of ParseError list
    | CompileFailed of CompileError list

/// The one way from a `.wot.trsx` file to a compiled plan (#44), so callers stop
/// chaining WotParser and WotCompiler and reporting their errors by hand.
module Workflow =

    /// Human-readable lines for a load failure, one per underlying error.
    let describe (error: WorkflowLoadError) : string list =
        match error with
        | ParseFailed errs -> errs |> List.map (fun e -> $"Parse error (line {e.Line}): {e.Message}")
        | CompileFailed errs ->
            errs
            |> List.map (function
                | InvalidGraph msg -> $"Compile error: {msg}"
                | InvalidNode(nodeId, msg) -> $"Compile error in node '{nodeId}': {msg}")

    let private compile (parsed: Result<DslWorkflow, ParseError list>) =
        match parsed with
        | Error errs -> Error(ParseFailed errs)
        | Ok workflow -> WotCompiler.compileWorkflowToPlanParsed workflow |> Result.mapError CompileFailed

    /// Parse and compile workflow source lines.
    let loadLines (lines: string list) : Result<Plan<Parsed>, WorkflowLoadError> = compile (WotParser.parseLines lines)

    /// Parse and compile a `.wot.trsx` file. A missing file is a parse error on line 0.
    let load (path: string) : Result<Plan<Parsed>, WorkflowLoadError> = compile (WotParser.parseFile path)
