namespace Tars.Llm

/// AI Function + ix convolution: verified intelligence pipeline.
///
/// Three patterns for combining LLM reasoning with ix numerical computation:
///
/// 1. **ixPostCondition** — ix validates the LLM's output (ML-verified post-conditions)
///    LLM classifies → ix cross-checks → retry if they disagree
///
/// 2. **ixEnrich** — ix computes first, results injected into the LLM prompt
///    ix does math → LLM interprets/explains → post-conditions verify
///
/// 3. **ixValidate** — ix and LLM both produce answers, post-condition compares
///    LLM proposes → ix evaluates → threshold check → retry with ix feedback
///
/// ix is called via a generic `IxCaller` function, keeping this module
/// decoupled from the subprocess/MCP transport layer.

open System.Text.Json

// ─── ix integration types ────────────────────────────────────────────────────

/// Generic ix tool caller: takes tool name + JSON args, returns JSON result or error.
/// Implementations can use CLI subprocess, MCP client, or in-process calls.
type IxCaller = string -> string -> Async<Result<string, string>>

/// Result from an ix validation check.
type IxValidation =
    | Agrees
    | Disagrees of reason: string
    | IxUnavailable of error: string

// ─── Convolution builders ────────────────────────────────────────────────────

module AiFunctionIx =

    let private jsonOptions =
        lazy (
            let o = JsonSerializerOptions(PropertyNameCaseInsensitive = true)
            o
        )

    /// Deserialize an ix JSON result to a typed value.
    let private deserializeIx<'t> (json: string) : Result<'t, string> =
        match JsonParsing.tryParseElement json with
        | Error e -> Error $"ix JSON parse failed: {e}"
        | Ok elem ->
            try
                Ok (JsonSerializer.Deserialize<'t>(elem.GetRawText(), jsonOptions.Value))
            with ex ->
                Error $"ix deserialization failed: {ex.Message}"

    // =========================================================================
    // Pattern 1: ix as post-condition validator
    // =========================================================================

    /// Create a post-condition that calls an ix tool to validate the LLM output.
    /// The `check` function receives the LLM output and the deserialized ix result,
    /// and returns Ok () if they agree or Error "reason" if not.
    let ixPostCondition<'output, 'ixResult>
        (name: string)
        (callIx: IxCaller)
        (ixTool: string)
        (buildIxArgs: 'output -> string)
        (check: 'output -> 'ixResult -> Result<unit, string>)
        : 'output -> Result<unit, string> =

        fun output ->
            let ixArgs = buildIxArgs output
            let result =
                callIx ixTool ixArgs
                |> Async.RunSynchronously

            match result with
            | Error err -> Error $"ix '{ixTool}' failed: {err}"
            | Ok json ->
                match deserializeIx<'ixResult> json with
                | Error err -> Error $"ix '{ixTool}' result parse failed: {err}"
                | Ok ixResult -> check output ixResult

    /// Add an ix-validated post-condition to an AI Function config.
    let withIxPostCondition<'i, 'o, 'ix>
        (name: string)
        (callIx: IxCaller)
        (ixTool: string)
        (buildIxArgs: 'o -> string)
        (check: 'o -> 'ix -> Result<unit, string>)
        (cfg: AiFunctionConfig<'i, 'o>)
        : AiFunctionConfig<'i, 'o> =

        let pc = ixPostCondition<'o, 'ix> name callIx ixTool buildIxArgs check
        { cfg with PostConditions = cfg.PostConditions @ [pc] }

    // =========================================================================
    // Pattern 2: ix computes, LLM interprets (enrichment)
    // =========================================================================

    /// Pre-enrich the input by calling ix first and injecting its result into the prompt.
    /// Returns a new AiFunctionConfig where FormatInput calls ix, then formats the
    /// combined (original input + ix result) for the LLM.
    let withIxEnrichment<'i, 'o>
        (callIx: IxCaller)
        (ixTool: string)
        (buildIxArgs: 'i -> string)
        (formatEnriched: 'i -> string -> string)
        (cfg: AiFunctionConfig<'i, 'o>)
        : AiFunctionConfig<'i, 'o> =

        let enrichedFormat (input: 'i) =
            let ixArgs = buildIxArgs input
            let ixResult =
                callIx ixTool ixArgs
                |> Async.RunSynchronously

            match ixResult with
            | Ok json -> formatEnriched input json
            | Error err -> formatEnriched input $"[ix unavailable: {err}]"

        { cfg with FormatInput = enrichedFormat }

    // =========================================================================
    // Pattern 3: ix validates numerically (threshold check)
    // =========================================================================

    /// Result from an ix evaluation (e.g., ML pipeline metrics).
    type IxEvaluation<'metrics> = {
        Metrics: 'metrics
        Passed: bool
        Reason: string
    }

    /// Add a post-condition that runs ix evaluation on the LLM output
    /// and checks a numeric threshold.
    let withIxThreshold<'i, 'o>
        (name: string)
        (callIx: IxCaller)
        (ixTool: string)
        (buildIxArgs: 'o -> string)
        (extractMetric: string -> Result<float, string>)
        (threshold: float)
        (cfg: AiFunctionConfig<'i, 'o>)
        : AiFunctionConfig<'i, 'o> =

        let pc (output: 'o) =
            let ixArgs = buildIxArgs output
            let result =
                callIx ixTool ixArgs
                |> Async.RunSynchronously

            match result with
            | Error err -> Error $"ix '{ixTool}' failed: {err}"
            | Ok json ->
                match extractMetric json with
                | Error err -> Error $"ix metric extraction failed: {err}"
                | Ok value ->
                    if value >= threshold then Ok ()
                    else Error $"Condition '{name}' failed: ix metric {value:F4} < threshold {threshold:F4}"

        { cfg with PostConditions = cfg.PostConditions @ [pc] }

    // =========================================================================
    // Pattern 4: Bidirectional — LLM proposes, ix scores, feedback loop
    // =========================================================================

    /// Execute an AI Function where ix scores each attempt and the score is
    /// fed back to the LLM on retry. Combines executeAsync with ix scoring.
    let executeWithIxScoringAsync
        (llm: ILlmService)
        (callIx: IxCaller)
        (ixTool: string)
        (buildIxArgs: 'output -> string)
        (formatScore: string -> string)
        (cfg: AiFunctionConfig<'input, 'output>)
        (input: 'input)
        : Async<Result<'output * string, AiFunctionError>> =
        async {
            let mutable attempt = 0
            let mutable feedback: string list = []
            let mutable result: Result<'output * string, AiFunctionError> =
                Error (MaxAttemptsExhausted(0, "not started"))

            while attempt < cfg.MaxAttempts do
                attempt <- attempt + 1
                let req = AiFunction.buildRequest cfg input feedback

                let! resp =
                    async {
                        try
                            let! r = llm.CompleteAsync(req) |> Async.AwaitTask
                            return Ok r
                        with ex ->
                            return Error (LlmCallFailed ex.Message)
                    }

                match resp with
                | Error e ->
                    result <- Error e
                    attempt <- cfg.MaxAttempts
                | Ok llmResp ->
                    match AiFunction.deserialize<'output> llmResp.Text with
                    | Error e ->
                        let msg = match e with DeserializationFailed(_, m) -> m | _ -> "unknown"
                        feedback <- feedback @ [$"Deserialization failed: {msg}"]
                        result <- Error e
                    | Ok output ->
                        // Run standard post-conditions first
                        match AiFunction.validate cfg.PostConditions output with
                        | Error reason ->
                            feedback <- feedback @ [$"Post-condition violated: {reason}"]
                            result <- Error (PostConditionFailed(attempt, reason))
                        | Ok valid ->
                            // Now score with ix
                            let ixArgs = buildIxArgs valid
                            let! ixResult = callIx ixTool ixArgs
                            match ixResult with
                            | Error err ->
                                // ix unavailable — accept the output anyway
                                result <- Ok (valid, $"ix unavailable: {err}")
                                attempt <- cfg.MaxAttempts
                            | Ok ixJson ->
                                let scoreMsg = formatScore ixJson
                                // If formatScore returns empty, treat as pass
                                if scoreMsg = "" || scoreMsg.StartsWith("PASS") then
                                    result <- Ok (valid, ixJson)
                                    attempt <- cfg.MaxAttempts
                                else
                                    feedback <- feedback @ [$"ix scoring feedback: {scoreMsg}"]
                                    result <- Error (PostConditionFailed(attempt, scoreMsg))

            match result with
            | Error (PostConditionFailed(_, reason)) ->
                return Error (MaxAttemptsExhausted(cfg.MaxAttempts, reason))
            | other -> return other
        }
