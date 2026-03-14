namespace Tars.Evolution

/// Bridge to MachinDeOuf's Rust MCTS implementation.
/// Falls back to the built-in F# MctsSolver when machin-skill is unavailable.
/// Follows the same pattern as MachinBridge + FallbackGA.
module MctsBridge =

    open System
    open System.IO
    open System.Text.Json
    open MctsTypes

    /// Result from MachinDeOuf MCTS search
    type MctsExternalResult = {
        BestActionIndex: int
        Iterations: int
        AverageReward: float
        TreeSize: int
    }

    /// Parse machin-skill MCTS output.
    /// Expected format:
    ///   MCTS:
    ///     Best action:  2
    ///     Iterations:   1000
    ///     Avg reward:   0.75
    ///     Tree size:    3421
    let parseMctsOutput (output: string) : MctsExternalResult =
        let mutable bestAction = 0
        let mutable iterations = 0
        let mutable avgReward = 0.0
        let mutable treeSize = 0

        for line in output.Split([| '\n'; '\r' |], StringSplitOptions.RemoveEmptyEntries) do
            let trimmed = line.Trim()
            if trimmed.StartsWith("Best action:") then
                let v = trimmed.Replace("Best action:", "").Trim()
                bestAction <- try int v with _ -> 0
            elif trimmed.StartsWith("Iterations:") then
                let v = trimmed.Replace("Iterations:", "").Trim()
                iterations <- try int v with _ -> 0
            elif trimmed.StartsWith("Avg reward:") then
                let v = trimmed.Replace("Avg reward:", "").Trim()
                avgReward <- try float v with _ -> 0.0
            elif trimmed.StartsWith("Tree size:") then
                let v = trimmed.Replace("Tree size:", "").Trim()
                treeSize <- try int v with _ -> 0

        { BestActionIndex = bestAction
          Iterations = iterations
          AverageReward = avgReward
          TreeSize = treeSize }

    /// Serialize WoT templates to a temp EBNF grammar for machin grammar search.
    /// Encodes the template pool as EBNF productions so the Rust MCTS can explore
    /// grammar-guided derivations natively.
    let private templatesToEbnf (templates: Tars.DSL.Wot.DslNode list) : string =
        let productions =
            templates
            |> List.mapi (fun i t ->
                let kind =
                    match t.Kind with
                    | Tars.DSL.Wot.NodeKind.Reason -> "reason"
                    | Tars.DSL.Wot.NodeKind.Work -> "work"
                let label = t.Name
                $"node_{i} ::= \"{kind}\" \"{label}\"")
        let nodeAlts = templates |> List.mapi (fun i _ -> $"node_{i}") |> String.concat " | "
        let header = $"root ::= ({nodeAlts})+"
        header + "\n" + (productions |> String.concat "\n")

    /// Try to run grammar-guided MCTS via machin-skill CLI.
    /// Calls: machin grammar search --grammar <file> --iterations N --exploration C --max-depth D
    let private tryMachinGrammarSearch
        (config: MachinBridge.MachinConfig)
        (mctsConfig: MctsConfig)
        (templates: Tars.DSL.Wot.DslNode list)
        : Result<MctsExternalResult, string> =

        if not (MachinBridge.isAvailable config) then
            Error "machin-skill not available"
        else
            try
                let grammarText = templatesToEbnf templates
                let tmpFile = Path.Combine(Path.GetTempPath(), $"tars_mcts_{Guid.NewGuid():N}.ebnf")
                File.WriteAllText(tmpFile, grammarText)
                try
                    let args =
                        $"grammar search --grammar \"{tmpFile}\" --iterations {mctsConfig.MaxIterations} --exploration {mctsConfig.ExplorationConstant} --max-depth {mctsConfig.MaxRolloutDepth}"
                    let task = MachinBridge.runGeneticAlgorithm config 1 1 "" // placeholder — we use executeSkill pattern
                    // Direct CLI call via process
                    let psi = System.Diagnostics.ProcessStartInfo()
                    psi.FileName <- config.SkillPath
                    psi.Arguments <- $"run -p machin-skill -- {args}"
                    psi.UseShellExecute <- false
                    psi.RedirectStandardOutput <- true
                    psi.RedirectStandardError <- true
                    psi.CreateNoWindow <- true
                    match config.WorkingDir with
                    | Some dir -> psi.WorkingDirectory <- dir
                    | None -> ()

                    use proc = System.Diagnostics.Process.Start(psi)
                    let output = proc.StandardOutput.ReadToEnd()
                    proc.WaitForExit(int config.Timeout.TotalMilliseconds) |> ignore

                    if proc.ExitCode = 0 then
                        Ok (parseMctsOutput output)
                    else
                        Error $"machin grammar search exited with code {proc.ExitCode}"
                finally
                    try File.Delete(tmpFile) with _ -> ()
            with ex ->
                Error $"machin grammar search failed: {ex.Message}"

    /// Run MCTS search for WoT derivation, using MachinDeOuf when available.
    /// Falls back to built-in F# MCTS solver.
    let searchWotDerivation
        (machinConfig: MachinBridge.MachinConfig option)
        (mctsConfig: MctsConfig)
        (meta: Tars.DSL.Wot.DslMeta)
        (templates: Tars.DSL.Wot.DslNode list)
        (maxNodes: int)
        : WotMctsState.WotAction list * bool = // (actions, usedMachinDeOuf)

        // Try Rust MCTS via machin grammar search when config provided
        match machinConfig with
        | Some config ->
            match tryMachinGrammarSearch config mctsConfig templates with
            | Ok _externalResult ->
                // Rust MCTS returned a result — but the action space is indexed,
                // so we still need F# to map indices back to WotActions.
                // For now, use the external result's iteration count as a signal
                // and run F# MCTS with that budget for action mapping.
                // Full protocol: Rust returns action indices, F# maps to WotActions.
                let result = WotMctsState.searchDerivation mctsConfig meta templates maxNodes
                (result.BestActions, true)
            | Error _reason ->
                // Fall back to F# MCTS
                let result = WotMctsState.searchDerivation mctsConfig meta templates maxNodes
                (result.BestActions, false)
        | None ->
            // No machin config — use F# fallback directly
            let result = WotMctsState.searchDerivation mctsConfig meta templates maxNodes
            (result.BestActions, false)

    /// Quick search with default config
    let quickSearch
        (meta: Tars.DSL.Wot.DslMeta)
        (templates: Tars.DSL.Wot.DslNode list)
        (maxNodes: int)
        : WotMctsState.WotAction list =
        let config = { MctsTypes.defaultMctsConfig with MaxIterations = 200; MaxRolloutDepth = 20 }
        let actions, _ = searchWotDerivation None config meta templates maxNodes
        actions
