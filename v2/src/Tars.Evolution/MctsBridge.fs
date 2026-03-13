namespace Tars.Evolution

/// Bridge to MachinDeOuf's Rust MCTS implementation.
/// Falls back to the built-in F# MctsSolver when machin-skill is unavailable.
/// Follows the same pattern as MachinBridge + FallbackGA.
module MctsBridge =

    open System
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

    /// Run MCTS search for WoT derivation, using MachinDeOuf when available.
    /// Falls back to built-in F# MCTS solver.
    let searchWotDerivation
        (machinConfig: MachinBridge.MachinConfig option)
        (mctsConfig: MctsConfig)
        (meta: Tars.DSL.Wot.DslMeta)
        (templates: Tars.DSL.Wot.DslNode list)
        (maxNodes: int)
        : WotMctsState.WotAction list * bool = // (actions, usedMachinDeOuf)

        // For now, always use F# fallback — MachinDeOuf MCTS bridge requires
        // the --stdin-fitness protocol (from the Ralph prompt) to be implemented first.
        // When machin-skill supports MCTS with custom state/action spaces via stdin,
        // this will call out to Rust for performance on large search spaces.
        let _useMachin =
            match machinConfig with
            | Some config -> MachinBridge.isAvailable config
            | None -> false

        // F# fallback: use built-in MctsSolver
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
