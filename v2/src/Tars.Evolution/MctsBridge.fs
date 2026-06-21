namespace Tars.Evolution

/// Bridge to ix's Rust MCTS implementation.
/// Falls back to the built-in F# MctsSolver when ix is unavailable.
/// Follows the same pattern as MachinBridge + FallbackGA.
module MctsBridge =

    open System
    open System.Text.Json
    open System.Text.RegularExpressions
    open MctsTypes

    /// Result from ix's `grammar.search` skill.
    type MctsExternalResult = {
        /// Ordered template indices chosen along the best derivation.
        NodeIndices: int list
        Reward: float
        Iterations: int
    }

    /// Parse ix `grammar.search` JSON output:
    ///   { "best_derivation": [{"nonterminal": "root", "alternative": "node_2"}, ...],
    ///     "reward": 0.75, "iterations": 500 }
    /// We harvest the ordered `node_<i>` references from the derivation; those
    /// indices map directly back onto the template pool that produced the EBNF.
    let parseMctsOutput (output: string) : MctsExternalResult =
        let reward, iterations =
            try
                use doc = JsonDocument.Parse(output)
                let root = doc.RootElement
                let r =
                    match root.TryGetProperty("reward") with
                    | true, v -> v.GetDouble()
                    | _ -> 0.0
                let it =
                    match root.TryGetProperty("iterations") with
                    | true, v -> v.GetInt32()
                    | _ -> 0
                r, it
            with _ -> 0.0, 0
        // node_<i> tokens, in derivation order. Robust to small shape drift in
        // how ix labels nonterminals vs. alternatives.
        let indices =
            Regex.Matches(output, "node_(\\d+)")
            |> Seq.map (fun m -> int m.Groups.[1].Value)
            |> Seq.toList
        { NodeIndices = indices; Reward = reward; Iterations = iterations }

    /// Serialize WoT templates to an EBNF grammar for ix's grammar-guided MCTS.
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

    /// Map the ordered template indices from an ix derivation back onto WotActions.
    /// Honors the F# derivation rules: no duplicate node ids, capped at maxNodes,
    /// terminated with Complete.
    let private indicesToActions
        (templates: Tars.DSL.Wot.DslNode list)
        (maxNodes: int)
        (indices: int list)
        : WotMctsState.WotAction list =
        let pool = List.toArray templates
        let seen = System.Collections.Generic.HashSet<Tars.DSL.Wot.DslId>()
        let actions = ResizeArray<WotMctsState.WotAction>()
        for i in indices do
            if i >= 0 && i < pool.Length && actions.Count < maxNodes then
                let node = pool.[i]
                if seen.Add(node.Id) then
                    actions.Add(WotMctsState.WotAction.AddNode node)
        if actions.Count > 0 then
            actions.Add(WotMctsState.WotAction.Complete)
        List.ofSeq actions

    /// Try grammar-guided MCTS via ix's `grammar.search` skill.
    let private tryIxGrammarSearch
        (config: MachinBridge.MachinConfig)
        (mctsConfig: MctsConfig)
        (templates: Tars.DSL.Wot.DslNode list)
        (maxNodes: int)
        : Result<WotMctsState.WotAction list, string> =

        if not (MachinBridge.isAvailable config) then
            Error "ix not available"
        else
            try
                let grammarText = templatesToEbnf templates
                let input =
                    JsonSerializer.Serialize(
                        {| grammar_ebnf = grammarText
                           max_iterations = mctsConfig.MaxIterations
                           exploration = mctsConfig.ExplorationConstant
                           max_depth = mctsConfig.MaxRolloutDepth |})
                let result =
                    (MachinBridge.runSkillJson config "grammar.search" input)
                        .GetAwaiter()
                        .GetResult()
                match result with
                | Error e -> Error e
                | Ok json ->
                    let ext = parseMctsOutput json
                    let actions = indicesToActions templates maxNodes ext.NodeIndices
                    if List.isEmpty actions then
                        Error "ix grammar.search returned no usable derivation"
                    else
                        Ok actions
            with ex ->
                Error $"ix grammar.search failed: {ex.Message}"

    /// Run MCTS search for WoT derivation, using ix when available.
    /// Falls back to built-in F# MCTS solver.
    let searchWotDerivation
        (machinConfig: MachinBridge.MachinConfig option)
        (mctsConfig: MctsConfig)
        (meta: Tars.DSL.Wot.DslMeta)
        (templates: Tars.DSL.Wot.DslNode list)
        (maxNodes: int)
        : WotMctsState.WotAction list * bool = // (actions, usedIx)

        let fallback () =
            let result = WotMctsState.searchDerivation mctsConfig meta templates maxNodes
            result.BestActions

        // Try Rust grammar-guided MCTS via ix when a config is provided.
        match machinConfig with
        | Some config ->
            match tryIxGrammarSearch config mctsConfig templates maxNodes with
            | Ok actions -> (actions, true)
            | Error reason ->
                if not (isNull (Environment.GetEnvironmentVariable "TARS_IX_DEBUG")) then
                    eprintfn "[ix-bridge] falling back to F# MCTS: %s" reason
                (fallback (), false)
        | None ->
            (fallback (), false)

    /// Quick search with default config
    let quickSearch
        (meta: Tars.DSL.Wot.DslMeta)
        (templates: Tars.DSL.Wot.DslNode list)
        (maxNodes: int)
        : WotMctsState.WotAction list =
        let config = { MctsTypes.defaultMctsConfig with MaxIterations = 200; MaxRolloutDepth = 20 }
        let actions, _ = searchWotDerivation None config meta templates maxNodes
        actions
