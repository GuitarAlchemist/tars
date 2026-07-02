namespace Tars.Evolution

open System
open System.IO
open System.Globalization
open System.Text
open System.Text.Json
open Tars.Core
open MctsTypes

/// Parallel grammar-config sweep over ix's DAG pipeline executor (ADR 0001).
///
/// Emits an `ix.yaml` mesh of N independent `grammar.search` stages — one per
/// sweep config — so ix-pipeline fans them out in parallel, then ranks the
/// results by reward TARS-side and returns the winning WoT derivation. The mesh
/// is a *parallel accelerator* of what `MctsBridge` does single-shot: when ix's
/// pipeline surface is unavailable, it degrades to running the same configs
/// serially through `MctsBridge.searchWotDerivation` (never hard-fails).
///
/// Node vocabulary is ix-registry compute only (`grammar.search`); governance +
/// provenance are inherited from `ix pipeline run` (ConstitutionGate + ix.lock).
module GrammarMeshBridge =

    /// One point in the sweep space — the knobs that vary per mesh node.
    type SweepConfig = { Exploration: float; MaxDepth: int }

    /// A ranked outcome from one mesh stage.
    type MeshStageResult =
        { StageId: string
          Reward: float
          Iterations: int
          NodeIndices: int list }

    /// Outcome of a sweep, including whether the parallel mesh path was used.
    type MeshOutcome =
        { Best: WotMctsState.WotAction list
          Ranked: MeshStageResult list
          UsedMesh: bool }

    /// Build the swept config grid: the cartesian product of explorations × depths.
    let buildSweep (explorations: float list) (depths: int list) : SweepConfig list =
        [ for e in explorations do
            for d in depths -> { Exploration = e; MaxDepth = d } ]

    /// Emit an `ix.yaml` mesh: one dep-free `grammar.search` stage per config, so
    /// the DAG executor runs them concurrently. The shared EBNF corpus is
    /// param-bound (`{param:"grammar"}`, supplied via `--param grammar=@file`);
    /// the per-stage knobs are inline. (ADR 0001 D2/D5)
    let buildMeshYaml (configs: SweepConfig list) : string =
        let f (x: float) = x.ToString(CultureInfo.InvariantCulture)
        let sb = StringBuilder()
        sb.AppendLine("version: \"1\"") |> ignore
        sb.AppendLine("params:") |> ignore
        sb.AppendLine("  grammar: null") |> ignore
        sb.AppendLine("stages:") |> ignore
        configs
        |> List.iteri (fun i c ->
            sb.AppendLine(sprintf "  s%d:" i) |> ignore
            sb.AppendLine("    skill: grammar.search") |> ignore
            sb.AppendLine("    args:") |> ignore
            sb.AppendLine("      grammar_ebnf: { param: \"grammar\" }") |> ignore
            sb.AppendLine(sprintf "      max_depth: %d" c.MaxDepth) |> ignore
            sb.AppendLine(sprintf "      exploration: %s" (f c.Exploration)) |> ignore)
        sb.ToString()

    /// Parse pipeline stdout `{ "stages": { "<id>": { "output": {...} } } }`,
    /// extracting reward + node indices per stage via MctsBridge's result parser.
    let parseMeshOutput (json: string) : MeshStageResult list =
        try
            use doc = JsonDocument.Parse(json)
            match doc.RootElement.TryGetProperty("stages") with
            | true, stages ->
                [ for s in stages.EnumerateObject() ->
                    let outputJson =
                        match s.Value.TryGetProperty("output") with
                        | true, o -> o.GetRawText()
                        | _ -> s.Value.GetRawText()
                    let ext = MctsBridge.parseMctsOutput outputJson
                    { StageId = s.Name
                      Reward = ext.Reward
                      Iterations = ext.Iterations
                      NodeIndices = ext.NodeIndices } ]
            | _ -> []
        with _ -> []

    let private toIxConfig (c: MachinBridge.MachinConfig) : IxSkill.Config =
        { CargoPath = c.SkillPath
          Timeout = c.Timeout
          RepoDir = c.WorkingDir }

    /// Run the grammar sweep. Uses the parallel ix mesh when the pipeline surface
    /// is available, else degrades to serial `MctsBridge.searchWotDerivation`.
    /// Winners route into promotion through the existing template→action path.
    /// (ADR 0001 D6/D7)
    let runSweep
        (machinConfig: MachinBridge.MachinConfig)
        (baseConfig: MctsConfig)
        (meta: Tars.DSL.Wot.DslMeta)
        (templates: Tars.DSL.Wot.DslNode list)
        (maxNodes: int)
        (configs: SweepConfig list)
        : MeshOutcome =

        let ixConfig = toIxConfig machinConfig

        // Serial degradation: run each config one-at-a-time through the existing
        // single-shot path. That path returns actions but no reward, so we rank by
        // derivation size as an explicit proxy (the mesh path ranks by true reward).
        let serialFallback () =
            let ranked =
                configs
                |> List.mapi (fun i c ->
                    let cfg =
                        { baseConfig with
                            ExplorationConstant = c.Exploration
                            MaxRolloutDepth = c.MaxDepth }
                    let actions, _ =
                        MctsBridge.searchWotDerivation (Some machinConfig) cfg meta templates maxNodes
                    { StageId = sprintf "s%d" i
                      Reward = float (List.length actions)
                      Iterations = cfg.MaxIterations
                      NodeIndices = [] },
                    actions)
            let best =
                ranked
                |> List.sortByDescending (fun (r, _) -> r.Reward)
                |> List.tryHead
                |> Option.map snd
                |> Option.defaultValue []
            { Best = best; Ranked = ranked |> List.map fst; UsedMesh = false }

        if List.isEmpty configs || not (IxSkill.pipelineAvailable ixConfig) then
            serialFallback ()
        else
            let tmpDir =
                Path.Combine(Path.GetTempPath(), sprintf "tars_mesh_%s" (Guid.NewGuid().ToString("N")))
            Directory.CreateDirectory tmpDir |> ignore
            try
                let yamlPath = Path.Combine(tmpDir, "mesh.yaml")
                let grammarPath = Path.Combine(tmpDir, "grammar.json")
                File.WriteAllText(yamlPath, buildMeshYaml configs)
                // The corpus file must be a JSON value; raw EBNF is wrapped as a
                // JSON-encoded string so `--param grammar=@file` parses it.
                File.WriteAllText(grammarPath, JsonSerializer.Serialize(MctsBridge.templatesToEbnf templates))

                let result =
                    (IxSkill.runPipelineJson ixConfig yamlPath [ "grammar", grammarPath ])
                        .GetAwaiter()
                        .GetResult()

                match result with
                | Result.Error _ -> serialFallback ()
                | Result.Ok json ->
                    let ranked = parseMeshOutput json
                    if List.isEmpty ranked then
                        serialFallback ()
                    else
                        let bestResult = ranked |> List.sortByDescending (fun r -> r.Reward) |> List.head
                        let best = MctsBridge.indicesToActions templates maxNodes bestResult.NodeIndices
                        if List.isEmpty best then serialFallback ()
                        else { Best = best; Ranked = ranked; UsedMesh = true }
            finally
                try Directory.Delete(tmpDir, true) with _ -> ()

    // ── Default evolve-cycle sweep ────────────────────────────────────────────
    // A self-contained entry point so the evolve loop can run a grammar mesh in
    // one call (ADR 0001 D4): sweep a small config grid over a default WoT
    // template pool, returning the winning derivation to record as an outcome.

    /// A minimal WoT template pool to sweep over when no caller supplies one —
    /// the common analyze→plan→execute→verify→refine shape.
    let defaultTemplates : Tars.DSL.Wot.DslNode list =
        [ { Tars.DSL.Wot.DslConvert.defaultNode "analyze" Tars.DSL.Wot.NodeKind.Reason with
              Goal = Some "Analyze the problem space" }
          { Tars.DSL.Wot.DslConvert.defaultNode "plan" Tars.DSL.Wot.NodeKind.Reason with
              Goal = Some "Create an execution plan" }
          { Tars.DSL.Wot.DslConvert.defaultNode "execute" Tars.DSL.Wot.NodeKind.Work with
              Tool = Some "code_execute" }
          { Tars.DSL.Wot.DslConvert.defaultNode "verify" Tars.DSL.Wot.NodeKind.Reason with
              Goal = Some "Verify correctness of results" }
          { Tars.DSL.Wot.DslConvert.defaultNode "refine" Tars.DSL.Wot.NodeKind.Reason with
              Goal = Some "Refine and improve the solution" } ]

    let private defaultMeta : Tars.DSL.Wot.DslMeta =
        { Id = "grammar-mesh-sweep"
          Title = "Grammar Mesh Sweep"
          Domain = "general"
          Objective = "Find optimal workflow structure via parallel grammar search"
          Constraints = []
          SuccessCriteria = [] }

    let private discoverMachinConfig () : MachinBridge.MachinConfig =
        match IxSkill.discover () with
        | Some c -> { MachinBridge.defaultConfig with WorkingDir = c.RepoDir }
        | None -> MachinBridge.defaultConfig

    /// Run the default evolve-cycle sweep: a small grid of grammar.search configs
    /// over the default template pool, ix discovered automatically. Used by
    /// `tars evolve --grammar-mesh`.
    let runDefaultSweep () : MeshOutcome =
        let baseConfig =
            { MctsTypes.defaultMctsConfig with MaxIterations = 500; MaxRolloutDepth = 12 }
        let sweep = buildSweep [ 1.0; 1.4; 2.0 ] [ 8; 12 ]
        runSweep (discoverMachinConfig ()) baseConfig defaultMeta defaultTemplates 5 sweep
