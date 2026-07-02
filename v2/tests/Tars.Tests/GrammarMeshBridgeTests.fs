namespace Tars.Tests

open Xunit
open Tars.Evolution
open Tars.Evolution.GrammarMeshBridge

/// Unit tests for the pure (ix-independent) surface of GrammarMeshBridge:
/// sweep grid construction, ix.yaml emission, and pipeline-output parsing.
/// The mesh-execution path itself is exercised against a live `ix` binary.
module GrammarMeshBridgeTests =

    // ── buildSweep ────────────────────────────────────────────────────────────

    [<Fact>]
    let ``buildSweep produces the cartesian product`` () =
        let sweep = GrammarMeshBridge.buildSweep [ 1.0; 1.4; 2.0 ] [ 4; 8 ]
        Assert.Equal(6, List.length sweep)
        // every (exploration, depth) pair present exactly once
        Assert.Contains({ Exploration = 1.4; MaxDepth = 8 }, sweep)
        Assert.Contains({ Exploration = 2.0; MaxDepth = 4 }, sweep)

    [<Fact>]
    let ``buildSweep is empty when either axis is empty`` () =
        Assert.Empty(GrammarMeshBridge.buildSweep [] [ 4 ])
        Assert.Empty(GrammarMeshBridge.buildSweep [ 1.0 ] [])

    // ── buildMeshYaml ─────────────────────────────────────────────────────────

    [<Fact>]
    let ``buildMeshYaml emits one stage per config, param-bound corpus, inline knobs`` () =
        let configs =
            [ { Exploration = 1.0; MaxDepth = 4 }
              { Exploration = 2.0; MaxDepth = 8 } ]
        let yaml = GrammarMeshBridge.buildMeshYaml configs

        // one grammar.search stage per config
        let stageCount =
            yaml.Split('\n') |> Array.filter (fun l -> l.Trim() = "skill: grammar.search") |> Array.length
        Assert.Equal(2, stageCount)
        Assert.Contains("s0:", yaml)
        Assert.Contains("s1:", yaml)
        // shared corpus is param-bound, not inlined
        Assert.Contains("grammar_ebnf: { param: \"grammar\" }", yaml)
        Assert.Contains("grammar: null", yaml)
        // per-stage knobs are inline
        Assert.Contains("max_depth: 8", yaml)
        Assert.Contains("exploration: 2", yaml)

    [<Fact>]
    let ``buildMeshYaml formats floats with invariant culture`` () =
        // A locale that uses ',' as the decimal separator must not leak into yaml.
        let prev = System.Threading.Thread.CurrentThread.CurrentCulture
        try
            System.Threading.Thread.CurrentThread.CurrentCulture <-
                System.Globalization.CultureInfo.GetCultureInfo("fr-FR")
            let yaml = GrammarMeshBridge.buildMeshYaml [ { Exploration = 1.5; MaxDepth = 4 } ]
            Assert.Contains("exploration: 1.5", yaml)
            Assert.DoesNotContain("exploration: 1,5", yaml)
        finally
            System.Threading.Thread.CurrentThread.CurrentCulture <- prev

    // ── parseMeshOutput ───────────────────────────────────────────────────────

    [<Fact>]
    let ``parseMeshOutput extracts reward and node indices per stage`` () =
        // Shape produced by `ix pipeline run --format json` for a grammar.search mesh.
        let json =
            """{ "stages": {
                   "s0": { "output": { "best_derivation": [ {"alternative":["node_1"],"nonterminal":"root"} ], "iterations": 500, "reward": 1.0 } },
                   "s1": { "output": { "best_derivation": [ {"alternative":["node_0"],"nonterminal":"root"} ], "iterations": 300, "reward": 0.5 } }
                 } }"""
        let results = GrammarMeshBridge.parseMeshOutput json

        Assert.Equal(2, List.length results)
        let s0 = results |> List.find (fun r -> r.StageId = "s0")
        Assert.Equal(1.0, s0.Reward)
        Assert.Equal(500, s0.Iterations)
        Assert.Equal<int list>([ 1 ], s0.NodeIndices)
        let s1 = results |> List.find (fun r -> r.StageId = "s1")
        Assert.Equal(0.5, s1.Reward)
        Assert.Equal<int list>([ 0 ], s1.NodeIndices)

    [<Fact>]
    let ``parseMeshOutput returns empty on malformed or stageless json`` () =
        Assert.Empty(GrammarMeshBridge.parseMeshOutput "not json")
        Assert.Empty(GrammarMeshBridge.parseMeshOutput """{ "other": 1 }""")
