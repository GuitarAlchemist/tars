namespace TarsEngine.FSharp.Core.Metascript

open System
open System.Globalization
open System.Text.RegularExpressions
open TarsEngine.FSharp.Core.Closures

/// Lightweight fractal metascript parser and generator used by the evolutionary closures.
module FractalGrammarMetascripts =

    /// Commands available in the fractal metascript DSL.
    type FractalRule =
        | SpawnAgent of agentType: GameTheoryAgentType * count: int * strategy: CoordinationStrategy
        | ConnectAgents of source: string * target: string * label: string
        | RepeatPattern of name: string * depth: int
        | EmitMetric of metric: string * value: float

    /// Parsed block containing rules and associated metadata.
    type FractalMetascriptBlock =
        { Name: string
          Rules: FractalRule list
          MaxDepth: int
          CreatedAt: DateTime }

    /// Parser for the fractal metascript DSL.
    type FractalMetascriptParser() =

        let strategyFromString (value: string) =
            match value.Trim().ToUpperInvariant() with
            | "HIERARCHICAL" -> CoordinationStrategy.Hierarchical "leader"
            | "DEMOCRATIC" -> CoordinationStrategy.Democratic
            | "SPECIALIZED" -> CoordinationStrategy.Specialized
            | "SWARM" -> CoordinationStrategy.Swarm
            | "FRACTAL" -> CoordinationStrategy.FractalSelfOrganizing
            | _ -> CoordinationStrategy.Hierarchical "leader"

        let agentTypeFromString (value: string) =
            match value.Trim().ToUpperInvariant() with
            | "QRE" -> GameTheoryAgentType.QuantalResponseEquilibrium 1.0
            | "CH" -> GameTheoryAgentType.CognitiveHierarchy 3
            | "NOREGRET" -> GameTheoryAgentType.NoRegretLearning 0.9
            | "EGT" -> GameTheoryAgentType.EvolutionaryGameTheory 0.1
            | "CE" -> GameTheoryAgentType.CorrelatedEquilibrium 0.0
            | "ML" -> GameTheoryAgentType.MachineLearningAgent "default"
            | _ -> GameTheoryAgentType.MachineLearningAgent value

        let parseSpawn tokens =
            match tokens with
            | [ agentType; count; strategy ] ->
                SpawnAgent(agentTypeFromString agentType, int count, strategyFromString strategy)
            | _ -> invalidArg "tokens" "SPAWN requires exactly three arguments."

        let parseConnect tokens =
            match tokens with
            | [ source; target; label ] -> ConnectAgents(source, target, label)
            | _ -> invalidArg "tokens" "CONNECT requires three arguments."

        let parseRepeat tokens =
            match tokens with
            | [ name; depth ] -> RepeatPattern(name, int depth)
            | _ -> invalidArg "tokens" "REPEAT requires two arguments."

        let parseMetric tokens =
            match tokens with
            | [ metric; value ] -> EmitMetric(metric, Double.Parse(value, CultureInfo.InvariantCulture))
            | _ -> invalidArg "tokens" "METRIC requires two arguments."

        member _.ParseFractalMetascript(content: string) : FractalMetascriptBlock =
            let lines =
                content.Split([| '\n'; '\r' |], StringSplitOptions.RemoveEmptyEntries)
                |> Array.map (fun line -> line.Trim())
                |> Array.filter (fun line -> not (String.IsNullOrWhiteSpace(line) || line.StartsWith("#")))

            let rules =
                lines
                |> Array.map (fun line ->
                    let segments = Regex.Split(line, "\s+") |> Array.toList
                    match segments with
                    | [] -> None
                    | command :: args ->
                        match command.ToUpperInvariant() with
                        | "SPAWN" -> Some(parseSpawn args)
                        | "CONNECT" -> Some(parseConnect args)
                        | "REPEAT" -> Some(parseRepeat args)
                        | "METRIC" -> Some(parseMetric args)
                        | _ -> raise (FormatException $"Unknown metascript command '{command}'."))
                |> Array.choose id
                |> Array.toList

            { Name = "Parsed Fractal Metascript"
              Rules = rules
              MaxDepth =
                  rules
                  |> List.choose (function RepeatPattern(_, depth) -> Some depth | _ -> None)
                  |> function
                      | [] -> 1
                      | depths -> List.max depths
              CreatedAt = DateTime.UtcNow }

    /// Generator producing deterministic scripts for the fractal metascript DSL.
    type FractalMetascriptGenerator() =

        member _.GenerateTeamCoordinationMetascript(teamSize: int, strategy: CoordinationStrategy) =
            let strategyName =
                match strategy with
                | CoordinationStrategy.Hierarchical _ -> "HIERARCHICAL"
                | CoordinationStrategy.Democratic -> "DEMOCRATIC"
                | CoordinationStrategy.Specialized -> "SPECIALIZED"
                | CoordinationStrategy.Swarm -> "SWARM"
                | CoordinationStrategy.FractalSelfOrganizing -> "FRACTAL"

            let lines =
                [ "# Fractal team coordination metascript"
                  $"SPAWN QRE {min teamSize 3} {strategyName}"
                  "SPAWN ML 2 FRACTAL"
                  "CONNECT leader agent-1 lead"
                  "CONNECT agent-1 agent-2 support"
                  "REPEAT coordination-pattern 2"
                  "METRIC team_cohesion 0.82" ]

            String.Join(Environment.NewLine, lines)

        member _.GenerateFractalSpawningMetascript(depth: int) =
            if depth <= 0 then invalidArg "depth" "Depth must be greater than zero."

            let lines =
                [ "# Fractal spawning metascript"
                  $"REPEAT spawn-pattern {depth}"
                  "SPAWN EGT 1 FRACTAL"
                  "SPAWN CH 2 HIERARCHICAL"
                  "CONNECT agent-0 agent-1 explore"
                  "CONNECT agent-0 agent-2 evaluate"
                  "METRIC branching_factor 0.75" ]

            String.Join(Environment.NewLine, lines)
