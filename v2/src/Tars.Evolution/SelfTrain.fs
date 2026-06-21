namespace Tars.Evolution

open System
open System.IO
open System.Text
open System.Text.Json

/// Level-4 self-improvement: turn the benchmark loop's **verified** solutions
/// into a supervised fine-tuning (SFT) dataset, so a local model can be trained
/// on its own correct experience.
///
/// The labels are not LLM-judged — every example is a solution that compiled and
/// printed PASS under deterministic validation (see BenchmarkRunner). That makes
/// this one of the few self-training loops that won't collapse: each row is
/// ground-truth-correct by construction.
///
/// This module produces the dataset (the part TARS can do natively). The actual
/// weight update is an external GPU step (unsloth / llama.cpp); `tars self-train`
/// prints the runbook. The A/B measurement is just two `benchmark code` runs.
module SelfTrain =

    /// All known problems, indexed by id (generic coding + GA music-theory).
    let private problemsById =
        (ProblemBank.all () @ GaProblemBank.all ())
        |> List.map (fun p -> p.Id, p)
        |> Map.ofList

    type ExportStats =
        { TotalAttempts: int
          VerifiedExamples: int
          UniqueProblems: int
          ByCategory: (string * int) list
          /// Number of problems where a single fastest (timed) variant was chosen.
          FastestSelected: int
          OutputPath: string
          ModelfilePath: string }

    /// Collect every validated attempt across saved benchmark runs and emit an
    /// SFT JSONL dataset. `domainFilter` optionally keeps only one category
    /// (e.g. MusicTheory for GA-only self-training).
    let exportDataset (outPath: string) (domainFilter: ProblemCategory option) : ExportStats =
        let summaries = BenchmarkRunner.loadHistory ()

        let verified =
            summaries
            |> List.collect (fun s -> s.Attempts)
            |> List.filter (fun a -> a.Validated)
            // Drop solutions that passed the example cases but FAILED property
            // testing (Some false) — they're overfit/buggy and must not become
            // training data. None (no properties) and Some true are kept.
            |> List.filter (fun a -> a.PropertiesValidated <> Some false)
            |> List.filter (fun a ->
                match domainFilter with
                | Some c -> a.Category = c
                | None -> true)

        // Per-problem selection. Where the problem is *timed* (has ExecutionNs),
        // keep only the single FASTEST verified variant, so the model trains on
        // performance idioms rather than any correct answer. Untimed problems keep
        // distinct-correct variants (unrankable, so diversity helps). The bool tags
        // a fastest-pick so we can count only picks that actually land in the set.
        let chosen : (BenchmarkAttempt * bool) list =
            verified
            |> List.groupBy (fun a -> a.ProblemId)
            |> List.collect (fun (_, attempts) ->
                let timed = attempts |> List.filter (fun a -> a.ExecutionNs.IsSome)
                if not (List.isEmpty timed) then
                    [ (timed |> List.minBy (fun a -> a.ExecutionNs.Value)), true ]
                else
                    attempts
                    |> List.distinctBy (fun a -> a.GeneratedCode.Trim())
                    |> List.map (fun a -> a, false))

        // Anonymous records serialize cleanly under System.Text.Json;
        // named F# records (without the FSharp converter) emit "{}".
        let examples =
            chosen
            |> List.choose (fun (a, isFastest) ->
                match Map.tryFind a.ProblemId problemsById with
                | Some problem ->
                    let ex =
                        {| messages =
                            [ {| role = "system"; content = BenchmarkRunner.solverSystemPrompt |}
                              {| role = "user"; content = BenchmarkRunner.buildSolverPrompt problem |}
                              {| role = "assistant"; content = a.GeneratedCode.Trim() |} ] |}
                    Some (a.ProblemId, a.Category, isFastest, ex)
                | None -> None)

        let fastestSelected = examples |> List.filter (fun (_, _, f, _) -> f) |> List.length

        let dir = Path.GetDirectoryName(outPath)
        if not (String.IsNullOrEmpty dir) && not (Directory.Exists dir) then
            Directory.CreateDirectory dir |> ignore

        let opts = JsonSerializerOptions(WriteIndented = false)
        let sb = StringBuilder()
        for (_, _, _, ex) in examples do
            sb.AppendLine(JsonSerializer.Serialize(ex, opts)) |> ignore
        File.WriteAllText(outPath, sb.ToString())

        // Drop an Ollama Modelfile template next to the dataset so step 3 of the
        // runbook is copy-paste once a fine-tuned GGUF exists. Inference params
        // (temperature, system prompt) mirror how the benchmark queries the model.
        let baseDir = if String.IsNullOrEmpty dir then "." else dir
        let modelfilePath = Path.Combine(baseDir, "Modelfile")
        let modelfile =
            String.concat "\n"
                [ "# TARS self-train — build the fine-tuned model with:"
                  "#   ollama create tars-coder -f Modelfile"
                  "# Place your fine-tuned GGUF (from unsloth / llama-factory, converted via"
                  "# llama.cpp convert_hf_to_gguf.py) next to this file as tars-coder.gguf."
                  "FROM ./tars-coder.gguf"
                  "PARAMETER temperature 0.2"
                  "PARAMETER num_ctx 4096"
                  sprintf "SYSTEM \"\"\"%s\"\"\"" BenchmarkRunner.solverSystemPrompt
                  "" ]
        File.WriteAllText(modelfilePath, modelfile)

        { TotalAttempts = summaries |> List.sumBy (fun s -> s.Attempts.Length)
          VerifiedExamples = examples.Length
          UniqueProblems = examples |> List.map (fun (pid, _, _, _) -> pid) |> List.distinct |> List.length
          ByCategory =
            examples
            |> List.countBy (fun (_, c, _, _) -> sprintf "%A" c)
            |> List.sortByDescending snd
          FastestSelected = fastestSelected
          OutputPath = outPath
          ModelfilePath = modelfilePath }
