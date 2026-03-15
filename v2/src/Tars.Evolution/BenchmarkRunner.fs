namespace Tars.Evolution

open System
open System.Diagnostics
open System.IO
open System.Text.Json
open System.Threading.Tasks
open Tars.Llm
open Tars.Cortex
open Tars.Cortex.WoTTypes

/// Executes benchmark problems: LLM generates solution -> compile via dotnet fsi -> validate output.
module BenchmarkRunner =

    let private jsonOptions =
        JsonSerializerOptions(WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase)

    /// Extract F# code from LLM response (strip markdown fences, explanations).
    let private extractCode (response: string) : string =
        let lines = response.Split([|'\n'|])
        let mutable inFence = false
        let mutable codeLines = ResizeArray<string>()
        let mutable foundFence = false

        for line in lines do
            let trimmed = line.TrimStart()
            if trimmed.StartsWith("```") then
                if inFence then
                    inFence <- false
                else
                    inFence <- true
                    foundFence <- true
            elif inFence then
                codeLines.Add(line)

        if foundFence && codeLines.Count > 0 then
            String.Join("\n", codeLines)
        else
            // No fences — use the whole response, stripping common preamble
            response
                .Replace("```fsharp", "")
                .Replace("```", "")
                .Trim()

    /// Generate a solution via LLM.
    let private solveWithLlm (llm: ILlmService) (problem: BenchmarkProblem) : Task<string * int64> =
        task {
            let hints =
                if problem.Hints.IsEmpty then ""
                else sprintf "\nHINTS:\n%s" (problem.Hints |> List.map (fun h -> $"- {h}") |> String.concat "\n")

            let prompt = $"""Write F# code to solve this problem.

PROBLEM: {problem.Description}
REQUIRED SIGNATURE: {problem.ExpectedSignature}
{hints}

Output ONLY the F# code. No explanations, no markdown, no module declarations.
The code must define the function(s) with exactly the required signature(s).
Do not use 'open' statements — write self-contained code."""

            let sw = Stopwatch.StartNew()
            let! response =
                llm.CompleteAsync(
                    { ModelHint = None
                      Model = None
                      SystemPrompt = Some "You are an F# programming expert. Output only valid F# code."
                      MaxTokens = Some 800
                      Temperature = Some 0.2
                      Stop = []
                      Messages = [ { Role = Role.User; Content = prompt } ]
                      Tools = []
                      ToolChoice = None
                      ResponseFormat = None
                      Stream = false
                      JsonMode = false
                      Seed = None
                      ContextWindow = None })
            sw.Stop()
            return extractCode response.Text, sw.ElapsedMilliseconds
        }

    /// Compile and validate a solution via dotnet fsi.
    let private compileAndValidate (solutionCode: string) (validationCode: string) : Task<bool * bool * string list * string> =
        task {
            let tmpDir = Path.Combine(Path.GetTempPath(), "tars-benchmark")
            if not (Directory.Exists tmpDir) then
                Directory.CreateDirectory tmpDir |> ignore

            let scriptPath = Path.Combine(tmpDir, $"bench_{Guid.NewGuid():N}.fsx")
            let fullScript = solutionCode + "\n\n// === VALIDATION ===\n" + validationCode

            try
                File.WriteAllText(scriptPath, fullScript)

                let psi = ProcessStartInfo("dotnet", $"fsi \"{scriptPath}\"")
                psi.RedirectStandardOutput <- true
                psi.RedirectStandardError <- true
                psi.UseShellExecute <- false
                psi.CreateNoWindow <- true

                use proc = Process.Start(psi)
                let! stdout = proc.StandardOutput.ReadToEndAsync()
                let! stderr = proc.StandardError.ReadToEndAsync()

                let completed = proc.WaitForExit(30_000) // 30s timeout
                if not completed then
                    try proc.Kill() with _ -> ()

                let compiled = not (stderr.Contains("error FS"))
                let validated = stdout.Contains("PASS") && not (stdout.Contains("FAIL"))

                let errors =
                    if compiled then []
                    else
                        stderr.Split([|'\n'|])
                        |> Array.filter (fun l -> l.Contains("error FS"))
                        |> Array.toList

                return compiled, validated, errors, stdout.Trim()
            finally
                try File.Delete(scriptPath) with _ -> ()
        }

    /// Run a single benchmark problem.
    let runProblem (llm: ILlmService) (problem: BenchmarkProblem) (retryOnFail: bool) (logger: string -> unit) : Task<BenchmarkAttempt> =
        task {
            logger $"  [{problem.Id}] {problem.Title} ({problem.Difficulty})..."

            let! code, genMs = solveWithLlm llm problem

            let sw = Stopwatch.StartNew()
            let! compiled, validated, errors, output = compileAndValidate code problem.ValidationCode
            sw.Stop()

            // Retry once with error feedback if compilation failed
            if retryOnFail && not compiled && errors.Length > 0 then
                logger $"  [{problem.Id}] Retry with error feedback..."
                let retryPrompt = $"Your previous F# code had compilation errors:\n{String.Join('\n', errors)}\n\nFix the code. Output ONLY valid F# code.\n\nOriginal problem: {problem.Description}\nRequired signature: {problem.ExpectedSignature}"

                let! retryResponse =
                    llm.CompleteAsync(
                        { ModelHint = None
                          Model = None
                          SystemPrompt = Some "You are an F# programming expert. Fix the compilation errors."
                          MaxTokens = Some 800
                          Temperature = Some 0.1
                          Stop = []
                          Messages = [ { Role = Role.User; Content = retryPrompt } ]
                          Tools = []
                          ToolChoice = None
                          ResponseFormat = None
                          Stream = false
                          JsonMode = false
                          Seed = None
                          ContextWindow = None })

                let retryCode = extractCode retryResponse.Text
                let! compiled2, validated2, errors2, output2 = compileAndValidate retryCode problem.ValidationCode

                let status = if validated2 then "PASS" elif compiled2 then "FAIL (validation)" else "FAIL (compile)"
                logger $"  [{problem.Id}] {status}"

                return
                    { ProblemId = problem.Id
                      Difficulty = problem.Difficulty
                      Category = problem.Category
                      GeneratedCode = retryCode
                      Compiled = compiled2
                      Validated = validated2
                      CompileErrors = errors2
                      ValidationOutput = output2
                      GenerationTimeMs = genMs
                      ValidationTimeMs = sw.ElapsedMilliseconds
                      Timestamp = DateTime.UtcNow }
            else
                let status = if validated then "PASS" elif compiled then "FAIL (validation)" else "FAIL (compile)"
                logger $"  [{problem.Id}] {status}"

                return
                    { ProblemId = problem.Id
                      Difficulty = problem.Difficulty
                      Category = problem.Category
                      GeneratedCode = code
                      Compiled = compiled
                      Validated = validated
                      CompileErrors = errors
                      ValidationOutput = output
                      GenerationTimeMs = genMs
                      ValidationTimeMs = sw.ElapsedMilliseconds
                      Timestamp = DateTime.UtcNow }
        }

    /// Run a benchmark suite.
    let runSuite
        (llm: ILlmService)
        (difficulty: ProblemDifficulty option)
        (category: ProblemCategory option)
        (maxProblems: int option)
        (retryOnFail: bool)
        (logger: string -> unit)
        : Task<BenchmarkRunSummary> =
        task {
            let problems =
                ProblemBank.all ()
                |> List.filter (fun p ->
                    (difficulty |> Option.map (fun d -> p.Difficulty = d) |> Option.defaultValue true)
                    && (category |> Option.map (fun c -> p.Category = c) |> Option.defaultValue true))
                |> fun ps ->
                    match maxProblems with
                    | Some n -> ps |> List.truncate n
                    | None -> ps

            logger $"Running {problems.Length} benchmark problems..."
            let sw = Stopwatch.StartNew()

            let mutable attempts = []
            for problem in problems do
                let! attempt = runProblem llm problem retryOnFail logger
                attempts <- attempts @ [attempt]

            sw.Stop()

            let compiled = attempts |> List.filter (fun a -> a.Compiled) |> List.length
            let validated = attempts |> List.filter (fun a -> a.Validated) |> List.length

            return
                { RunId = Guid.NewGuid()
                  Timestamp = DateTime.UtcNow
                  ModelUsed = "default"
                  TotalProblems = attempts.Length
                  Compiled = compiled
                  Validated = validated
                  CompileRate = if attempts.Length > 0 then float compiled / float attempts.Length else 0.0
                  PassRate = if attempts.Length > 0 then float validated / float attempts.Length else 0.0
                  Attempts = attempts
                  TotalDurationMs = sw.ElapsedMilliseconds }
        }

    /// Record benchmark results into PatternOutcomeStore for self-improvement feedback.
    let recordOutcomes (summary: BenchmarkRunSummary) : unit =
        for attempt in summary.Attempts do
            PatternOutcomeStore.record
                { PatternKind = Custom $"benchmark:{attempt.Category}"
                  Goal = attempt.ProblemId
                  Success = attempt.Validated
                  DurationMs = attempt.GenerationTimeMs + attempt.ValidationTimeMs
                  Timestamp = attempt.Timestamp }

    /// Persist results to ~/.tars/benchmark_results/.
    let saveResults (summary: BenchmarkRunSummary) : string =
        let dir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".tars", "benchmark_results")
        if not (Directory.Exists dir) then
            Directory.CreateDirectory dir |> ignore

        let path = Path.Combine(dir, $"run_{summary.Timestamp:yyyyMMdd_HHmmss}.json")
        let json = JsonSerializer.Serialize(summary, jsonOptions)
        File.WriteAllText(path, json)

        // Also write latest.json
        let latestPath = Path.Combine(dir, "latest.json")
        File.WriteAllText(latestPath, json)

        path

    /// Load historical benchmark results.
    let loadHistory () : BenchmarkRunSummary list =
        try
            let dir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".tars", "benchmark_results")
            if Directory.Exists dir then
                Directory.GetFiles(dir, "run_*.json")
                |> Array.map (fun f ->
                    try
                        let json = File.ReadAllText(f)
                        Some (JsonSerializer.Deserialize<BenchmarkRunSummary>(json, jsonOptions))
                    with _ -> None)
                |> Array.choose id
                |> Array.sortBy (fun s -> s.Timestamp)
                |> Array.toList
            else []
        with _ -> []
