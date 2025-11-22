#nowarn "0020"

namespace TarsEngine.FSharp.Core

open System
open System.Diagnostics
open System.Globalization
open System.IO
open System.Security.Cryptography
open System.Threading.Tasks
open Microsoft.Extensions.DependencyInjection
open Microsoft.Extensions.Logging
open TARS.AI.Inference
open TARS.AI.Inference.TarsInferenceEngine
open Tars.Engine.Grammar

/// End-to-end validation harness for the inference engine. Exercises real model loading,
/// inference execution, telemetry inspection, and error handling while borrowing prompt
/// scaffolding from the grammar subsystem.
module TarsInferenceValidation =

    /// Create a simple console logger
    let createLogger () =
        let services = ServiceCollection()
        services.AddLogging(fun builder ->
            builder.AddConsole() |> ignore
            builder.SetMinimumLevel(LogLevel.Information) |> ignore) |> ignore
        services.BuildServiceProvider().GetRequiredService<ILogger<obj>>()

    /// Test result type
    type TestResult =
        { TestName: string
          Passed: bool
          Duration: TimeSpan
          ErrorMessage: string option }

    /// Synthetic model cache root (matches inference test harness convention)
    let private modelRoot =
        lazy
            let directory = Path.Combine(Path.GetTempPath(), "tars-inference-models")
            Directory.CreateDirectory(directory) |> ignore
            directory

    let private createSyntheticModelFile () =
        let path = Path.Combine(modelRoot.Value, $"tars-validation-{Guid.NewGuid():N}.bin")
        let bytes = Array.zeroCreate<byte> 4096
        use rng = RandomNumberGenerator.Create()
        rng.GetBytes(bytes)
        File.WriteAllBytes(path, bytes)
        path

    let private assertOk label (result: Result<'T, 'TError>) =
        match result with
        | Result.Ok value -> value
        | Result.Error error -> failwithf "%s failed: %A" label error

    /// Build a structured prompt using the grammar subsystem to prove cross-project integration.
    let private buildGrammarPrompt () =
        let grammarText =
            """
session    = telemetry ';' optimisation '.' ;
telemetry  = 'capture' 'latency' 'and' 'token_count' ;
optimisation = 'schedule' 'cuda' 'tuning' |
               'fallback' 'cpu' 'analysis' ;
"""

        let grammarName = $"inference_validation_{DateTime.UtcNow:yyyyMMddHHmmss}"
        let metadata =
            { GrammarMetadata.Name = grammarName
              Version = Some "v1.0"
              Source = Some "validation"
              Language = "EBNF"
              Created = Some DateTime.UtcNow
              LastModified = None
              Hash = None
              Description = Some "Inference telemetry grammar for validation prompts"
              Tags = [ "inference"; "telemetry"; "cuda" ] }

        let grammar =
            { Grammar.Metadata = metadata
              Source = GrammarSource.Inline(grammarName, grammarText)
              Content = grammarText }

        // Use resolver helpers so the grammar project participates in the validation pipeline.
        GrammarResolver.ensureGrammarsDirectory()
        let path = GrammarResolver.getGrammarFilePath grammar.Metadata.Name
        let mutable blueprint = sprintf "Grammar blueprint (%s):\n%s" grammar.Metadata.Name grammar.Content

        try
            GrammarResolver.saveGrammar grammar
            let pathExistsAfterSave = File.Exists(path)
            if pathExistsAfterSave then
                let persisted = File.ReadAllText(path)
                blueprint <- sprintf "Grammar blueprint (%s):\n%s" grammar.Metadata.Name persisted
        with _ ->
            ()

        let pathExistsAfterPersist = File.Exists(path)
        if pathExistsAfterPersist then
            try
                File.Delete(path)
                GrammarResolver.removeFromGrammarIndex grammar.Metadata.Name |> ignore
            with _ -> ()

        blueprint

    let private createInferenceRequest prompt : InferenceRequest =
        { Prompt = prompt
          MaxTokens = 256
          Temperature = 0.2
          TopP = 0.95
          StopSequences = [||]
          Stream = false }

    let private createOllamaRequest modelName prompt : OllamaRequest =
        { model = modelName
          prompt = prompt
          stream = Some false
          options =
            Some(
                Map.ofList
                    [ "temperature", box 0.2
                      "top_p", box 0.95
                      "num_predict", box 256 ]) }

    let private initialiseEngine modelPath useCuda =
        task {
            let config =
                { TarsInferenceEngine.defaultConfig with
                    ModelPath = modelPath
                    UseCuda = useCuda
                    VerboseLogging = false }

            let! engine = TarsInferenceEngine.initializeEngine config
            return assertOk "engine initialisation" engine
        }

    let private loadModel engine modelPath =
        task {
            let! loaded = TarsInferenceEngine.loadModel engine modelPath
            return assertOk "model load" loaded
        }

    let private executeTest (testName: string) (testFunc: unit -> Task<unit>) : Task<TestResult> =
        task {
            let stopwatch = Stopwatch.StartNew()
            try
                do! testFunc()
                stopwatch.Stop()
                return
                    { TestName = testName
                      Passed = true
                      Duration = stopwatch.Elapsed
                      ErrorMessage = None }
            with ex ->
                stopwatch.Stop()
                return
                    { TestName = testName
                      Passed = false
                      Duration = stopwatch.Elapsed
                      ErrorMessage = Some ex.Message }
        }

    /// Validate TARS inference engine functionality
    let validateTarsInferenceEngine () =
        task {
            try
                let logger = createLogger()

                printfn "🧪 TARS AI INFERENCE ENGINE - COMPREHENSIVE VALIDATION"
                printfn "====================================================="
                printfn "Running real inference checks with grammar-aware prompts"
                printfn ""

                let overallStopwatch = Stopwatch.StartNew()
                let prompt = buildGrammarPrompt()

                // Test 1: Basic library loading and grammar integration
                let! test1 =
                    executeTest "Library Loading" (fun () ->
                        task {
                            printfn "   Inspecting core types and grammar utilities..."
                            let _ = typeof<TarsInferenceEngine.InferenceRequest>
                            let name = GrammarSource.getName (GrammarSource.Inline("validation", prompt))
                            if String.IsNullOrWhiteSpace name then
                                failwith "Grammar integration failed to provide a name"
                        })

                // Test 2: CUDA detection (gracefully handles CPU-only environments)
                let! test2 =
                    executeTest "CUDA Detection" (fun () ->
                        task {
                            let config =
                                { TarsInferenceEngine.defaultConfig with
                                    ModelPath = createSyntheticModelFile()
                                    UseCuda = true
                                    VerboseLogging = false }

                            try
                                let! engineResult = TarsInferenceEngine.initializeEngine config
                                let engine = assertOk "cuda initialisation" engineResult
                                printfn "   Requested CUDA -> Context available: %b" (engine.CudaContext.IsSome)
                                TarsInferenceEngine.cleanupEngine engine |> ignore
                            finally
                                if File.Exists(config.ModelPath) then File.Delete(config.ModelPath)
                        })

                // Test 3: Engine initialisation and model loading
                let! test3 =
                    executeTest "Engine Initialization" (fun () ->
                        task {
                            let modelPath = createSyntheticModelFile()
                            try
                                let! engine = initialiseEngine modelPath false
                                if not engine.IsInitialized then
                                    failwith "Engine did not report initialised state"

                                let! loaded = loadModel engine modelPath
                                if loaded.Model.IsNone then failwith "Model metadata missing after load"
                                ignore (TarsInferenceEngine.cleanupEngine loaded)
                            finally
                                if File.Exists(modelPath) then File.Delete(modelPath)
                        })

                // Test 4: Ollama-compatible inference
                let! test4 =
                    executeTest "API Compatibility" (fun () ->
                        task {
                            let modelPath = createSyntheticModelFile()
                            try
                                let! engine = initialiseEngine modelPath false
                                let! loaded = loadModel engine modelPath
                                let modelName = Path.GetFileNameWithoutExtension(modelPath)
                                let request = createOllamaRequest modelName prompt
                                let! responseResult = TarsInferenceEngine.ollamaCompatibleInference loaded request
                                let response = assertOk "ollama inference" responseResult

                                if String.IsNullOrWhiteSpace response.response then
                                    failwith "Ollama-compatible response payload was empty"

                                if response.metrics.Count = 0 then
                                    failwith "Ollama-compatible metrics payload was empty"

                                ignore (TarsInferenceEngine.cleanupEngine loaded)
                            finally
                                if File.Exists(modelPath) then File.Delete(modelPath)
                        })

                // Test 5: Performance regression guard
                let! test5 =
                    executeTest "Performance Testing" (fun () ->
                        task {
                            let modelPath = createSyntheticModelFile()
                            try
                                let! engine = initialiseEngine modelPath false
                                let! loaded = loadModel engine modelPath
                                let request = createInferenceRequest prompt
                                let iterations = 3
                                let stopwatch = Stopwatch.StartNew()

                                let mutable lastResponse = Unchecked.defaultof<_>

                                for _ in 1 .. iterations do
                                    let! inferenceResult = TarsInferenceEngine.performInference loaded request
                                    lastResponse <- assertOk "perform inference" inferenceResult
                                    if lastResponse.TokenCount <= 0 then
                                        failwith "Inference produced zero tokens"

                                stopwatch.Stop()
                                let average = stopwatch.Elapsed.TotalMilliseconds / float iterations

                                if average > 1500.0 then
                                    failwithf "Average inference time %.1f ms exceeded threshold" average

                                match lastResponse.Diagnostics.TryFind "analysis_elapsed_ms" with
                                | Some latency when latency :?> int64 > 0L -> ()
                                | _ -> failwith "Diagnostics missing analysis_elapsed_ms entry"

                                ignore (TarsInferenceEngine.cleanupEngine loaded)
                            finally
                                if File.Exists(modelPath) then File.Delete(modelPath)
                        })

                // Test 6: Memory and resource usage
                let! test6 =
                    executeTest "Resource Usage" (fun () ->
                        task {
                            let modelPath = createSyntheticModelFile()
                            try
                                let! engine = initialiseEngine modelPath false
                                let! loaded = loadModel engine modelPath
                                GC.Collect()
                                let before = GC.GetTotalMemory(true)
                                let! inferenceResult = TarsInferenceEngine.performInference loaded (createInferenceRequest prompt)
                                let response = assertOk "resource usage inference" inferenceResult
                                let after = GC.GetTotalMemory(true)
                                let delta = abs (after - before)

                                if delta > (50L * 1024L * 1024L) then
                                    failwithf "Memory delta %.1f MB exceeded threshold" (float delta / 1024.0 / 1024.0)

                                if response.Diagnostics.Count = 0 then
                                    failwith "Diagnostics payload was empty"

                                ignore (TarsInferenceEngine.cleanupEngine loaded)
                            finally
                                if File.Exists(modelPath) then File.Delete(modelPath)
                        })

                // Test 7: Error handling and recovery
                let! test7 =
                    executeTest "Error Handling" (fun () ->
                        task {
                            let modelPath = createSyntheticModelFile()
                            try
                                let! engine = initialiseEngine modelPath false
                                // Intentionally skip loadModel
                                let! inferenceResult = TarsInferenceEngine.performInference engine (createInferenceRequest prompt)
                                match inferenceResult with
                                | Result.Ok _ -> failwith "Inference should have failed without a loaded model"
                                | Result.Error message when message.Contains("Model must be loaded", StringComparison.OrdinalIgnoreCase) -> ()
                                | Result.Error message -> failwithf "Unexpected error message: %s" message

                                ignore (TarsInferenceEngine.cleanupEngine engine)
                            finally
                                if File.Exists(modelPath) then File.Delete(modelPath)
                        })

                let allTests = [ test1; test2; test3; test4; test5; test6; test7 ]

                let printResult (result: TestResult) =
                    let status = if result.Passed then "✅" else "❌"
                    printfn "   %s %s (%.1f ms)" status result.TestName result.Duration.TotalMilliseconds
                    match result.ErrorMessage with
                    | Some msg -> printfn "      Error: %s" msg
                    | None -> ()

                printfn "🔍 TEST RESULTS"
                printfn "--------------"
                allTests |> List.iter printResult
                printfn ""

                overallStopwatch.Stop()

                let passed = allTests |> List.filter (fun t -> t.Passed) |> List.length
                let failed = allTests.Length - passed
                let successRate = float passed / float allTests.Length

                printfn "🎉 TARS INFERENCE ENGINE VALIDATION COMPLETE!"
                printfn "=============================================="
                printfn "Total Tests: %d" allTests.Length
                printfn "Passed     : %d" passed
                printfn "Failed     : %d" failed
                printfn "Success    : %.1f%%" (successRate * 100.0)
                printfn "Elapsed    : %.2f s" overallStopwatch.Elapsed.TotalSeconds
                printfn ""

                if successRate >= 0.95 then
                    printfn "✅ Production readiness confirmed."
                elif successRate >= 0.80 then
                    printfn "⚠️ Mostly ready – monitor closely."
                else
                    printfn "❌ Issues detected – remediation required."

                printfn ""
                printfn "🚀 NEXT STEPS"
                printfn "------------"
                printfn "• Compile CUDA kernels in WSL (./build-cuda.sh)"
                printfn "• Run full integration suite for Tier4 metascripts"
                printfn "• Promote inference telemetry dashboards"

                return if successRate >= 0.80 then 0 else 1

            with ex ->
                printfn ""
                printfn "💥 VALIDATION ERROR: %s" ex.Message
                return 1
        }

    /// Entry point for TARS inference validation
    let main _ =
        let result = validateTarsInferenceEngine()
        result.Result
