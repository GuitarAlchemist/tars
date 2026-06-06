namespace Tars.Tests

open System
open System.Net.Http
open System.Threading.Tasks
open Xunit
open Xunit.Abstractions
open Tars.Core
open Tars.Llm

/// Integration tests for Puzzles (requires running LLM)
/// Set TARS_INTEGRATION=1 to run these tests
type PuzzleIntegrationTests(output: ITestOutputHelper) =

    let httpClient = new HttpClient()
    let ollamaUri = Uri("http://localhost:11434")

    // Helper to check if integration tests should run
    let shouldRun () =
        let env = Environment.GetEnvironmentVariable("TARS_INTEGRATION")
        not (String.IsNullOrWhiteSpace(env))

    // Helper to create a simple LLM service wrapper
    let createService (http: HttpClient) (uri: Uri) (modelName: string) =
        { new ILlmService with
            member _.CompleteAsync(req) =
                OllamaClient.sendChatAsync http uri modelName None req

            member _.CompleteStreamAsync(req, onToken) =
                OllamaClient.sendChatStreamAsync http uri modelName None req onToken

            member _.EmbedAsync(text) =
                OllamaClient.getEmbeddingsAsync http uri "nomic-embed-text" text

            member _.RouteAsync(req) =
                Task.FromResult(
                    { Backend = Ollama modelName
                      Endpoint = uri
                      ApiKey = None }
                    : Tars.Llm.Routing.RoutedBackend
                ) }

    // Helper to solve a puzzle with Ollama
    let solvePuzzle (puzzle: Puzzle) =
        task {
            if not (shouldRun ()) then
                output.WriteLine("Skipping integration test (TARS_INTEGRATION not set)")
                return true
            else
                output.WriteLine($"Solving puzzle: {puzzle.Name}")

                // Create LLM Service
                let model = "qwen2.5-coder:1.5b"
                let llm = createService httpClient ollamaUri model

                let systemPrompt =
                    """You are an advanced reasoning engine. Solve the given puzzle step-by-step.
If the puzzle involves logic, be rigorous.
If it involves math, show calculations.
Finally, provide the answer clearly."""

                let userPrompt =
                    $"""PUZZLE: {puzzle.Name}
{puzzle.Description}

PROBLEM:
{puzzle.Prompt}

Solve this."""

                let req =
                    { LlmRequest.Default with
                        SystemPrompt = Some systemPrompt
                        Messages =
                            [ { Role = Role.User
                                Content = userPrompt } ]
                        Temperature = Some 0.1 // Low temp for logic
                        MaxTokens = Some 1000 }

                let! response = llm.CompleteAsync(req)

                output.WriteLine($"LLM Response:\n{response.Text}")

                let isCorrect = puzzle.Validator response.Text

                if isCorrect then
                    output.WriteLine("✅ SOLVED")
                else
                    output.WriteLine("❌ FAILED VALIDATION")

                return isCorrect
        }

    [<Fact>]
    member _.``Integration: Can solve River Crossing (Logic)``() =
        task {
            let puzzle = Puzzles.riverCrossingPuzzle
            let! success = solvePuzzle puzzle

            if shouldRun () then
                Assert.True(success, $"Failed to solve {puzzle.Name}")
        }

    [<Fact>]
    member _.``Integration: Can solve Math Word Problem``() =
        task {
            let puzzle = Puzzles.mathWordPuzzle
            let! success = solvePuzzle puzzle

            if shouldRun () then
                Assert.True(success, $"Failed to solve {puzzle.Name}")
        }
