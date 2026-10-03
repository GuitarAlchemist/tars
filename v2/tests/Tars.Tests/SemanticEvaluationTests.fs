namespace Tars.Tests

open System
open System.Threading.Tasks
open Xunit
open Tars.Evolution
open Tars.Llm
open Tars.Core

module SemanticEvaluationTests =

    type FixedLlm(responseText: string) =
        interface ILlmService with
            member _.CompleteAsync(_req) =
                task {
                    return
                        { Text = responseText
                          FinishReason = Some "stop"
                          Usage = None
                          Raw = None }
                }

            member _.CompleteStreamAsync(_req, _handler) =
                task {
                    return
                        { Text = responseText
                          FinishReason = Some "stop"
                          Usage = None
                          Raw = None }
                }

            member _.EmbedAsync(_text) = Task.FromResult([| 0.0f |])
            member _.RouteAsync(_) = task { return { Backend = Ollama "mock"; Endpoint = Uri "http://localhost:11434"; ApiKey = None } }

    /// Gives `responseText` to every request, and records each request's messages in `prompts`.
    type RecordingLlm(responseText: string, prompts: ResizeArray<string>) =
        interface ILlmService with
            member _.CompleteAsync(req) =
                prompts.Add(req.Messages |> List.map (fun m -> m.Content) |> String.concat "\n")
                (FixedLlm(responseText) :> ILlmService).CompleteAsync(req)

            member _.CompleteStreamAsync(req, handler) =
                (FixedLlm(responseText) :> ILlmService).CompleteStreamAsync(req, handler)

            member _.EmbedAsync(text) = Task.FromResult([| 0.0f |])
            member _.RouteAsync(req) = (FixedLlm(responseText) :> ILlmService).RouteAsync(req)

    let private sampleTask =
        { Id = Guid.NewGuid()
          DifficultyLevel = 1
          Goal = "Return the sum of two integers."
          Constraints = []
          ValidationCriteria = "Sum is correct"
          Timeout = TimeSpan.FromSeconds(1.0)
          Score = 1.0 }

    let private sampleResult =
        { TaskId = sampleTask.Id
          TaskGoal = sampleTask.Goal
          ExecutorId = AgentId(Guid.NewGuid())
          Success = true
          Output = "let add a b = a + b"
          ExecutionTrace = []
          Duration = TimeSpan.FromSeconds(1.0)
          Evaluation = None }

    [<Fact>]
    let ``SemanticEvaluation parses valid JSON`` () =
        task {
            let llm =
                FixedLlm(
                    "{\"passed\":true,\"confidence\":0.9,\"summary\":\"ok\",\"issues\":[],\"suggested_fixes\":[]}"
                )
                :> ILlmService

            let evaluator = SemanticEvaluation(llm, minConfidence = 0.6) :> IEvaluationStrategy
            let! eval = evaluator.Evaluate(sampleTask, sampleResult)
            Assert.True(eval.Passed)
            Assert.True(eval.Confidence > 0.8)
        }

    [<Fact>]
    let ``SemanticEvaluation enforces confidence threshold`` () =
        task {
            let llm =
                FixedLlm(
                    "{\"passed\":true,\"confidence\":0.2,\"summary\":\"low confidence\",\"issues\":[],\"suggested_fixes\":[]}"
                )
                :> ILlmService

            let evaluator = SemanticEvaluation(llm, minConfidence = 0.6) :> IEvaluationStrategy
            let! eval = evaluator.Evaluate(sampleTask, sampleResult)
            Assert.False(eval.Passed)
            Assert.Contains("below threshold", eval.Summary)
        }

    [<Fact>]
    let ``SemanticEvaluation judges only what the passing examples do not check`` () =
        task {
            // With --run-code the examples ran with the code and passed; they check its values,
            // not what else the goal asks, like a `flatten` written without List.concat.
            let prompts = ResizeArray<string>()

            let llm =
                RecordingLlm(
                    "{\"passed\":true,\"confidence\":0.9,\"summary\":\"ok\",\"issues\":[],\"suggested_fixes\":[]}",
                    prompts
                )
                :> ILlmService

            let examplesPassed =
                { Passed = true
                  Confidence = 1.0
                  Summary = "The code ran and gave the expected value for all 2 examples."
                  Issues = []
                  SuggestedFixes = []
                  EvaluatedAt = DateTime.UtcNow }

            let evaluator = SemanticEvaluation(llm, minConfidence = 0.6) :> IEvaluationStrategy
            let! _ = evaluator.Evaluate(sampleTask, { sampleResult with Evaluation = Some examplesPassed })
            let! _ = evaluator.Evaluate(sampleTask, sampleResult)

            Assert.Contains("The code ran and gave the expected value for all 2 examples.", prompts.[0])
            Assert.Contains("Judge only", prompts.[0])
            Assert.DoesNotContain("Judge only", prompts.[1])
        }
