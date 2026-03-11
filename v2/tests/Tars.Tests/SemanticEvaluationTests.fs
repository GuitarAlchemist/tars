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
