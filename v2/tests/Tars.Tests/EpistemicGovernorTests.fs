namespace Tars.Tests

open System
open System.Threading.Tasks
open Xunit
open Tars.Core
open Tars.Cortex
open Tars.Llm
open Tars.Llm.LlmService

module EpistemicGovernorTests =

    let createMockLlm (handler: LlmRequest -> string) =
        { new ILlmService with
            member _.CompleteAsync req =
                task {
                    return
                        { Text = handler req
                          Usage = None
                          FinishReason = Some "stop"
                          Raw = None }
                }

            member _.CompleteStreamAsync(req, handler) = raise (NotImplementedException())
            member _.EmbedAsync text = raise (NotImplementedException()) }

    [<Fact>]
    let ``ExtractPrinciple parses LLM response correctly`` () =
        task {
            let llmResponse =
                """Statement: Use recursion for tree traversal.
Context: When data structure is hierarchical."""

            let mockLlm = createMockLlm (fun _ -> llmResponse)
            let governor = EpistemicGovernor(mockLlm, None, None) :> IEpistemicGovernor

            let! belief = governor.ExtractPrinciple("Traverse a tree", "let rec traverse node = ...")

            Assert.Equal("Use recursion for tree traversal.", belief.Statement)
            Assert.Equal("When data structure is hierarchical.", belief.Context)
            Assert.Equal(EpistemicStatus.Hypothesis, belief.Status)
        }

    [<Fact>]
    let ``SuggestCurriculum returns trimmed response`` () =
        task {
            let mockLlm = createMockLlm (fun _ -> "  Learn Graph Algorithms  ")
            let governor = EpistemicGovernor(mockLlm, None, None) :> IEpistemicGovernor

            let! suggestion = governor.SuggestCurriculum([ "Task 1" ], [ "Belief 1" ], false)

            Assert.Equal("Learn Graph Algorithms", suggestion)
        }

    [<Fact>]
    let ``Verify returns true when LLM says VERIFIED`` () =
        task {
            let mockLlm = createMockLlm (fun _ -> "This statement is VERIFIED.")
            let governor = EpistemicGovernor(mockLlm, None, None) :> IEpistemicGovernor

            let! result = governor.Verify("Sky is blue")

            Assert.True(result)
        }

    [<Fact>]
    let ``Verify returns false when LLM says REJECTED`` () =
        task {
            let mockLlm = createMockLlm (fun _ -> "This statement is REJECTED.")
            let governor = EpistemicGovernor(mockLlm, None, None) :> IEpistemicGovernor

            let! result = governor.Verify("Sky is green")

            Assert.False(result)
        }

    [<Fact>]
    let ``GetRelatedCodeContext returns message when graph is missing`` () =
        task {
            let mockLlm = createMockLlm (fun _ -> "")
            let governor = EpistemicGovernor(mockLlm, None, None) :> IEpistemicGovernor

            let! context = governor.GetRelatedCodeContext("search")

            Assert.Equal("No knowledge graph available.", context)
        }
