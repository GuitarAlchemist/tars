module Tars.Tests.EvolutionTaskGateTests

open System
open Xunit
open Tars.Core
open Tars.Knowledge
open Tars.Evolution

// Issue #245: the ledger-contradiction gate and the prompt's memory/belief sections were
// inlined in Engine.executeTask, reachable only by running a full evolution task.

let private belief subject obj confidence =
    { Belief.fromTriple subject (RelationType.Custom "uses") obj with
        Confidence = confidence }

[<Fact>]
let ``selectRelevantBeliefs keeps beliefs named in the goal, most confident first, at most five`` () =
    let beliefs =
        [ belief "Parser" "Grammar" 0.4
          belief "Router" "Cache" 0.9 // unrelated to the goal
          belief "LEXER" "Tokens" 0.8 // subject matches case-insensitively
          belief "Engine" "parser" 0.6 ] // object matches
        @ [ for i in 1..5 -> belief $"parser{i}" "x" (float i / 10.0) ]

    let selected = Engine.selectRelevantBeliefs "Fix the parser and the lexer" beliefs

    Assert.Equal(5, selected.Length)
    Assert.DoesNotContain(selected, fun b -> b.Subject.Value = "Router")
    let confidences = selected |> List.map (fun b -> b.Confidence)
    Assert.True(([ 0.8; 0.6; 0.5; 0.4; 0.4 ] = confidences), $"%A{confidences}")

[<Theory>]
[<InlineData("check_contradictions", true)>]
[<InlineData("LEDGER_GATE", true)>]
[<InlineData("enforce_ledger", true)>]
[<InlineData("no_gate_requested", false)>]
let ``shouldGateContradictions requires an opt-in constraint`` (name: string) (expected: bool) =
    Assert.Equal(expected, Engine.shouldGateContradictions [ name ] [ belief "a" "b" 0.5 ])

[<Fact>]
let ``shouldGateContradictions honours the allow_contradictions opt-out`` () =
    Assert.False(
        Engine.shouldGateContradictions [ "check_contradictions"; "Allow_Contradictions" ] [ belief "a" "b" 0.5 ]
    )

[<Fact>]
let ``shouldGateContradictions does not gate when no belief is relevant`` () =
    Assert.False(Engine.shouldGateContradictions [ "check_contradictions" ] [])

[<Fact>]
let ``formatLedgerContext lists beliefs and is empty without any`` () =
    Assert.Equal("", Engine.formatLedgerContext [])

    // Confidence is formatted with the current culture, so only the rest is pinned here.
    let context = Engine.formatLedgerContext [ belief "Parser" "Grammar" 0.75 ]
    Assert.StartsWith("\nKnown Beliefs:\n- [", context)
    Assert.EndsWith("] Parser uses Grammar\n", context)

[<Fact>]
let ``formatMemoryContext summarises episodes and defaults missing logical memory`` () =
    let episode logical : MemorySchema =
        { Id = "m"
          Logical = logical
          Perceptual = None
          CreatedAt = DateTime.UtcNow
          LastUsedAt = None
          UsageCount = 0 }

    Assert.Equal("", Engine.formatMemoryContext [])

    let learned: LogicalMemory =
        { ProblemSummary = "Parser dropped trailing comments"
          StrategySummary = ""
          ErrorKinds = []
          ErrorTags = []
          OutcomeLabel = "failure"
          Score = None
          CostTokens = None
          Embedding = [||]
          Tags = [] }

    Assert.Equal(
        "\nLessons Learned from Past Episodes:\n- [failure] Parser dropped trailing comments\n- [unknown] Unknown Task\n",
        Engine.formatMemoryContext [ episode (Some learned); episode None ]
    )

// In every cycle of the 2026-10-01 evolve baseline the curriculum agent answered in
// prose ("Please provide the current context..."), and each prose answer was replaced by
// a canned task: the schema-constrained call only ran when the agent said nothing.

[<Fact>]
let ``a prose curriculum answer is not task JSON, so it gets the schema-constrained call`` () =
    Assert.False(
        Engine.curriculumAnswerIsJson
            "Please provide the current context or any specific areas where you need assistance, and I will generate the next F# coding task accordingly."
    )

    Assert.False(Engine.curriculumAnswerIsJson "")
    Assert.False(Engine.curriculumAnswerIsJson "   ")

[<Fact>]
let ``a task list answer, bare or fenced, is used as it is`` () =
    let tasks =
        """{"tasks":[{"goal":"Write a parser in F#","constraints":["pure"],"validation_criteria":"tests pass"}]}"""

    Assert.True(Engine.curriculumAnswerIsJson tasks)
    Assert.True(Engine.curriculumAnswerIsJson("Here are the tasks:\n```json\n" + tasks + "\n```"))
