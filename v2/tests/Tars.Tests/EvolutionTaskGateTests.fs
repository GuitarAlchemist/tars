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
let ``a prose curriculum answer holds no task list, so it gets the schema-constrained call`` () =
    Assert.False(
        Engine.curriculumAnswerHasTasks
            "Please provide the current context or any specific areas where you need assistance, and I will generate the next F# coding task accordingly."
    )

    Assert.False(Engine.curriculumAnswerHasTasks "")
    Assert.False(Engine.curriculumAnswerHasTasks "   ")

[<Fact>]
let ``JSON of another shape holds no task list either`` () =
    // Each of these parses as JSON, but the task parser would find no task in it, or
    // throw reading one, and fall back to a canned task.
    Assert.False(Engine.curriculumAnswerHasTasks """{"message":"Please provide the current context"}""")
    Assert.False(Engine.curriculumAnswerHasTasks """{"tasks":[]}""")
    Assert.False(Engine.curriculumAnswerHasTasks """{"tasks":[{"goal":"Write a parser in F#"}]}""")

    Assert.False(
        Engine.curriculumAnswerHasTasks
            """{"tasks":[{"goal":"Write a parser in F#","constraints":[1],"validation_criteria":"tests pass"}]}"""
    )

[<Fact>]
let ``a task list answer, bare, wrapped or fenced, is used as it is`` () =
    let task =
        """{"goal":"Write a parser in F#","constraints":["pure"],"validation_criteria":"tests pass"}"""

    let tasks = $"""{{"tasks":[{task}]}}"""
    Assert.True(Engine.curriculumAnswerHasTasks tasks)
    Assert.True(Engine.curriculumAnswerHasTasks $"[{task}]")
    Assert.True(Engine.curriculumAnswerHasTasks("Here are the tasks:\n```json\n" + tasks + "\n```"))

[<Fact>]
let ``the task prompt states the task and leaves the tool list to the executor`` () =
    // In evolve, the task prompt listed the whole tool registry: 17.7 KB of its 22 KB,
    // past the 4k context window. It was summarized, and the executor never saw its goal.
    let taskDef =
        { Id = Guid.NewGuid()
          DifficultyLevel = 1
          Goal = "Write a function in F# that checks if a given string is a palindrome."
          Constraints = [ "Use recursion" ]
          ValidationCriteria = "racecar is a palindrome, tars is not"
          Timeout = TimeSpan.FromMinutes 1.0
          Score = 0.0 }

    let prompt = Engine.buildTaskPrompt taskDef "Related Code Structure: none" "" ""

    Assert.Contains(taskDef.Goal, prompt)
    Assert.Contains("Use recursion", prompt)
    Assert.Contains(taskDef.ValidationCriteria, prompt)
    Assert.Contains("Related Code Structure: none", prompt)
    Assert.DoesNotContain("[AVAILABLE TOOLS]", prompt)

[<Fact>]
let ``an answer that only talks about the code shows no code`` () =
    // Answers the evolve executor gave in place of a solution.
    Assert.False(Engine.answerHasCode "ACT: REQUEST: Please provide the current project structure so I can understand the file paths.")
    Assert.False(Engine.answerHasCode "ACT: INFORM: I will create a function to reverse a string in F# using immutable data structures.")
    Assert.False(Engine.answerHasCode "")
    // A tool call is not code shown in the answer.
    Assert.False(Engine.answerHasCode "```tool\n{\"name\": \"plan_task\", \"arguments\": {}}\n```")

[<Fact>]
let ``an answer with a fenced code block shows code`` () =
    Assert.True(Engine.answerHasCode "Here it is:\n```fsharp\nlet rec fact n = if n <= 1 then 1 else n * fact (n - 1)\n```")
    Assert.True(Engine.answerHasCode "```\nlet x = 1\n```")
    Assert.True(Engine.answerHasCode "```tool\n{}\n```\nDone:\n```fsharp\nlet x = 1\n```")

[<Fact>]
let ``a task that makes code asks for code`` () =
    // Goals the curriculum generated in evolve runs, and the one canned task that builds something.
    for goal in
        [ "Implement a function to calculate the nth Fibonacci number in F#."
          "Write a function in F# that checks if a given string is a palindrome."
          "Refactor the existing `fibonacci` function to use memoization for improved performance."
          "Create a tool for analyzing F# code quality using a static code analysis library."
          "Failed to generate novel tasks. Refactor the existing code for better maintainability."
          "Create a new dynamic tool named 'check_todo' that searches the project for 'TODO' comments and returns a formatted list." ] do
        Assert.True(Engine.taskAsksForCode goal, goal)

[<Fact>]
let ``a task that analyzes or summarizes does not ask for code`` () =
    // The other canned tasks, and a topic the epistemic governor proposed: prose is the answer.
    for goal in
        [ "Scan the src/Tars.Tools directory and identify 2 tools that lack proper error handling in their JSON parsing logic."
          "Analyze the current Evolution Engine loop in src/Tars.Evolution/Engine.fs and suggest a way to implement better task pivoting after 3 failures."
          "Read src/Tars.Core/Domain.fs and write a summary of the 'AgentIntent' discriminated union."
          "List all files in src/Tars.Evolution and summarize the responsibility of each file."
          "Explore the foundational principles of epistemology." ] do
        Assert.False(Engine.taskAsksForCode goal, goal)

[<Fact>]
let ``the script of an answer is its F# code and nothing else`` () =
    Assert.Equal(None, Engine.scriptOfAnswer "I will write a recursive factorial function in F#.")
    Assert.Equal(None, Engine.scriptOfAnswer "```tool\n{\"name\": \"write_code\", \"arguments\": {}}\n```")
    // An unlabeled block may be program output rather than code.
    Assert.Equal(None, Engine.scriptOfAnswer "Output:\n```\n[1; 2; 3]\n```")
    // A tool call the executor put in an F# block, from a live evolve run.
    Assert.Equal(
        None,
        Engine.scriptOfAnswer "ACT: INFORM: ```fsharp\n{\n  \"name\": \"read_code\",\n  \"arguments\": {\n    \"path\": \"Program.fs\"\n  }\n}\n```"
    )

    Assert.Equal(
        Some "let x = 1\n\nprintfn \"%d\" x",
        Engine.scriptOfAnswer "```tool\n{}\n```\n```fsharp\nlet x = 1\n```\nThen:\n```fsharp\nprintfn \"%d\" x\n```"
    )

[<Fact>]
let ``code that uses TARS's own projects is not run as a script`` () =
    Assert.Equal(None, Engine.scriptOfAnswer "```fsharp\nopen Tars.Core\nlet x = 1\n```")
    Assert.Equal(None, Engine.scriptOfAnswer "```fsharp\nlet run = Tars.Evolution.Engine.step\n```")
    // Mentioning TARS in a comment does not make the code need it.
    Assert.Equal(
        Some "// Like Tars.Core.Domain, but on its own.\nlet x = 1",
        Engine.scriptOfAnswer "```fsharp\n// Like Tars.Core.Domain, but on its own.\nlet x = 1\n```"
    )

[<Fact>]
let ``an answer written as a .fs file runs as a script`` () =
    task {
        // dotnet fsi rejects a namespace line and a top-level module declaration, which
        // are valid, and common, at the top of a .fs file.
        let answer =
            "```fsharp\nnamespace Demo\n\nmodule Fact\n\nlet rec fact n = if n <= 1 then 1 else n * fact (n - 1)\nprintfn \"%d\" (fact 5)\n```\n"
            + "```fsharp\nprintfn \"%d\" (Fact.fact 6)\n```"

        let! run = Engine.runScript (TimeSpan.FromSeconds 60.0) (Engine.scriptOfAnswer answer).Value

        Assert.Equal(Result.Ok(), run)
    }

[<Fact>]
let ``a script that does not compile, throws or never ends fails with what went wrong`` () =
    task {
        let! notCompiling = Engine.runScript (TimeSpan.FromSeconds 60.0) "let x: int = \"a\""
        let! throwing =
            Engine.runScript
                (TimeSpan.FromSeconds 60.0)
                "let f n = if n < 0 then failwith \"Negative input not allowed\" else n\nprintfn \"%d\" (f -1)"
        let! endless = Engine.runScript (TimeSpan.FromSeconds 3.0) "while true do ()"

        match notCompiling, throwing, endless with
        | Result.Error compile, Result.Error thrown, Result.Error timedOut ->
            Assert.Contains("error FS0001", compile)
            Assert.Contains("Negative input not allowed", thrown)
            Assert.Contains("did not finish", timedOut)
        | other -> Assert.Fail $"%A{other}"
    }

let private taskWith goal constraints =
    { Id = Guid.NewGuid()
      DifficultyLevel = 1
      Goal = goal
      Constraints = constraints
      ValidationCriteria = ""
      Timeout = TimeSpan.FromMinutes 1.0
      Score = 0.0 }

[<Fact>]
let ``a task about code it does not include, or about an unnamed tool, is not specified`` () =
    // Goals the curriculum generated, or fell back to, in live evolve runs. All of them failed:
    // the executor asked for the code or the specification, or refused.
    for goal in
        [ "Failed to generate novel tasks. Refactor the existing code for better readability."
          "Refactor an existing F# codebase to use pattern matching instead of if-else statements."
          "Refactor the provided F# code to use pattern matching instead of traditional conditional statements wherever possible."
          "Refactor the prime number checking function to use a different algorithm for better performance"
          "Refactor a given piece of code to use immutable data structures in F#"
          "Create a tool to analyze the performance of F# code snippets"
          "Create a tool for analyzing F# code complexity."
          "Create a tool for analyzing and refactoring F# code to improve readability." ] do
        Assert.False(Engine.taskIsSpecified (taskWith goal []), goal)

[<Fact>]
let ``a self-contained task, a refactor that includes its code, or a named tool is specified`` () =
    for goal in
        [ "Implement a function in F# that calculates the nth Fibonacci number using recursion."
          "Write a function in F# that checks if a given string is a palindrome."
          "Refactor this F# function to use pattern matching: `let sign x = if x > 0 then 1 elif x < 0 then -1 else 0`"
          "Create a new dynamic tool named 'check_todo' that searches the project for 'TODO' comments and returns a formatted list." ] do
        Assert.True(Engine.taskIsSpecified (taskWith goal []), goal)

    // The code to refactor may come in a constraint.
    Assert.True(
        Engine.taskIsSpecified (
            taskWith
                "Refactor this function to use pattern matching in F#."
                [ "The function: `let sign x = if x > 0 then 1 elif x < 0 then -1 else 0`" ]
        )
    )

[<Fact>]
let ``the concrete tasks are specified coding tasks with examples`` () =
    Assert.True(Engine.concreteTasks.Length >= 10)

    for goal, criteria in Engine.concreteTasks do
        Assert.True(Engine.taskIsSpecified (taskWith goal []), goal)
        Assert.True(Engine.taskAsksForCode goal, goal)
        Assert.Contains("in F#", goal)
        Assert.Contains(" = ", criteria)

[<Fact>]
let ``the next concrete task is the first one not done yet`` () =
    let first, _ = Engine.concreteTasks.[0]
    let second, _ = Engine.concreteTasks.[1]
    let all = Engine.concreteTasks |> List.map fst

    Assert.Equal(first, fst (Engine.nextConcreteTask []))
    Assert.Equal(second, fst (Engine.nextConcreteTask [ first.ToUpperInvariant() ]))
    // When every one is done, it still gives one.
    Assert.Contains(fst (Engine.nextConcreteTask all), all)
