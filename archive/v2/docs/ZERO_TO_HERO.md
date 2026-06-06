# From Zero to Hero: F# and TARS

A practical guide that teaches you F# through building and using TARS — a neuro-symbolic self-improving AI agent system.

---

## Part 1: F# Fundamentals (The Language TARS Thinks In)

### Why F#?

TARS is written in F# because the language naturally expresses what AI agents need:
- **Discriminated unions** model the branching decisions agents make
- **Pattern matching** handles complex control flow without bugs
- **Immutable-by-default** prevents agents from corrupting their own state
- **Pipeline operators** read like reasoning chains
- **Computation expressions** make async agent workflows elegant
- **Type inference** keeps code concise without losing safety

F# runs on .NET, so you get the entire .NET ecosystem (NuGet packages, cross-platform, high performance).

### Setup

```bash
# Install .NET 10 SDK
# Windows: winget install Microsoft.DotNet.SDK.10
# macOS: brew install dotnet@10
# Linux: see https://dot.net/install

# Verify
dotnet --version   # Should show 10.x.x

# Clone TARS
git clone https://github.com/GuitarAlchemist/tars.git
cd tars/v2

# Build
dotnet build

# Run tests (should show 695+ passing)
dotnet test
```

### F# in 15 Minutes

Open any `.fs` file in TARS and you'll see these patterns everywhere.

#### Values and Functions

```fsharp
// Values are immutable by default
let name = "TARS"
let version = 2

// Functions are just values
let greet who = sprintf "Hello, %s!" who
greet "world"  // "Hello, world!"

// Multi-parameter functions (curried)
let add x y = x + y
let add5 = add 5       // Partial application
add5 3                  // 8

// Pipeline operator: pass result left-to-right
"hello" |> String.length |> sprintf "Length: %d"
// Same as: sprintf "Length: %d" (String.length "hello")
```

**Where you'll see this in TARS:** Every module uses pipelines. Look at `PatternSelector.fs` — pattern scores flow through `|> List.sortByDescending snd |> List.head`.

#### Discriminated Unions (The Heart of TARS)

```fsharp
// A type that can be one of several cases
type PatternKind =
    | ChainOfThought
    | ReAct
    | TreeOfThoughts
    | GraphOfThoughts
    | WorkflowOfThought
    | Custom of string

// Each case can carry different data
type FailureRootCause =
    | MissingTool of toolName: string
    | WrongPattern of used: string * suggested: string
    | KnowledgeGap of domain: string
    | Unknown of description: string
```

**Where you'll see this:** Open `src/Tars.Core/MetaCognition/MetaCognitionTypes.fs` — the entire meta-cognitive system is modeled as DUs. `PatternKind`, `FailureRootCause`, `GapRemedy`, `AdaptationSignal`, `AdaptationAction` are all DUs.

#### Pattern Matching (How TARS Decides)

```fsharp
// Exhaustive matching — compiler catches missing cases
let describe pattern =
    match pattern with
    | ChainOfThought -> "Step-by-step reasoning"
    | ReAct -> "Reason-Act loop with tools"
    | TreeOfThoughts -> "Explore multiple paths"
    | GraphOfThoughts -> "Connect ideas in a graph"
    | WorkflowOfThought -> "Structured workflow"
    | Custom name -> sprintf "Custom: %s" name

// Matching with guards and destructuring
let classifyGap gap =
    match gap with
    | { FailureRate = r } when r > 0.8 -> "Critical"
    | { FailureRate = r } when r > 0.5 -> "Moderate"
    | _ -> "Minor"
```

**Where you'll see this:** `MetaCommand.fs` line 101 matches `FailureRootCause` cases to display cluster info. `PatternSelector.fs` matches `PatternKind` to compile WoT plans.

#### Records (TARS's Data Structures)

```fsharp
// Immutable data containers with named fields
type PatternOutcome =
    { PatternKind: PatternKind
      Goal: string
      Success: bool
      DurationMs: int64
      Timestamp: DateTime }

// Create
let outcome =
    { PatternKind = ChainOfThought
      Goal = "Analyze code"
      Success = true
      DurationMs = 1500L
      Timestamp = DateTime.UtcNow }

// Update (creates a copy — original unchanged)
let failed = { outcome with Success = false; DurationMs = 30000L }
```

**Where you'll see this:** Every file. `LlmRequest`, `LlmResponse`, `WoTPlan`, `CapabilityGap`, `PatternGenome` — all records.

#### Lists and Pipelines (TARS's Data Flow)

```fsharp
let outcomes = [ outcome1; outcome2; outcome3 ]

// Pipeline processing — reads like English
let successRate =
    outcomes
    |> List.filter (fun o -> o.Success)
    |> List.length
    |> fun successes -> float successes / float outcomes.Length

// Group, sort, transform
let byPattern =
    outcomes
    |> List.groupBy (fun o -> o.PatternKind)
    |> List.sortByDescending (fun (_, items) -> items.Length)
    |> List.map (fun (kind, items) ->
        kind, items.Length, items |> List.filter (fun o -> o.Success) |> List.length)
```

**Where you'll see this:** `MetaCommand.fs` line 47 groups outcomes by pattern. `GapDetection.fs` pipelines failures through clustering, gap detection, and ranking.

#### Modules (How TARS Organizes Code)

```fsharp
// A module is a namespace for related functions
module GapDetection =

    let extractDomainTags (goal: string) : string list =
        // ...

    let detectGaps threshold clusters successes failures =
        // ...

    let rankGaps (gaps: CapabilityGap list) =
        gaps |> List.sortByDescending (fun g -> g.FailureRate * float g.SampleSize)
```

**Where you'll see this:** Every TARS file is a module. `FailureClustering`, `GapDetection`, `AdaptiveSignals`, `MachinBridge`, `RalphBridge` — each a focused module.

#### Async and Task (How TARS Calls LLMs)

```fsharp
// Async computation expression
let fetchAndAnalyze (url: string) =
    async {
        let! data = httpClient.GetStringAsync(url) |> Async.AwaitTask
        let parsed = JsonDocument.Parse(data)
        return parsed.RootElement
    }

// Task computation expression (interop with .NET)
let completeAsync (req: LlmRequest) =
    task {
        let prompt = buildPrompt req
        let! result = executeClaudeProcess config prompt
        match result with
        | Ok output -> return parseResponse output
        | Error err -> return { Text = err; FinishReason = Some "error"; Usage = None; Raw = None }
    }
```

**Where you'll see this:** `ClaudeCodeService.fs` uses `task {}` for subprocess calls. `TarsWoTAgent.fs` uses `async {}` for WoT execution. `RetroactionLoop.fs` uses `async {}` for the learning cycle.

#### Interfaces and Object Expressions (TARS's Extension Points)

```fsharp
// Interface definition
type ILlmService =
    abstract CompleteAsync: LlmRequest -> Task<LlmResponse>
    abstract EmbedAsync: string -> Task<float32[]>
    abstract RouteAsync: LlmRequest -> Task<RoutedBackend>

// Implementation via class
type ClaudeCodeLlmService(config) =
    interface ILlmService with
        member _.CompleteAsync(req) = task { (* ... *) }
        member _.EmbedAsync(_) = Task.FromResult(Array.empty)
        member _.RouteAsync(_) = task { (* ... *) }
```

**Where you'll see this:** `ILlmService` in `Tars.Llm`, `IWoTExecutor` in `Tars.Cortex`, `IPatternSelector` for pattern selection, `IVectorStore` for storage backends.

---

## Part 2: TARS Architecture (How It All Fits Together)

### The Layer Cake

```
┌─────────────────────────────────────────────────────────┐
│  CLI Layer (Tars.Interface.Cli)                         │
│  Program.fs → Commands/ (ask, chat, evolve, breed...)   │
├─────────────────────────────────────────────────────────┤
│  Agent Layer (Tars.Cortex)                              │
│  TarsWoTAgent, PatternSelector, AgentOrchestrator       │
│  AdaptiveExecutor, RalphBridge, ClaudeCodeBridge        │
├─────────────────────────────────────────────────────────┤
│  Evolution Layer (Tars.Evolution)                       │
│  RetroactionLoop, PromotionPipeline, MachinBridge       │
│  EvolutionaryPatternBreeder, MetaCognitionOrchestrator  │
├─────────────────────────────────────────────────────────┤
│  Reasoning Layer (Tars.Core)                            │
│  WorkflowOfThought types, MetaCognition types           │
│  AgentWorkflow, BudgetGovernance, Constitution          │
├─────────────────────────────────────────────────────────┤
│  LLM Layer (Tars.Llm)                                  │
│  ILlmService, Routing, ClaudeCodeService                │
│  DefaultLlmService (Ollama/OpenAI/Anthropic/vLLM)       │
├─────────────────────────────────────────────────────────┤
│  Foundation (Tars.Kernel, Tars.Tools, Tars.Knowledge)   │
│  EventBus, ToolRegistry, KnowledgeLedger, VectorStores  │
└─────────────────────────────────────────────────────────┘
```

### Key Concept: Workflow of Thought (WoT)

WoT is TARS's unified reasoning framework. Every reasoning pattern (CoT, ReAct, ToT, GoT) compiles down to a **WoTPlan** — a directed acyclic graph of reasoning nodes:

```fsharp
type WoTPlan =
    { Id: Guid
      Nodes: WoTNode list          // Reasoning steps
      Edges: WoTEdge list          // Dependencies between steps
      EntryNode: string            // Where execution starts
      Metadata: PlanMetadata       // Pattern kind, goal, estimates
      Policy: WoTPolicy list }     // Runtime constraints

type WoTNode =
    { Id: string
      Kind: string                 // "Reason", "Act", "Synthesize", "Branch"
      Prompt: string               // What the LLM sees
      Dependencies: string list }  // Must complete before this node runs
```

The **WoTExecutor** walks the DAG, calling the LLM at each node, collecting results into a **WoTTrace** (the execution log).

### Key Concept: Retroaction Loop

The self-improvement feedback cycle:

```
Problem → Pattern Match → Execute → Score → Learn
   ↑                                           │
   └───────── New Pattern ←────────────────────┘
```

1. **Get a problem** from the curriculum
2. **Match** it against the pattern library
3. **Execute** using the matched pattern (or default reasoning)
4. **Score** the result (validation, confidence, efficiency)
5. **If good enough**, compile a new pattern from the execution trace
6. **Amplify** — mutate the pattern to create variants, keep improvements
7. **Promote** — patterns that recur climb the staircase (Implementation → Helper → Builder → DslClause → GrammarRule)

### Key Concept: Meta-Cognition

TARS can analyze its own execution history to find what it's bad at:

```
Execution History → Failure Clustering → Gap Detection → Curriculum → Fix
```

- **FailureClustering** groups similar failures using Jaccard similarity
- **GapDetection** identifies domains with high failure rates
- **CurriculumPlanner** generates targeted tasks to address gaps
- **ReflectionEngine** compares intent vs outcome after each run

### Key Concept: Evolutionary Pattern Breeding

Real genetic algorithms optimize TARS's pattern hyperparameters:

```fsharp
type PatternGenome =
    { CotWeight: float          // How much to favor Chain-of-Thought
      ReactWeight: float        // How much to favor ReAct
      TotWeight: float          // How much to favor Tree-of-Thoughts
      GotWeight: float          // How much to favor Graph-of-Thoughts
      StepMultiplier: float     // How many reasoning steps
      Temperature: float        // LLM creativity level
      ConfidenceThreshold: float // When to stop early
      BranchingFactor: float }  // How wide to explore
```

The GA evolves these using execution history as fitness. It uses ix's Rust GA when available, or a built-in F# GA.

---

## Part 3: Hands-On (Using TARS)

### Level 1: Basic Commands

```bash
cd tars/v2

# Check system health
dotnet run --project src/Tars.Interface.Cli -- diag

# Ask a question (requires Ollama or Claude Code)
dotnet run --project src/Tars.Interface.Cli -- ask "What is F#?"

# Interactive chat
dotnet run --project src/Tars.Interface.Cli -- chat
```

### Level 2: Reasoning Patterns

```bash
# Chain of Thought — step-by-step reasoning
dotnet run --project src/Tars.Interface.Cli -- agent cot "Explain monads"

# ReAct — reasoning with tool use
dotnet run --project src/Tars.Interface.Cli -- agent react "Find files with TODO comments"

# Tree of Thoughts — explore multiple approaches
dotnet run --project src/Tars.Interface.Cli -- agent tot "Design a cache eviction strategy"

# Graph of Thoughts — connect ideas
dotnet run --project src/Tars.Interface.Cli -- agent got "Compare REST vs GraphQL"

# Workflow of Thoughts — structured multi-step
dotnet run --project src/Tars.Interface.Cli -- agent wot "Refactor this module for testability"

# Full MAF orchestration (auto-selects pattern)
dotnet run --project src/Tars.Interface.Cli -- agent run "Analyze the codebase for security issues"
```

### Level 3: Evolution

```bash
# Run the evolution engine (5 iterations)
dotnet run --project src/Tars.Interface.Cli -- evolve --max-iterations 5

# Run with tracing and budget
dotnet run --project src/Tars.Interface.Cli -- evolve --trace --budget 0.50

# Run N full cycles back-to-back
dotnet run --project src/Tars.Interface.Cli -- evolve --loop 3 --max-iterations 5

# Check what TARS learned
dotnet run --project src/Tars.Interface.Cli -- promote status
dotnet run --project src/Tars.Interface.Cli -- promote lineage
```

### Level 4: Meta-Cognition

```bash
# See execution statistics
dotnet run --project src/Tars.Interface.Cli -- meta stats

# Detect capability gaps
dotnet run --project src/Tars.Interface.Cli -- meta gaps

# Show failure clusters
dotnet run --project src/Tars.Interface.Cli -- meta clusters

# Generate targeted learning tasks
dotnet run --project src/Tars.Interface.Cli -- meta curriculum

# Full analysis cycle
dotnet run --project src/Tars.Interface.Cli -- meta analyze

# With Claude Code for LLM-enhanced analysis
dotnet run --project src/Tars.Interface.Cli -- meta analyze --use-claude
```

### Level 5: Evolutionary Breeding

```bash
# Run GA on pattern hyperparameters
dotnet run --project src/Tars.Interface.Cli -- breed

# With genome details
dotnet run --project src/Tars.Interface.Cli -- breed --show-genome --generations 100

# Check ix bridge
dotnet run --project src/Tars.Interface.Cli -- breed status
```

### Level 6: Ralph Loops (Iterative Self-Improvement)

```bash
# Preview what Ralph would work on
dotnet run --project src/Tars.Interface.Cli -- ralph prompt

# Start a gap-driven improvement loop
dotnet run --project src/Tars.Interface.Cli -- ralph start --max-iterations 10

# Start a focused loop
dotnet run --project src/Tars.Interface.Cli -- ralph start --focus "ReAct pattern"

# Start a task-specific loop
dotnet run --project src/Tars.Interface.Cli -- ralph start --goal "Add caching to the LLM layer"

# Check loop status
dotnet run --project src/Tars.Interface.Cli -- ralph status

# Stop the loop
dotnet run --project src/Tars.Interface.Cli -- ralph stop
```

### Level 7: Knowledge Management

```bash
# Knowledge Ledger status
dotnet run --project src/Tars.Interface.Cli -- know status

# Assert a belief
dotnet run --project src/Tars.Interface.Cli -- know assert "F#" "is-a" "programming language"

# Fetch knowledge from Wikipedia
dotnet run --project src/Tars.Interface.Cli -- know fetch "functional programming"

# Trace execution through knowledge graph
dotnet run --project src/Tars.Interface.Cli -- kg trace
```

---

## Part 4: Extending TARS (Writing F# for TARS)

### Adding a New CLI Command

1. Create `src/Tars.Interface.Cli/Commands/MyCommand.fs`:

```fsharp
namespace Tars.Interface.Cli.Commands

open Spectre.Console

module MyCommand =

    let run (args: string list) =
        AnsiConsole.MarkupLine("[bold cyan]My Command[/]")

        match args with
        | "hello" :: name :: _ ->
            AnsiConsole.MarkupLine(sprintf "Hello, [green]%s[/]!" name)
            0
        | _ ->
            AnsiConsole.MarkupLine("Usage: tars my-cmd hello <name>")
            0
```

2. Add to `Tars.Interface.Cli.fsproj` (order matters in F#!):
```xml
<Compile Include="Commands\MyCommand.fs" />
```

3. Wire into `Program.fs`:
```fsharp
| args when args.Length > 0 && args.[0] = "my-cmd" ->
    let subArgs = args |> Array.skip 1 |> Array.toList
    return MyCommand.run subArgs
```

### Adding a New Tool

Tools are functions that agents can call during execution.

```fsharp
// In src/Tars.Tools/MyTools.fs
namespace Tars.Tools

open Tars.Tools.Domain

module MyTools =

    [<TarsTool("word_count", "Count words in text")>]
    let wordCount (input: string) =
        async {
            let count = input.Split(' ') |> Array.length
            return Ok (sprintf "Word count: %d" count)
        }
```

Register in the tool registry and agents can use it during ReAct loops.

### Adding a New Reasoning Pattern

1. Define the pattern compilation in `PatternCompiler.fs`:

```fsharp
member this.CompileMyPattern(goal: string) : WoTPlan =
    let nodes =
        [ { Id = "decompose"; Kind = "Reason"
            Prompt = sprintf "Break this goal into sub-goals: %s" goal
            Dependencies = [] }
          { Id = "solve"; Kind = "Act"
            Prompt = "Solve each sub-goal"
            Dependencies = ["decompose"] }
          { Id = "synthesize"; Kind = "Reason"
            Prompt = "Combine the sub-solutions"
            Dependencies = ["solve"] } ]
    // ... build WoTPlan with nodes, edges, metadata
```

2. Add the pattern kind to `PatternKind` DU in `WoTTypes.fs`
3. Add a case to the pattern selector in `TarsWoTAgent.fs`

### Writing Tests

TARS uses xUnit. Tests go in `tests/Tars.Tests/`:

```fsharp
namespace Tars.Tests

open Xunit

module MyTests =

    [<Fact>]
    let ``my function does the right thing`` () =
        let result = MyModule.myFunction "input"
        Assert.Equal("expected", result)

    [<Fact>]
    let ``handles edge case`` () =
        let result = MyModule.myFunction ""
        Assert.True(result.Length = 0)

    [<Theory>]
    [<InlineData("a", 1)>]
    [<InlineData("a b", 2)>]
    [<InlineData("a b c", 3)>]
    let ``word count is correct`` (input: string) (expected: int) =
        let count = input.Split(' ') |> Array.length
        Assert.Equal(expected, count)
```

Add the file to `Tars.Tests.fsproj` and run with `dotnet test`.

---

## Part 5: The Big Picture (How TARS Improves Itself)

### The Self-Improvement Stack

```
Layer 5: Ralph Loop
  └─ Iterates the entire improvement cycle from a fixed prompt
  └─ Each iteration sees previous work in files/git

Layer 4: Evolutionary Breeding (GA)
  └─ Optimizes pattern hyperparameters using execution history
  └─ Uses ix's Rust GA or built-in F# fallback

Layer 3: Meta-Cognition
  └─ Identifies capability gaps from failure clustering
  └─ Generates targeted curriculum to address gaps
  └─ Reflects on intent vs outcome

Layer 2: Retroaction Loop
  └─ Scores each execution
  └─ Compiles successful traces into patterns
  └─ Amplifies patterns through LLM mutation
  └─ Promotes recurring patterns up the staircase

Layer 1: WoT Execution
  └─ Pattern selector chooses best reasoning approach
  └─ WoT executor runs the DAG plan
  └─ Records outcome to PatternOutcomeStore
```

Each layer feeds the layers above it. Execution history (Layer 1) feeds the retroaction loop (Layer 2), which feeds meta-cognition (Layer 3), which feeds GA breeding (Layer 4), which can be wrapped in a Ralph loop (Layer 5) for continuous iteration.

### The Three-Repo Ecosystem

TARS doesn't work alone:

| Repo | Language | Role |
|------|----------|------|
| **TARS** | F# | Agent runtime, evolution, meta-cognition |
| **ix** | Rust | ML algorithms (GA, PSO, MCTS, clustering, FFT) |
| **Guitar Alchemist** | C#/F# | Music domain, spectral analysis, DSL patterns |

Connected via MCP (Model Context Protocol) — each repo exposes tools the others can call.

### What Makes TARS Different

Most AI agent frameworks are **stateless** — they solve a task and forget. TARS is **cumulative**:

1. **It remembers being wrong** — PatternOutcomeStore tracks every execution
2. **It learns from failure** — MetaCognition clusters failures and detects gaps
3. **It breeds better strategies** — GA evolves pattern weights from real data
4. **It promotes what works** — Patterns climb from raw code to grammar rules
5. **It iterates relentlessly** — Ralph loops keep hammering until gaps close

The philosophy: *LLMs are stochastic generators. Make them useful by wrapping them in deterministic feedback loops that remember, learn, and adapt.*

---

## Quick Reference

### Project Structure
```
v2/
├── src/
│   ├── Tars.Core/           # Types, WoT, MetaCognition, AgentWorkflow
│   ├── Tars.Cortex/         # Agent runtime, executors, vector stores
│   ├── Tars.Evolution/      # RetroactionLoop, promotion, GA, MachinBridge
│   ├── Tars.Llm/            # LLM abstraction, routing, Claude Code adapter
│   ├── Tars.Kernel/         # EventBus, infrastructure
│   ├── Tars.Tools/          # 150+ tools for agent use
│   ├── Tars.Knowledge/      # Knowledge ledger, linked data
│   ├── Tars.DSL/            # WoT DSL parser
│   ├── Tars.Graph/          # DAG executor
│   ├── Tars.Security/       # Credential vault
│   └── Tars.Interface.Cli/  # CLI commands
├── tests/
│   └── Tars.Tests/          # 695+ tests
└── docs/                    # Documentation
```

### F# Cheat Sheet (TARS Edition)
```fsharp
// Pipeline
data |> transform |> filter |> result

// Pattern match
match x with
| Case1 value -> handle value
| Case2(a, b) -> combine a b
| _ -> default

// Record update
{ existing with Field = newValue }

// Async
async { let! result = someAsyncOp(); return result }

// Task (for .NET interop)
task { let! result = someTask(); return result }

// List comprehension
[ for i in 1..10 do if i % 2 = 0 then yield i * i ]

// Option handling
match maybeValue with
| Some v -> use v
| None -> handle missing

// Result handling
match operation() with
| Ok value -> succeed value
| Error msg -> fail msg
```

### Common Commands
```bash
dotnet build                    # Build everything
dotnet test                     # Run all tests
dotnet run --project src/Tars.Interface.Cli -- <command>

# Shortcuts (if you have the tars alias):
tars diag                       # System diagnostics
tars agent run "goal"           # Run agent on a goal
tars evolve --loop 3            # 3 evolution cycles
tars meta analyze               # Meta-cognitive analysis
tars breed --show-genome        # Evolve pattern weights
tars ralph start                # Start improvement loop
```
