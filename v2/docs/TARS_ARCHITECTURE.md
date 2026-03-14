# TARS: Typed Agent Reasoning System -- Architecture Reference

**Version**: v2 | **Language**: F# (.NET 9) | **Last Updated**: 2026-03-10

---

## 1. System Overview

TARS (Typed Agent Reasoning System) is an autonomous AI agent framework written in F# that combines large language models with symbolic reasoning, formal epistemology, and self-improving reasoning patterns. Its core innovation is **neuro-symbolic self-improvement via Workflow of Thought (WoT)**: every reasoning strategy (Chain of Thought, ReAct, Graph of Thoughts, Tree of Thoughts) compiles to a unified WoT graph that produces auditable, diffable execution traces. These traces feed a self-improvement loop where TARS evolves its own reasoning patterns based on measured fitness.

**Core Philosophy**:
- All reasoning must be **auditable** -- no opaque LLM calls.
- All patterns compile to the **same WoT intermediate representation** before execution.
- The system maintains a **formal belief store** with AGM-style revision and proof-backed evidence.
- Improvement is **measured** via fitness scoring and validated via Golden Traces (verified correct reasoning paths).
- Local-first LLM execution (Ollama, LlamaCpp, vLLM) with cloud fallback (OpenAI-compatible).

**Design Motto**: "A narrow victory is better than ten universal promises." -- consolidation over feature expansion.

---

## 2. Core Architecture

### 2.1 Project Structure

| Project | Purpose |
|---------|---------|
| **Tars.Core** | Domain types, abstractions, belief revision, proof system, WoT trace types, agent registry |
| **Tars.Llm** | LLM service interfaces and backends (Ollama, OpenAI, vLLM, LlamaCpp, Microsoft.Extensions.AI bridge) |
| **Tars.DSL** | WoT DSL grammar -- AST, parser, compiler for `.wot.trsx` workflow files |
| **Tars.Cortex** | Cognitive layer -- reasoning patterns, pattern compiler, WoT executor, pattern selector, epistemic governor |
| **Tars.Evolution** | Self-improvement -- pattern library, curriculum manager, fitness scoring, mutation engine, selection logic |
| **Tars.Tools** | 152+ tool implementations (Git, file, web, semantic, refactoring, reasoning, MCP, etc.) |
| **Tars.Connectors** | External integrations -- Ollama, OpenAI, Augment, Graphiti knowledge graph, MCP server bridges |
| **Tars.Knowledge** | Knowledge ledger, plan management, verifier agent, PostgreSQL storage |
| **Tars.Kernel** | Agent factory, governance, safety gate, semantic memory, registry |
| **Tars.Graph** | Graph execution engine for workflow DAGs |
| **Tars.Metascript** | TARS Metascript DSL engine (v1 parser, macro registry) |
| **Tars.LinkedData** | RDF/SPARQL integration, Fuseki triple store, linked data mapping |
| **Tars.Symbolic** | Constraint scoring for symbolic reasoning |
| **Tars.Sandbox** | Sandboxed code execution |
| **Tars.Interface.Cli** | CLI commands (`tars evolve`, `tars chat`, `tars agent`, `tars wot`, etc.) |
| **Tars.Interface.Ui** | GUI frontend |

### 2.2 Dependency Graph

```
Tars.Core (domain types, abstractions)
  |
  +-- Tars.Llm (LLM service, routing, MAF bridge)
  |     |
  |     +-- Tars.Cortex (patterns, compiler, WoT executor)
  |     |     |
  |     |     +-- Tars.Evolution (self-improvement loop)
  |     |
  |     +-- Tars.DSL (WoT grammar, parser)
  |
  +-- Tars.Tools (tool implementations)
  |
  +-- Tars.Connectors (Ollama, OpenAI, MCP, Graphiti)
  |
  +-- Tars.Knowledge (ledger, plans, verifier)
  |
  +-- Tars.Kernel (agent factory, governance)
        |
        +-- Tars.Interface.Cli (user-facing commands)
```

### 2.3 Key Interfaces

```fsharp
/// Core LLM abstraction -- all backends implement this
type ILlmService =
    abstract CompleteAsync: LlmRequest -> Task<LlmResponse>
    abstract EmbedAsync: text: string -> Task<float32[]>
    abstract CompleteStreamAsync: LlmRequest * (string -> unit) -> Task<LlmResponse>
    abstract RouteAsync: LlmRequest -> Task<RoutedBackend>

/// Tool registry -- 152+ tools registered here
type IToolRegistry =
    abstract Register: tool: Tool -> unit
    abstract Get: name: string -> Tool option
    abstract GetAll: unit -> Tool list

/// Compiles reasoning patterns to executable WoT plans
type IPatternCompiler =
    abstract CompileChainOfThought: steps: int * goal: string -> WoTPlan
    abstract CompileReAct: tools: string list * maxSteps: int * goal: string -> WoTPlan
    abstract CompileGraphOfThoughts: branchingFactor: int * maxDepth: int * goal: string -> WoTPlan
    abstract CompileTreeOfThoughts: beamWidth: int * searchDepth: int * goal: string -> WoTPlan
    abstract CompilePattern: pattern: ReasoningPattern * goal: string -> WoTPlan

/// Executes compiled WoT plans
type IWoTExecutor =
    abstract Execute: plan: WoTPlan * context: AgentContext -> Async<WoTResult>
    abstract ExecuteWithProgress: plan: WoTPlan * context: AgentContext * onProgress: (WoTTraceStep -> unit) -> Async<WoTResult>

/// Selects the optimal reasoning pattern for a goal
type IPatternSelector =
    abstract Recommend: goal: string * state: WoTCognitiveState -> PatternKind
    abstract Score: goal: string -> Map<PatternKind, float>
```

---

## 3. Workflow of Thought (WoT) DSL

The WoT DSL is TARS's unified intermediate representation. Every reasoning pattern compiles to a WoT graph before execution. The DSL is defined in `.wot.trsx` files and parsed into an AST.

### 3.1 Node Types

```fsharp
type NodeKind =
    | Reason   // LLM generation
    | Work     // Tool execution (side-effect)

type WoTNodeKind =
    | Reason   // Think / LLM call
    | Tool     // Act / external action
    | Validate // Symbolic check (invariants)
    | Memory   // Knowledge graph read/write
    | Control  // Branch / Loop / Parallel / Decide / Observe
```

### 3.2 GoT Transformations

```fsharp
type GoTTransformation =
    | Generate    // Create new thoughts
    | Aggregate   // Merge multiple thoughts
    | Refine      // Improve existing thought
    | Contradict  // Find contradictions
    | Distill     // Summarize/compress
    | Backtrack   // Undo failed path
    | Score       // Evaluate quality
```

### 3.3 Typed Edges

```fsharp
type EdgeRelation =
    | EdgeDependsOn    // Execution dependency
    | EdgeSupports     // Provides evidence for
    | EdgeContradicts  // Conflicts with
    | EdgeRefines      // Improves upon
    | EdgeTriggers     // Causes execution of
    | EdgeAggregates   // Merges from multiple sources
```

### 3.4 Structured Output Schema

Nodes can declare typed output schemas for downstream consumption:

```fsharp
type OutputFieldType =
    | StringType | NumberType | BoolType
    | ListType of OutputFieldType
    | ObjectType of OutputSchema

type NodeOutput =
    | SimpleOutput of string
    | StructuredOutput of string * OutputSchema
```

### 3.5 Policy System

```fsharp
type DslBudgets =
    { MaxToolCalls: int         // Default: 20
      MaxLlmTokens: int        // Default: 50000
      MaxRuntimeMinutes: int    // Default: 30
      MaxRetries: int }         // Default: 3

type DslSafetyPolicy =
    { PiiRedaction: bool
      ExternalClaimsRequireEvidence: bool
      LegalReviewRequiredFor: string list
      RequireApprovalAboveRisk: float option }
```

### 3.6 Multi-Agent Routing

Each WoT node can be assigned to a specialized agent:

```fsharp
type NodeAgent =
    | ByRole of string   // Route by role name (e.g., "Skeptic")
    | ById of string     // Route by specific agent ID
    | Default            // Use default routing
```

### 3.7 Success Criteria

```fsharp
type SuccessCriterion =
    | ConfidenceAbove of float
    | ContainsValue of string * string
    | MatchesRegex of string * string
    | ClaimHasEvidence of string
    | AllChecksPass
```

### 3.8 The DslNode (Full Definition)

A node in a WoT workflow carries:
- **Id, Kind, Name, Title** -- identity
- **Inputs/Outputs** -- data flow with optional structured schemas
- **Tool, Args, Checks** -- for Work nodes
- **Goal, Invariants, Constraints, Verdict** -- for Reason nodes
- **Agent** -- multi-agent routing
- **Transformation** -- GoT operation type
- **RequiresEvidence, EvidenceRefs** -- provenance tracking
- **Metadata** -- extensible key-value metadata

---

## 4. Reasoning Patterns

### 4.1 Pattern Kinds

```fsharp
type PatternKind =
    | Linear      // Sequential steps (Chain of Thought)
    | TreeSearch  // Branching with backtracking (Tree of Thoughts)
    | Loop        // Circular feedback (ReAct, Reflexion)
    | Graph       // DAG with aggregation (Graph of Thoughts)
    | Parallel    // Concurrent execution with consensus
```

### 4.2 Pattern Steps

```fsharp
type PatternStep =
    { Id: string                          // Unique step identifier
      Role: string                        // "Reason", "Tool", "Validate", "Critique"
      InstructionTemplate: string option  // Prompt template (supports {goal} substitution)
      Parameters: Map<string, string>     // Step-specific config
      Dependencies: string list }         // IDs of prerequisite steps
```

### 4.3 ReasoningPattern (The Evolvable Artifact)

```fsharp
type ReasoningPattern =
    { Id: string
      Name: string
      Kind: PatternKind
      Steps: PatternStep list
      Constraints: PatternConstraint list   // MaxSteps, Invariant, RequiredTools, OutputFormat
      Version: int                          // Evolution tracking
      Description: string                   // Used for pattern selection
      Metadata: Map<string, string> }       // Parent pattern ID, generation, etc.
```

### 4.4 Built-in Pattern Library

**linearCoT** (Linear): Decompose -> Reason -> Synthesize -> Verify. Standard chain-of-thought for routine tasks.

**criticRefinement** (Loop): Draft -> Critique -> Refine -> Final Check. Self-correcting pattern for quality-sensitive tasks.

**parallelBrainstorming** (Graph): Decompose -> [Idea 1 | Idea 2 | Idea 3] -> Synthesize. Generates multiple independent approaches and merges the best.

### 4.5 Pattern Compilation Pipeline

The `PatternCompiler` transforms high-level `ReasoningPattern` into executable `WoTPlan`:

1. **Step Mapping**: Each `PatternStep` becomes a `WoTNode` (Role "reason" -> Think node, "tool" -> Act node, "validate" -> Validate node, "critique" -> Think with "Critique:" prefix).
2. **Template Hydration**: `{goal}` in `InstructionTemplate` is replaced with the actual goal string.
3. **Dependency Resolution**: `Dependencies` list maps to `WoTEdge` connections.
4. **Entry Detection**: Nodes with no incoming edges become roots. Multiple roots get a synthetic start node.
5. **Plan Assembly**: Final `WoTPlan` includes nodes, edges, entry point, pattern metadata, and policy list.

Specialized compilers exist for each pattern family:
- `compileChainOfThought`: Linear sequence of Think nodes with Smart model hint
- `compileReAct`: Think -> Decide -> [Tool branches] -> Observe loop
- `compileGraphOfThoughts`: Decompose -> Branch(N) -> Evaluate(N) -> Synthesize
- `compileTreeOfThoughts`: Beam search with `beamWidth` candidates at each of `searchDepth` levels

---

## 5. Self-Improvement Loop

The self-improvement loop is TARS's key differentiator. It operates as: **Execute -> Trace -> Score -> Evolve -> Validate**.

### 5.1 Pattern Library (Tars.Evolution.PatternLibrary)

- **loadAll**: Loads pattern definitions from `.tars/patterns/*.json`.
- **findMatch**: Uses LLM to match a goal to the best available pattern from the library.
- **hydrate**: Substitutes `{{key}}` variables in pattern templates with context-specific values.
- **executePattern**: Runs a hydrated pattern template through the LLM and returns the result.
- **validateResult**: Checks execution output against success criteria (numeric containment, text match).

### 5.2 Fitness Scoring and Selection (Phase 15.3-15.5)

Traces are scored on:
- **Correctness**: Does the output match expected results?
- **Efficiency**: Token usage and step count
- **Robustness**: Success rate across problem variants

Selection uses Golden Traces -- verified correct reasoning paths that serve as reference standards. New pattern variants are diffed against goldens to measure improvement.

### 5.3 Curriculum Manager (Tars.Evolution.CurriculumManager)

Manages progressive difficulty for training TARS's reasoning:

```fsharp
type Difficulty = Beginner | Intermediate | Advanced | Expert | Unascertained

type CurriculumState =
    { CompletedProblems: Set<ProblemId>
      FailedProblems: Map<ProblemId, int>   // problem -> failure count
      CurrentDifficulty: Difficulty
      MasteryScore: float }                 // 0.0 - 1.0
```

**Promotion Logic**: After solving 5+ problems with mastery > 0.8, the agent advances to the next difficulty level. Success adds +0.1 mastery; failure subtracts -0.05. Problems are selected at current difficulty first, then easier (reinforce), then harder (challenge).

### 5.4 Mutation Engine (Phase 15.4)

Generates pattern variants by modifying:
- Instruction templates (rephrasing, adding constraints)
- Step ordering and dependencies
- Adding/removing validation steps
- Adjusting constraints (MaxSteps, required tools)

### 5.5 Golden Traces

A Golden Trace is a verified-correct execution trace for a specific goal. It includes:
- The WoTPlan that was executed
- Every WoTTraceStep with inputs, outputs, confidence, and token usage
- The final WoTResult with metrics
- A canonical hash for diffing against future executions

---

## 6. Cognitive Core

### 6.1 WoT Execution Types

```fsharp
type WoTPlan =
    { Id: Guid
      Nodes: WoTNode list
      Edges: WoTEdge list
      EntryNode: string
      Metadata: PatternMetadata
      Policy: string list }

type WoTResult =
    { Output: string
      Success: bool
      Trace: WoTTrace
      TriplesDelta: WoTTriple list       // Knowledge produced
      ToolsUsed: string list
      Metrics: ExecutionMetrics
      Warnings: string list
      Errors: string list
      CognitiveStateAfter: WoTCognitiveState option }
```

### 6.2 Cognitive State

```fsharp
type WoTCognitiveMode = Exploratory | Convergent | Critical

type WoTCognitiveState =
    { Mode: WoTCognitiveMode
      Eigenvalue: float           // System stability (0.0 - 1.0)
      Entropy: float              // Information diversity (0.0 - 1.0)
      BranchingFactor: float      // Reasoning graph complexity
      ActivePattern: PatternKind option
      WoTRunId: Guid option
      StepCount: int
      TokenBudget: int option
      LastTransition: DateTime
      ConstraintScore: float option
      SuccessRate: float }
```

Mode transitions: **Exploratory** (high entropy, branching) -> **Convergent** (narrowing to solution) -> **Critical** (validation and proof). Entropy > 0.8 triggers a switch to Graph of Thoughts for broader exploration.

### 6.3 WoT Node Payloads

```fsharp
type ReasonPayload = { Prompt: string; Hint: ModelHint option }
type ToolPayload = { Tool: string; Args: Map<string, obj> }
type ValidatePayload = { Invariants: WoTInvariant list }
type MemoryPayload = { Operation: MemoryOp }  // Query, Assert, Retract, Search
type ControlPayload =
    | Branch of condition * ifTrue * ifFalse
    | Loop of bodyEntry * until * maxIterations
    | Parallel of children
    | Decide of candidates * criteria
    | Observe of input * transform
```

### 6.4 Execution Metrics

```fsharp
type ExecutionMetrics =
    { TotalSteps: int
      SuccessfulSteps: int
      FailedSteps: int
      TotalTokens: int
      TotalDurationMs: int64
      BranchingFactor: float
      ConstraintScore: float option }
```

### 6.5 Knowledge Triples

Every WoT execution can produce knowledge triples:

```fsharp
type WoTTriple =
    { Subject: string; Predicate: string; Object: string
      Confidence: float; ProducedBy: string; ProducedAt: DateTime }
```

---

## 7. Formal Epistemology

### 7.1 Belief Revision (AGM-style)

TARS maintains a `ReflectionBeliefStore` with formal revision operations:

**Operations**: AddBelief, RevokeBelief, AdjustConfidence, ResolveContradiction, MergeBeliefs, SplitBelief, AddEvidence, RemoveEvidence.

**Contradiction Detection**: Checks for negation conflicts ("X" vs "not X") before adding beliefs.

**Cascade Updates**: Revoking a belief reduces confidence of dependent beliefs by 20%. Confidence dropping below 0.1 triggers automatic revocation.

**Conflict Resolution Strategies**: `HighestConfidenceWins` (keep strongest), `MergeCompatible` (combine non-contradictory aspects).

**Audit Trail**: Every revision emits typed `RevisionEvent` records for complete audit history.

### 7.2 Proof System

```fsharp
type ReflectionProof =
    | Tautology of statement: string                              // Strength: 1.0
    | ProofContradiction of statement: string                     // Strength: 0.0
    | ValidationSuccess of testName: string * details: string     // Strength: 0.95
    | ValidationFailure of testName: string * error: string       // Strength: 0.1
    | LogicalInference of premises * conclusion * rule            // Varies by rule
    | StatisticalEvidence of samples * successRate * confidence   // Sample-weighted
    | ExpertAssertion of source * credibility                     // Capped at 0.8

type ReflectionInferenceRule =
    | ModusPonens        // Strength: 0.9
    | ModusTollens       // 0.85
    | Syllogism          // 0.8
    | Contraposition     // 0.85
    | Generalization     // 0.6 (inductive, weaker)
    | Specialization     // 0.9
    | Abduction          // 0.5 (speculative)
    | Analogy            // 0.4 (weakest)
    | StatisticalInference // 0.7
```

**Proof Combination**: Uses simplified Dempster-Shafer theory: `combined = fold (fun acc s -> acc + s * (1.0 - acc)) 0.0 strengths`.

**Proof Strength Categories**: VeryWeak (0.0-0.2), Weak (0.2-0.4), Moderate (0.4-0.6), Strong (0.6-0.8), VeryStrong (0.8-1.0).

### 7.3 Epistemic Status Lifecycle

```fsharp
type EpistemicStatus =
    | Hypothesis          // Proposed, untested
    | VerifiedFact        // Passed generalization tests
    | UniversalPrinciple  // Abstracted and reused multiple times
    | Heuristic           // Useful but known brittle
    | Fallacy             // Proven false
```

---

## 8. Agent System

### 8.1 Agent Definition

```fsharp
type Agent =
    { Id: AgentId; Name: string; Version: string
      Model: string; SystemPrompt: string
      Tools: Tool list; Capabilities: Capability list
      State: AgentState; Memory: Message list
      Fitness: float; Drives: BaseDrives
      Constitution: AgentConstitution }

type BaseDrives =
    { Accuracy: float; Speed: float; Creativity: float; Safety: float }

type AgentState = Idle | Thinking | Acting | Observing | WaitingForUser | Error
```

### 8.2 Agent Roles (AgentRegistry)

Built-in roles with specialized system prompts and model hints:

| Role | Purpose | Model Hint | Temperature |
|------|---------|------------|-------------|
| **Planner** | Strategic planning and synthesis | reasoning | 0.3 |
| **QAEngineer** | Root cause analysis, pattern detection | reasoning | 0.2 |
| **Skeptic** | Adversarial critique, contradiction finding | reasoning | 0.5 |
| **Verifier** | Evidence auditing, claim validation | reasoning | 0.1 |
| **Comms** | Communication drafting | fast | 0.4 |
| **Regulatory** | Compliance and risk assessment | reasoning | 0.1 |
| **SupplyChain** | Logistics and operations analysis | reasoning | 0.2 |
| **Default** | Generic reasoning | reasoning | -- |

### 8.3 Message Protocol

```fsharp
type Performative =
    | Request | Inform | Query | Propose
    | Refuse | Failure | NotUnderstood | Event

type SemanticMessage<'T> =
    { Id: Guid; CorrelationId: CorrelationId
      Sender: MessageEndpoint; Receiver: MessageEndpoint option
      Performative: Performative; Intent: AgentDomain option
      Constraints: SemanticConstraints; Content: 'T
      Ontology: string option; Language: string
      Timestamp: DateTime; Metadata: Map<string, string> }
```

### 8.4 Routing Strategies

```fsharp
type RoutingStrategy =
    | Pinned of AgentId              // Always route to specific agent
    | Canary of primary * canary * weight  // A/B testing (weight = canary %)
    | RoundRobin of AgentId list     // Even distribution
```

---

## 9. Infrastructure

### 9.1 LLM Service (Multi-Backend)

TARS supports multiple LLM backends through `ILlmService`:

- **Ollama** (local): Primary backend for development and privacy-sensitive workloads
- **OpenAI-compatible** (cloud): Azure OpenAI, OpenRouter, any OpenAI-API server
- **vLLM** (local/remote): High-throughput serving
- **LlamaCpp** (local): Direct GGUF model inference
- **LlamaSharp** (local): .NET-native inference

Model routing uses hints: `Fast` (speed-optimized), `Smart` (quality-optimized), `Reasoning` (extended thinking), `Specific(model)`.

### 9.2 Microsoft.Extensions.AI Integration (ChatClientAdapter)

Bidirectional adapters bridge TARS's `ILlmService` with Microsoft's `IChatClient`:

- **LlmServiceChatClient**: Wraps `ILlmService` as `IChatClient` -- allows TARS backends to be consumed by any MAF-based code. Supports both completion and streaming.
- **ChatClientLlmService**: Wraps `IChatClient` as `ILlmService` -- allows any Microsoft.Extensions.AI provider (Azure AI, Semantic Kernel, etc.) to be used through the TARS interface.

Both adapters handle role mapping, option conversion, usage tracking, and finish reason translation.

### 9.3 MCP Server (OfficialMcpBridge)

TARS exposes its tool registry via the Model Context Protocol using the official `ModelContextProtocol` NuGet SDK:

- **toOfficialMcpTool**: Converts a TARS `Tool` record to an `McpServerTool` with proper delegate wrapping.
- **configureHost**: Sets up `HostApplicationBuilder` with MCP server over stdio transport.
- **runStdioServer**: One-call entry point that creates and runs the MCP server.

This allows any MCP-compatible client (Claude Desktop, Cursor, etc.) to use all 152+ TARS tools.

### 9.4 Existing Reasoning Patterns (Tars.Cortex.Patterns)

Pre-WoT implementations that are being migrated to compile through the WoT spine:

- **chainOfThought**: Sequential LLM reasoning with step-by-step analysis
- **reAct**: Think -> Act -> Observe loop with tool integration
- **planAndExecute**: Goal decomposition into subtasks with sequential execution
- **graphOfThoughts**: Branching exploration with scoring and synthesis
- **treeOfThoughts**: Beam search with evaluation at each depth level
- **workflowOfThought**: Traced execution with variable resolution

---

## 10. Current State & Roadmap

### 10.1 Completion Status

| Component | Status |
|-----------|--------|
| Core domain types, abstractions | Complete |
| LLM multi-backend service | Complete |
| WoT DSL grammar (typed edges, policies, agents, structured output) | Complete |
| Multi-agent routing with 8 built-in roles | Complete |
| Pattern Compiler (ReasoningPattern -> WoTPlan) | Complete |
| Belief Revision engine (AGM-style) | Complete |
| Proof System (7 proof types, 9 inference rules) | Complete |
| Fitness Scoring | Complete |
| Mutation Engine v0 | Complete |
| Selection Logic (Golden/Diff) | Complete |
| Pattern Library (load, match, hydrate, execute, validate) | Complete |
| Curriculum Manager (difficulty progression) | Complete |
| Evidence Provenance tracking | Complete |
| Feedback Loops in WoT | Complete |
| Microsoft.Extensions.AI bidirectional bridge | Complete |
| MCP Server via official SDK | Complete |
| Neuro-Symbolic Benchmark Suite | Complete |

### 10.2 In Progress

- **Phase 15.6: Pattern Compiler maturation** -- expanding the set of patterns that compile cleanly to WoT, improving the compilation fidelity.
- **Phase 9: Symbolic Knowledge & Internet Ingestion** -- connecting execution traces to persistent knowledge graph storage.

### 10.3 Planned Next Steps

1. **Promotion Pipeline**: Automated promotion of successful pattern variants from candidate to library.
2. **Grammar Governor**: Runtime enforcement of WoT grammar constraints during execution (budget limits, safety policies, consensus protocols).
3. **Layer 4 Metadata**: Rich metadata on WoT nodes for observability -- token costs, latency percentiles, confidence distributions.
4. **Knowledge Graph Cold Start**: Phase 1 append-only triples, Phase 2 provenance, Phase 3 inference.
5. **Cognitive State Dashboard**: Real-time visualization of entropy, branching factor, mode transitions during WoT execution.

### 10.4 Paused (Consolidation)

- Phase 10: 3D Knowledge Graph Visualization
- Phase 11: Cognitive Grounding (too abstract for current needs)

### 10.5 The Canonical Loop

The target end-state for TARS operations:

```
Objective -> Pattern Selection -> WoT Compilation -> Execution -> Trace
    ^                                                                |
    |   Fitness Scoring <- Golden Diff <- Trace Analysis  <----------+
    |         |
    +-- Mutation/Selection -> Updated Pattern Library
```

Every execution produces a canonical trace. Traces are diffed against golden references. Fitness scores drive mutation and selection. Successful variants are promoted to the pattern library. The system improves itself through measured, auditable iteration.
