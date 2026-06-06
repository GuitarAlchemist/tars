# TARS v2 Research Synthesis

## Key Insights from `docs/__research`

### 1. Multi-Agent System Protocols (Most Critical)

**Emerging Standard Stack:**

- **Transport**: gRPC (HTTP/2) or NATS for events
- **Encoding**: Protobuf/MessagePack (JSON only for LLMs)
- **Semantic Layer**: Agent Speech Acts (ASK, TELL, PROPOSE, ACCEPT, REJECT, ACT, EVENT, ERROR)

**Concrete Recommendations for TARS:**

1. **In-Process First**: F# `MailboxProcessor` or `Channel<T>` (already have `EventBus` ✅)
2. **Network Skin**: gRPC + Protobuf when agents are in different processes
3. **Events**: NATS or Redis Streams for `tars.belief.updated`, `tars.plan.candidate`
4. **Semantic Protocol**: Minimal speech acts vocabulary

**Key Quote:**
> "The best protocol isn't a single wire format; it's a small, explicit semantic vocabulary (your agent speech acts) running on top of boring, battle-tested plumbing."

### 2. Backpressure & AI Design Patterns (Game-Changing)

**Core Insight**:
Backpressure in AI = "cognitive load" and "semantic fan-out," not bytes/sec.

**12 Critical Patterns to Implement:**

#### Tier 1 (Implement in v2.1)

1. **Token Budget Governor** - Cap resources per workflow (tokens, LLM calls, time, $)
2. **Semantic Fan-out Limiter** - Prevent planners from exploding (limit subtasks to top-K)
3. **Adaptive Reflection Loop** - Stop reflecting when improvement < epsilon
4. **Context Compaction Pipeline** - Compress memory to avoid context overflow

#### Tier 2 (Implement in v2.2)

5. **Uncertainty-Gated Planner** - Adjust thinking depth based on uncertainty
6. **Consensus Circuit Breaker** - Stop when agents violently disagree
7. **Tool Circuit Breaker** - Stop hammering flaky tools
8. **Semantic Watchdog** - Kill pathological loops

#### Tier 3 (Future)

9. **Memory Ingress Throttler** - Rate-limit RAG updates
10. **Triage Summarizer** - Summarize long-tail instead of dropping
11. **Cognitive Bulkheads** - Prevent experiments from starving critical flows
12. **Backpressure-Oriented Orchestrator** - Meta-pattern combining all

**Key Mapping:**

```
Bounded Queue → Bounded task queue with semantic dropping
Rate Limiter → Token/time/money/thinking-depth budgets
State Machine → Reasoning phases with guard conditions
Actor Model → Agent teams with mailbox limits & supervision
Circuit Breaker → Isolation of risky tools and expensive reasoning paths
```

### 3. Emerging AI Standards (Beyond MCP)

**The 5-Piece Convergence:**

1. **Transport**: WebSocket (bidirectional, low-latency)
2. **RPC Format**: JSON-RPC or MsgPack-RPC
3. **Capability Definition**: OpenAI Tool JSON Schema
4. **Model Calls**: OpenAI-compatible REST
5. **Containers**: WASI Component Model (future)

**Other Emerging Standards:**

- **LSP-style Protocols**: JSON-RPC + declarative capabilities (agents as "language servers")
- **OpenAI Actions Schema**: Universal tool definition format
- **JAXON/JIO**: Agentic OS research (early but influential)
- **GraphRAG Lineage**: Graph-based message passing
- **FLYTE/Temporal**: Durable workflow orchestration

## Impact on TARS v2 Architecture

### Immediate Changes (v2.1)

#### 1. **Add Backpressure Patterns to Evolution Engine**

```fsharp
type BudgetGovernor =
    { MaxTokens: int
      MaxLLMCalls: int
      MaxWallTimeMs: int
      mutable TokensUsed: int
      mutable CallsUsed: int }
    
    member this.TryConsume(tokens: int, calls: int) =
        if this.TokensUsed + tokens <= this.MaxTokens &&
           this.CallsUsed + calls <= this.MaxLLMCalls then
            this.TokensUsed <- this.TokensUsed + tokens
            this.CallsUsed <- this.CallsUsed + calls
            true
        else
            false
```

#### 2. **Formalize Agent Speech Acts**

```fsharp
type AgentIntent =
    | Ask      // Request computation/answer
    | Tell     // Share fact/belief
    | Propose  // Suggest plan/action
    | Accept   // Agree to proposal
    | Reject   // Decline proposal
    | Act      // Execute concrete action
    | Event    // Broadcast observation
    | Error    // Report failure

type AgentMessage =
    { Id: Guid
      CorrelationId: Guid
      From: AgentId
      To: AgentId option  // None = broadcast
      Intent: AgentIntent
      ContentType: string  // "proto:SetGoal", "json:LLMCall"
      Payload: obj
      Timestamp: DateTime }
```

#### 3. **Add Semantic Fan-out Limiting**

```fsharp
// In Curriculum Agent
let selectTopK (tasks: TaskDefinition list) (k: int) =
    tasks
    |> List.sortByDescending (fun t -> t.Score)  // Add scoring
    |> List.truncate k
```

#### 4. **Implement Adaptive Reflection**

```fsharp
let rec reflectUntilConvergence (state: AgentState) (maxReflections: int) =
    task {
        if maxReflections = 0 then return state
        else
            let! newState = reflectOnce state
            let improvement = measureImprovement state newState
            if improvement < 0.05 then  // Diminishing returns
                return newState
            else
                return! reflectUntilConvergence newState (maxReflections - 1)
    }
```

### Medium-Term (v2.2)

#### 1. **gRPC Agent Communication**

- Implement `AgentBus` interface with gRPC backend
- Support both in-process and cross-process agents

#### 2. **NATS Event Bus**

- Topics: `tars.evolution.task_generated`, `tars.evolution.task_completed`
- Decouple agents via pub-sub

#### 3. **Tool Registry with Circuit Breakers**

```fsharp
type ToolHealthMonitor() =
    let mutable failureRate = 0.0
    let circuitOpen = ref false
    
    member this.RecordCall(success: bool) =
        // Track rolling failure rate
        if failureRate > 0.5 then circuitOpen := true
    
    member this.IsAvailable() = not !circuitOpen
```

### Long-Term (v2.3+)

#### 1. **Full Backpressure-Oriented Orchestrator**

- Budget Governor at top level
- Cognitive Bulkheads per agent type
- Semantic Watchdog for runaway loops

#### 2. **MCP + OpenAI-Compatible Stack**

- MCP for agent-to-agent communication
- OpenAI REST for model calls
- Clear separation of concerns

## Updated Recommendations

### **What to Do Now:**

1. ✅ Continue with Phase 5 (Metascript) - already done!
2. **Phase 5.1**: Add `BudgetGovernor` to `EvolutionContext`
3. **Phase 5.2**: Implement `AgentIntent` DU and update `AgentMessage`
4. **Phase 5.3**: Add semantic fan-out limiting to Curriculum Agent

### **What to Do in v2.1:**

1. Implement top 4 backpressure patterns (Budget, Fan-out, Reflection, Compaction)
2. Formalize agent communication with speech acts
3. Add circuit breakers to tool execution

### **What to Defer to v2.2:**

1. gRPC/NATS infrastructure
2. GAM integration (Python microservice)
3. Full orchestrator with all 12 patterns

## Key Design Principles Extracted

1. **"Semantic Load" is Real**: Treat LLM calls, tokens, and "thinking depth" as finite resources
2. **"Boring Plumbing, Smart Semantics"**: Use proven transport (gRPC, NATS) but add intelligent routing
3. **"Agents as Actors"**: Each agent = MailboxProcessor with supervision and backpressure
4. **"Budget Everything"**: Tokens, time, money, reflection depth—all need governors
5. **"Fail Gracefully"**: Circuit breakers, timeouts, and "best effort + limitations" outputs
6. **"Compress Don't Drop"**: When context overflows, summarize instead of losing signal

## Critical Quotes to Remember

> "The best protocol isn't a single wire format; it's a small, explicit semantic vocabulary running on top of boring, battle-tested plumbing."

> "You don't need full FIPA ACL, but you do want a tiny, opinionated set of 'speech acts'."

> "Once you start treating LLMs as components inside a resource-aware semantic OS, patterns like these become your 'cognitive syscalls'."

> "All of these are 100% implementable today with boring constructs you already know: queues, budgets, state machines, actor/mailbox limits, circuit breakers, etc. You just upgrade the knobs from 'requests per second' to 'how much thinking, and where, is actually worth it?'"

### 4. Architectural Constitution & Refined Roadmap (From Feedback)

**Source:** `docs/__research/ChatGPT-Document feedback summary (1).md`

**Core Insight:**
To prevent entropy in a self-evolving system, TARS needs a "Constitution" — a set of inviolable laws that precede modules.

#### The Constitution (Invariants)

1. **Immutability by Default**: Configs, grammars, agent definitions, and skills are versioned and immutable once released. Mutation only happens via explicit evolution workflows.
11. **Cognitive Bulkheads** - Prevent experiments from starving critical flows
12. **Backpressure-Oriented Orchestrator** - Meta-pattern combining all

**Key Mapping:**

```
Bounded Queue → Bounded task queue with semantic dropping
Rate Limiter → Token/time/money/thinking-depth budgets
State Machine → Reasoning phases with guard conditions
Actor Model → Agent teams with mailbox limits & supervision
Circuit Breaker → Isolation of risky tools and expensive reasoning paths
```

### 3. Emerging AI Standards (Beyond MCP)

**The 5-Piece Convergence:**

1. **Transport**: WebSocket (bidirectional, low-latency)
2. **RPC Format**: JSON-RPC or MsgPack-RPC
3. **Capability Definition**: OpenAI Tool JSON Schema
4. **Model Calls**: OpenAI-compatible REST
5. **Containers**: WASI Component Model (future)

**Other Emerging Standards:**

- **LSP-style Protocols**: JSON-RPC + declarative capabilities (agents as "language servers")
- **OpenAI Actions Schema**: Universal tool definition format
- **JAXON/JIO**: Agentic OS research (early but influential)
- **GraphRAG Lineage**: Graph-based message passing
- **FLYTE/Temporal**: Durable workflow orchestration

## Impact on TARS v2 Architecture

### Immediate Changes (v2.1)

#### 1. **Add Backpressure Patterns to Evolution Engine**

```fsharp
type BudgetGovernor =
    { MaxTokens: int
      MaxLLMCalls: int
      MaxWallTimeMs: int
      mutable TokensUsed: int
      mutable CallsUsed: int }
    
    member this.TryConsume(tokens: int, calls: int) =
        if this.TokensUsed + tokens <= this.MaxTokens &&
           this.CallsUsed + calls <= this.MaxLLMCalls then
            this.TokensUsed <- this.TokensUsed + tokens
            this.CallsUsed <- this.CallsUsed + calls
            true
        else
            false
```

#### 2. **Formalize Agent Speech Acts**

```fsharp
type AgentIntent =
    | Ask      // Request computation/answer
    | Tell     // Share fact/belief
    | Propose  // Suggest plan/action
    | Accept   // Agree to proposal
    | Reject   // Decline proposal
    | Act      // Execute concrete action
    | Event    // Broadcast observation
    | Error    // Report failure

type AgentMessage =
    { Id: Guid
      CorrelationId: Guid
      From: AgentId
      To: AgentId option  // None = broadcast
      Intent: AgentIntent
      ContentType: string  // "proto:SetGoal", "json:LLMCall"
      Payload: obj
      Timestamp: DateTime }
```

#### 3. **Add Semantic Fan-out Limiting**

```fsharp
// In Curriculum Agent
let selectTopK (tasks: TaskDefinition list) (k: int) =
    tasks
    |> List.sortByDescending (fun t -> t.Score)  // Add scoring
    |> List.truncate k
```

#### 4. **Implement Adaptive Reflection**

```fsharp
let rec reflectUntilConvergence (state: AgentState) (maxReflections: int) =
    task {
        if maxReflections = 0 then return state
        else
            let! newState = reflectOnce state
            let improvement = measureImprovement state newState
            if improvement < 0.05 then  // Diminishing returns
                return newState
            else
                return! reflectUntilConvergence newState (maxReflections - 1)
    }
```

### Medium-Term (v2.2)

#### 1. **gRPC Agent Communication**

- Implement `AgentBus` interface with gRPC backend
- Support both in-process and cross-process agents

#### 2. **NATS Event Bus**

- Topics: `tars.evolution.task_generated`, `tars.evolution.task_completed`
- Decouple agents via pub-sub

#### 3. **Tool Registry with Circuit Breakers**

```fsharp
type ToolHealthMonitor() =
    let mutable failureRate = 0.0
    let circuitOpen = ref false
    
    member this.RecordCall(success: bool) =
        // Track rolling failure rate
        if failureRate > 0.5 then circuitOpen := true
    
    member this.IsAvailable() = not !circuitOpen
```

### Long-Term (v2.3+)

#### 1. **Full Backpressure-Oriented Orchestrator**

- Budget Governor at top level
- Cognitive Bulkheads per agent type
- Semantic Watchdog for runaway loops

#### 2. **MCP + OpenAI-Compatible Stack**

- MCP for agent-to-agent communication
- OpenAI REST for model calls
- Clear separation of concerns

## Updated Recommendations

### **What to Do Now:**

1. ✅ Continue with Phase 5 (Metascript) - already done!
2. **Phase 5.1**: Add `BudgetGovernor` to `EvolutionContext`
3. **Phase 5.2**: Implement `AgentIntent` DU and update `AgentMessage`
4. **Phase 5.3**: Add semantic fan-out limiting to Curriculum Agent

### **What to Do in v2.1:**

1. Implement top 4 backpressure patterns (Budget, Fan-out, Reflection, Compaction)
2. Formalize agent communication with speech acts
3. Add circuit breakers to tool execution

### **What to Defer to v2.2:**

1. gRPC/NATS infrastructure
2. GAM integration (Python microservice)
3. Full orchestrator with all 12 patterns

## Key Design Principles Extracted

1. **"Semantic Load" is Real**: Treat LLM calls, tokens, and "thinking depth" as finite resources
2. **"Boring Plumbing, Smart Semantics"**: Use proven transport (gRPC, NATS) but add intelligent routing
3. **"Agents as Actors"**: Each agent = MailboxProcessor with supervision and backpressure
4. **"Budget Everything"**: Tokens, time, money, reflection depth—all need governors
5. **"Fail Gracefully"**: Circuit breakers, timeouts, and "best effort + limitations" outputs
6. **"Compress Don't Drop"**: When context overflows, summarize instead of losing signal

## Critical Quotes to Remember

> "The best protocol isn't a single wire format; it's a small, explicit semantic vocabulary running on top of boring, battle-tested plumbing."

> "You don't need full FIPA ACL, but you do want a tiny, opinionated set of 'speech acts'."

> "Once you start treating LLMs as components inside a resource-aware semantic OS, patterns like these become your 'cognitive syscalls'."

> "All of these are 100% implementable today with boring constructs you already know: queues, budgets, state machines, actor/mailbox limits, circuit breakers, etc. You just upgrade the knobs from 'requests per second' to 'how much thinking, and where, is actually worth it?'"

### 4. Architectural Constitution & Refined Roadmap (From Feedback)

**Source:** `docs/__research/ChatGPT-Document feedback summary (1).md`

**Core Insight:**
To prevent entropy in a self-evolving system, TARS needs an "Constitution" — a set of inviolable laws that precede modules.

#### The Constitution (Invariants)

1. **Immutability by Default**: Configs, grammars, agent definitions, and skills are versioned and immutable once released. Mutation only happens via explicit evolution workflows.
2. **Universal Versioning**: Everything (Agents, Skills, Grammars, Beliefs) must have `version`, `parentVersion`, and `createdAt`. This enables rollback and evolution tracking.
3. **Time as First-Class Citizen**: All state must have `validFrom`, `lastUsed`, and `decayScore`. The system must actively prune and compact old state.
4. **Safety Gates**: No code mutation without passing: Static Checks → Test Harness → Sandbox Execution → Human Override (optional).

#### Refined Module Responsibilities

* **Tars.Kernel**: Micro-kernel + Constraint Engine. Handles lifecycle, routing, budget enforcement, and safety gates. *Not* intelligence.
- **Tars.Cortex**: Cognitive layer. Abstracts LLMs, runs FLUX/DSLs, handles planning and grammar distillation.
- **Tars.Memory**: The Grid. Unifies Context (short-term), VectorStore (long-term), and **Internal Knowledge Graph** (Graphiti-style, backed by SQLite). *Drop external Triple Stores for now.*
- **Tars.Agents**: Data-first definitions of agents (capabilities, roles, protocols) + Runtime adapter. Use **F# Discriminated Unions** for strict state safety.
- **Tars.Skills**: Versioned registry, execution host (**WASM/Docker**), and signing/trust anchors.
- **Tars.Observability**: The "Black Box". Traces reasoning chains, memory diffs, and metrics.

#### Execution Roadmap Adjustments

* **Phase 0**: Repo structure alignment & **F# Type Definitions** (DUs, Units of Measure).
- **Phase 1**: Minimal Kernel + Single Agent + In-Memory Memory (MVP Spine).
- **Phase 2**: Cortex + Real LLM + **Internal KG** (Simple SQLite/Graphiti).
- **Phase 3**: Skills & Tooling Layer (**Containerized** execution).
- **Phase 4**: Observability + Evolution Scaffolding.
- **Phase 5**: Safe Self-Modification (The Loop).
- **Phase 6**: Grammar & Multi-Agent Evolution.
