# Circuit-Based AI Architecture for TARS v2

**Date:** 2025-11-27  
**Status:** Research Integration  
**Research Source:** `docs/__research/ChatGPT-Circuit-based AI architecture.md`

## Overview

This document describes how **electrical circuit analogies** can model information flow in TARS v2's multi-agent system. By mapping circuit elements to software constructs, we gain a powerful framework for designing, reasoning about, and implementing cognitive token flows.

## Core Analogy: Tokens as Electrical Current

In TARS:

- **Tokens/Messages** = Electric current or fluid flow
- **Agents** = Circuit components (resistors, capacitors, transistors)
- **Message Channels** = Wires/pipes connecting components
- **Backpressure** = Voltage drop / resistance limiting current

### Continuous vs. Discrete Flows

- **Continuous Flow:** Streaming token-by-token processing (real-time reasoning)
- **Discrete Flow:** Batch processing of complete messages/responses

Both modes are supported in TARS v2 through the EventBus and agent channels.

---

## The Three Circuit Elements

### 1. Resistors: Throttling and Backpressure Control

**Electrical Analog:** Limits current flow, drops voltage  
**Software Analog:** Rate limiting, bounded channels, backpressure

#### Implementation in TARS v2

**Current Implementation:**

```fsharp
// In EventBus.fs - bounded channel acts as resistor
let channel = Channel.CreateUnbounded<SemanticMessage<obj>>()
// Future: use BoundedCapacity
let boundedChannel = Channel.CreateBounded<T>(
    BoundedChannelOptions(
        Capacity = 100,  // "Resistance value"
        FullMode = BoundedChannelFullMode.Wait  // Backpressure behavior
    ))
```

**Agent Message Processing (existing):**

```fsharp
// AgentCommunication.fs pattern (from TARS v1)
// Process max 10 messages, then sleep 100ms
// This is a software resistor limiting flow to ~100 msg/sec
```

**Recommended F# Patterns:**

- `MailboxProcessor` with limited queue
- TPL Dataflow `BufferBlock<T>` with `BoundedCapacity`
- `System.Threading.Channels` with bounded capacity

**Use Cases:**

- Prevent fast LLM agents from overwhelming slower reasoning agents
- Fair resource allocation across agents
- Stability under load (no cascade failures)

### 2. Capacitors: Buffers and Memory

**Electrical Analog:** Stores charge, smooths fluctuations  
**Software Analog:** Message queues, working memory, batch accumulators

#### Implementation in TARS v2

**Message Queues (existing):**

```fsharp
// Each agent has a built-in buffer (capacitor)
type Agent = {
    ...
    Memory: Message list  // Short-term buffer
}
```

**Batch Processing (TPL Dataflow):**

```fsharp
// BatchBlock acts as capacitor - accumulates, then discharges
let batchBuffer = BatchBlock<Token>(batchSize = 10)
// Collects 10 tokens, then releases as batch
```

**Async Sequences (streaming):**

```fsharp
open FSharp.Control

// Accumulate tokens over time window (capacitive delay)
let bufferTokens (stream: AsyncSeq<Token>) =
    stream
    |> AsyncSeq.bufferByTime (TimeSpan.FromSeconds(1.0))
    // Charges for 1 second, then discharges batch
```

**Use Cases:**

- Smooth bursty LLM output into steady downstream processing
- Accumulate partial results before fusion
- Working memory for agents (context window management)

### 3. Transistors: Gating and Conditional Flow

**Electrical Analog:** Controlled switch, amplifies or blocks current  
**Software Analog:** Task dependencies, conditional routing, synchronization barriers

#### Implementation in TARS v2

**Task Dependencies** (existing in TARS DSL):

```fsharp
// From docs/DSL/block-types.md
TASK AnalyzeCode { ... }

TASK RefactorCode {
    dependencies: [AnalyzeCode];  // Gate: blocks until AnalyzeCode completes
    ACTION {
        let analysis = getTaskResult(AnalyzeCode);  // Control signal
        ... refactor based on analysis ...
    }
}
```

**TPL Dataflow Joins (AND gates):**

```fsharp
// Wait for BOTH inputs before proceeding
let joinGate = JoinBlock<AgentResult, AgentResult>()

// Links from two parallel agents
agent1Output.LinkTo(joinGate.Target1)
agent2Output.LinkTo(joinGate.Target2)

// Downstream receives tuple only when BOTH ready
let fusionAgent = ActionBlock(fun (r1, r2) -> 
    // Fusion logic here
)
joinGate.LinkTo(fusionAgent)
```

**Mailbox Processor Gating:**

```fsharp
// Agent waits for "unlock" message before processing others
let gatedAgent = MailboxProcessor.Start(fun inbox ->
    let rec locked () = async {
        let! msg = inbox.Receive()
        match msg with
        | Unlock -> return! unlocked ()
        | _      -> return! locked ()  // Ignore other messages
    }
    and unlocked () = async {
        let! msg = inbox.Receive()
        match msg with
        | Lock   -> return! locked ()
        | Data d -> 
            // Process data
            return! unlocked ()
    }
    locked ()
)
```

**Use Cases:**

- Synchronize parallel agent results before fusion
- Enforce workflow ordering (analysis → generation → review)
- Conditional branching based on agent state

---

## Multi-Agent Coordination Patterns

### Pattern 1: Regulated Pipeline

```
[Fast Agent] --[Resistor]--> [Buffer] --[Gate]--> [Slow Agent]
     A            (rate      (queue)   (wait for    B
                   limit)               B ready)
```

**F# Implementation:**

```fsharp
let coordinatedPipeline (agentA: IAgent) (agentB: IAgent) =
    
    // Resistor: bounded channel limits A→B flow
    let pipe = Channel.CreateBounded<Message>(capacity = 50)
    
    // Capacitor: batch accumulator
    let batcher = BatchBlock<Message>(batchSize = 10)
    pipe.Reader |> linkTo batcher
    
    // Transistor: gate that opens when B is ready
    let gate = ActionBlock<Message[]>(fun batch ->
        if agentB.IsReady then
            agentB.ProcessBatch(batch)
        else
            // Re-enqueue or wait
            Task.Delay(100) |> Async.AwaitTask
            |> Async.RunSynchronously
            agentB.ProcessBatch(batch)
    )
    batcher.LinkTo(gate)
    
    // AgentA writes to pipe.Writer
    // Flow is automatically regulated
```

### Pattern 2: Fan-Out with Backpressure

```
                  /--> [Agent 1]
[Coordinator] ---|----> [Agent 2]
                  \--> [Agent 3]
```

**F# Implementation:**

```fsharp
let fanOutWithBackpressure (coordinator: IAgent) (workers: IAgent list) =
    
    // Each worker has bounded input (resistor)
    let queues = 
        workers 
        |> List.map (fun _ -> 
            Channel.CreateBounded<Task>(capacity = 20))
    
    // Coordinator tries to enqueue
    // If full, `WriteAsync` awaits (backpressure!)
    let distribute (task: Task) = async {
        let! _ = 
            queues
            |> List.map (fun q -> q.Writer.WriteAsync(task).AsTask())
            |> Task.WhenAny  // Send to first available
        return ()
    }
```

### Pattern 3: Synchronization Barrier (AND Gate)

```
[Vision Agent] \
                 --> [Fusion Agent] --> [Output]
[Language Agent]/
```

**F# Implementation:**

```fsharp
let fusionBarrier (visionAgent: IAgent) (langAgent: IAgent) =
    
    // JoinBlock = AND gate
    let barrier = JoinBlock<VisionResult, LanguageResult>()
    
    // Wire agents to barrier inputs
    visionAgent.Output.LinkTo(barrier.Target1)
    langAgent.Output.LinkTo(barrier.Target2)
    
    // Fusion agent receives ONLY when BOTH present
    let fusion = ActionBlock(fun (vis, lang) ->
        // Combine modalities
        combineResults vis lang
    )
    barrier.LinkTo(fusion)
```

---

## Pre-LLM Transformer Pipeline (Advanced Pattern)

### Concept: Circuit Board Before Main LLM

Instead of sending raw user input directly to the main LLM, pass it through a "circuit" of smaller transformers:

```
[User Input] 
    |
    v
[Safety Filter] ----blocked---> [Error Response]
    |pass
    v
[Intent Classifier] --> (routing decision)
    |
    v
[Context Summarizer] (capacitor: compress context)
    |
    v
[Prompt Rewriter] (transformer: normalize query)
    |
    v
[Main LLM] --> [Final Response]
```

### F# Implementation Sketch

**Stage Interface:**

```fsharp
type IPreLlmStage =
    abstract member Name: string
    abstract member Execute: 
        ctx: PreLlmContext * ct: CancellationToken -> 
        Async<PreLlmContext>

// Pipeline = list of stages
type PreLlmPipeline = IPreLlmStage list
```

**Pipeline Execution (fold pattern):**

```fsharp
let runPipeline (pipeline: PreLlmPipeline) (ct: CancellationToken) (input: RawUserInput) = async {
    let! result =
        pipeline
        |> List.fold (fun ctxAsync stage -> async {
            let! ctx = ctxAsync
            // Could short-circuit on Safety = Blocked
            match ctx.Safety with
            | Some (Blocked _) -> return ctx
            | _ -> return! stage.Execute(ctx, ct)
        }) (async { return PreLlmContext.fromRaw input })
    return result
}
```

**Concrete Stages:**

1. **Safety Filter (Transistor):** Blocks unsafe content
2. **Intent Classifier (Router):** Determines routing strategy
3. **Context Summarizer (Capacitor):** Compresses long context
4. **Prompt Rewriter (Transformer):** Normalizes/clarifies query

**Integration with TARS:**

```fsharp
type PreLlmAgent(pipeline: PreLlmPipeline, llm: ILlmClient) =
    interface IAgent with
        member _.Id = "pre-llm-orchestrator"
        
        member _.Handle(msg: AgentMessage, ct: CancellationToken) = task {
            match msg with
            | UserQuery (userId, text) ->
                let input = { Text = text; UserId = userId; ChannelId = None }
                let! ctx = runPipeline pipeline ct input |> Async.StartAsTask
                
                // Build final LLM request from enriched context
                let request = buildLlmRequest ctx
                let! response = llm.Call(request, ct)
                return TextResponse response
            | _ ->
                return ErrorResponse "Unsupported message"
        }
```

---

## Configuration-Driven Circuits (FLUX/Metascript)

### Metascript Example

```flux
pre_llm {
  enabled = true
  
  stage safety_filter {
    model = "tars-safety-mini"
    params = { threshold = "0.85" }
  }
  
  stage intent_classifier {
    model = "tars-intent-mini"
  }
  
  stage context_summarizer {
    model = "tars-summarizer-small"
  }
  
  stage prompt_rewriter {
    model = "tars-rewriter-small"
  }
}
```

### Benefits

1. **Compositional:** Swap stages without recompiling
2. **Observable:** Each stage can emit metrics
3. **Evolvable:** TARS can modify its own pre-LLM circuit
4. **Testable:** Each stage is isolated and unit-testable

---

## Recommended Implementation Phases

### Phase 1: Backpressure Foundation

- [ ] Replace unbounded channels with bounded in `EventBus`
- [ ] Add capacity monitoring/metrics
- [ ] Implement adaptive throttling based on load

### Phase 2: Buffering Enhancements

- [ ] Add `BatchBlock` for token accumulation patterns
- [ ] Implement working memory as capacitors
- [ ] Add time-based buffering (AsyncSeq patterns)

### Phase 3: Gating and Synchronization

- [ ] Enhance task dependencies with conditional logic
- [ ] Implement `JoinBlock` for multi-agent fusion
- [ ] Add circuit-breaker patterns (already started in Phase 6.0)

### Phase 4: Pre-LLM Pipeline

- [ ] Implement `IPreLlmStage` interface
- [ ] Create safety, intent, summarization, rewriting stages
- [ ] Integrate into `LlmService`
- [ ] Add FLUX configuration support

---

## Integration with Phase 6.5 (Agentic Interfaces)

The circuit-based architecture **complements** the agentic interfaces work:

- **Resistors (backpressure)** → Helps manage `PartialSuccess` flows
- **Capacitors (buffers)** → Accumulates warnings/errors for aggregation
- **Transistors (gates)** → Implements capability-based routing
- **Pre-LLM pipeline** → Enhances prompt quality before `agent { }` CE

**Combined Pattern:**

```fsharp
let cognitiveCircuit (task: TaskSpec) : AgentWorkflow<string> =
  agent {
      // Pre-LLM stages enrich the prompt
      let! enrichedPrompt = preLlmPipeline task
      
      // Main agent workflow with circuit-based coordination
      let! planJson = 
          callAgentByCapability Planning enrichedPrompt
          |> withBackpressure maxRate=10  // Resistor
          |> withBuffer size=5             // Capacitor
      
      let! execution = 
          callAgentByCapability TaskExecution planJson
          |> gateOn (planJson.Confidence > 0.8)  // Transistor
      
      return execution
  }
```

---

## References

1. **Research**: `docs/__research/ChatGPT-Circuit-based AI architecture.md`
2. **TARS v1 Patterns**: AgentCommunication.fs message queues
3. **TPL Dataflow**: `System.Threading.Tasks.Dataflow`
4. **F# Async**: `MailboxProcessor`, `AsyncSeq`
5. **Related**: Phase 6.0 (Backpressure), Phase 6.5 (Agentic Interfaces)

---

## Key Insights

1. **Analog circuits provide intuitive mental models** for cognitive flow control
2. **F# is exceptionally well-suited** for implementing these patterns
3. **Pre-LLM pipelines dramatically improve** prompt quality and system robustness
4. **Configuration-driven circuits** enable TARS to evolve its own architecture
5. **Circuit analogies bridge** traditional systems engineering and AI architecture

This framework transforms TARS v2 into a **"Cognitive Circuit Board"** where intelligence flows through well-designed, controllable pathways.
