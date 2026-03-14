# Multi-agent system protocols

**User:** Stephane Pareilleux (spareilleux@gmail.com)  
**Created:** 11/26/2025 0:45:50  
**Updated:** 11/26/2025 0:51:09  
**Exported:** 11/27/2025 0:56:16  
**Link:** [https://chatgpt.com/c/692693f3-1ef8-832d-bf87-fc7b5c24ba1e](https://chatgpt.com/c/692693f3-1ef8-832d-bf87-fc7b5c24ba1e)  

## Prompt:
What's the best internal protocol to build a multi agent system?

## Response:
Pensé pendant13s

Short answer: there _is no single best protocol_, but there **are a few patterns that almost always win**, and you can layer them cleanly:

*   **Transport:** gRPC (HTTP/2) or a lightweight message bus (NATS / ZeroMQ)
*   **Encoding:** Protobuf or MessagePack (JSON only at the edges / for LLMs)
*   **Semantics:** a tiny “agent protocol” with a handful of _intents_ (ASK, TELL, ACT, EVENT, ERROR, etc.)

Let’s unpack this in a way that’s actually usable for building a real multi-agent system.

* * *

1\. Separate three layers in your head
--------------------------------------

Instead of “what protocol”, think in **three layers**:

1.  **Transport layer – how bytes move**
    *   same machine vs. multi-node
    *   low latency vs. “meh it’s fine”
2.  **Encoding layer – how you put structure on the bytes**
    *   strongly typed vs. flexible
3.  **Semantic layer – what an _agent message_ actually _means_**
    *   goal, observation, plan, error, negotiation, etc.

You can mix and match. That keeps you from tying “agent semantics” to “HTTP because that’s what we had lying around”.

* * *

2\. Transport: what tends to work best
--------------------------------------

### a) Single machine / lab setting

For your TARS-style experiments, honestly the best “protocol” is:

*   **In-process message passing**:
    *   C#/F#: `Channel<T>`, MailboxProcessor, or an actor library (Akka.NET style)
    *   No serialization, just pass objects / records
    *   You can always _wrap_ this behind an `IAgentBus` and later swap in networked transport

This gives you:

*   minimal latency
*   no network flakiness
*   easy debugging

### b) Multi-process / multi-machine

When agents are in different processes / containers:

**1\. gRPC over HTTP/2**

*   Good when:
    *   you have a **service-like** interaction (`AskPlan`, `Evaluate`, `GenerateCode`)
    *   you want **streaming** (server/client/bidi) for long-running tasks
*   Pair it with Protobuf for schema evolution.

**2\. NATS / RabbitMQ / Kafka-like pub-sub**

*   Good when:
    *   your agents are **loosely coupled**
    *   you want topics: `belief.updated`, `plan.proposed`, `task.completed`
*   NATS in particular is:
    *   tiny, fast, simple config
    *   perfect as an internal “agent event bus”

**3\. ZeroMQ / nanomsg style sockets**

*   Good when:
    *   you want funky topologies (ring, mesh, pipeline)
    *   you want to stay close to the metal but still avoid raw TCP

If you want one concrete, pragmatic choice:

*   **NATS for events** + **gRPC for direct “RPC-like” calls** is a very solid combo.

* * *

3\. Encoding: what goes inside the envelope
-------------------------------------------

You basically want an **envelope + payload** pattern.

**Envelope (common to all messages)**

```proto
message AgentEnvelope {
  string message_id      = 1;
  string correlation_id  = 2;  // for replies / tracing
  string from_agent      = 3;
  string to_agent        = 4;  // optional for broadcast
  string intent          = 5;  // "ASK", "TELL", "PROPOSE", "ERROR", ...
  string content_type    = 6;  // "proto:SetGoal", "json:LLMCall", etc.
  bytes  payload         = 7;
  int64  timestamp_utc   = 8;
}
```

Then for `payload`:

*   **Protobuf / MessagePack** for internal typed messages:
    *   `SetGoal`, `ReportObservation`, `SubmitPlan`
*   **JSON** only when:
    *   you’re feeding / reading LLMs
    *   or you genuinely need free-form structures

This lets you:

*   keep strong typing where it matters
*   still have flexible blobs where LLMs live

* * *

4\. Semantic protocol: the actually interesting part
----------------------------------------------------

This is where “multi-agent” stops being “microservices but more chaotic”.

You don’t need full FIPA ACL, but you _do_ want a tiny, **opinionated set of “speech acts”**. Example:

*   `ASK` – “please compute something for me / answer a question”
*   `TELL` – “here’s a fact / belief update”
*   `PROPOSE` – “I suggest this plan / action”
*   `ACCEPT` / `REJECT` – “yes/no to a proposal”
*   `ACT` – “execute this concrete action in the environment”
*   `EVENT` – “broadcasted observation, no reply expected”
*   `ERROR` – “I tried, it exploded. Here’s why.”

You can model that as:

```fsharp
type AgentIntent =
    | Ask
    | Tell
    | Propose
    | Accept
    | Reject
    | Act
    | Event
    | Error
```

Agents then implement a very small surface:

```fsharp
type AgentMessage = {
    Envelope : AgentEnvelope
    Intent   : AgentIntent
    Body     : obj // or DU of known internal payloads
}

type IAgent =
    abstract member Id : string
    abstract member Handle : AgentMessage -> Async<unit>
```

That’s your **internal protocol** at the _semantic_ level.  
Transport/encoding can change; this doesn’t.

* * *

5\. Interaction patterns that matter
------------------------------------

On top of those intents you get recurring patterns:

*   **Request–Reply** (`ASK` → `TELL/ERROR`)
*   **One-way commands** (`ACT`)
*   **Gossip / broadcast** (`EVENT`)
*   **Negotiation / bidding** (`PROPOSE` → `ACCEPT/REJECT`)
*   **Blackboard / shared memory**
    *   Agents don’t talk directly, they write to a “world state” and subscribe to changes.

If you define these patterns explicitly, you can:

*   log them
*   visualize them
*   throttle them (backpressure, priority queues, etc.)

* * *

6\. A concrete internal stack I’d recommend for you
---------------------------------------------------

Given your ecosystem (C#/F#, TARS, local experiments, eventual distribution), I’d go with:

1.  **In-process first**
    *   F# MailboxProcessors or C# Channels for agents
    *   `AgentBus` interface wrapping the messaging
2.  **gRPC + Protobuf as your “network skin”**
    *   `Send(AgentEnvelope)` + streaming variant
    *   Only used when agents are _not_ in the same process
3.  **NATS (or Redis streams if you already have Redis) for events**
    *   Topics like `tars.belief.updated`, `tars.plan.candidate`, `tars.diagnostics.alert`
4.  **A minimal semantic protocol:**
    *   `ASK/TELL/PROPOSE/ACCEPT/REJECT/ACT/EVENT/ERROR`
    *   Logged and traced as first-class citizens in your diagnostics

That gives you:

*   simple mental model
*   easy to simulate locally
*   room to grow into multi-process / multi-machine without rewriting the “agent brain”

* * *

In other words: the “best protocol” isn’t a single wire format; it’s a **small, explicit semantic vocabulary** (your agent speech acts) running on top of **boring, battle-tested plumbing** (in-proc bus → gRPC/NATS) that you can evolve without losing your mind.

## Prompt:
I want a standard AI protocol that can work on remote machines

## Response:
Pensé pendant17s

Right now there _isn’t_ a single ISO-style “one protocol to rule them all”… but there **are** two very strong de-facto standards you can lean on that _do_ work nicely across remote machines:

1.  **OpenAI-compatible HTTP API** – for talking to _models_
2.  **MCP (Model Context Protocol)** – for talking to _tools / agents / capabilities_

If you want something standard-ish, future-proof, and remote-friendly, those are the two I’d build around.

* * *

1\. OpenAI-compatible HTTP API (for models anywhere)
----------------------------------------------------

This is the boring but powerful answer: a _huge_ amount of modern AI infra has converged on **“OpenAI-style REST API over HTTP(S)”**:

*   `POST /v1/chat/completions`
*   `POST /v1/completions`
*   `POST /v1/embeddings`
*   Streaming via SSE or WebSocket

Tons of servers implement this:

*   vLLM’s server
*   Ollama’s “OpenAI compatible” endpoints
*   llama.cpp HTTP servers
*   Proprietary clouds that say “OpenAI-compatible”

So as a **remote protocol**, this gives you:

*   Standard request/response shapes
*   Interop between local & remote models
*   Easy to reverse-proxy, load-balance, secure (TLS, auth tokens, etc.)

In a multi-agent system you can treat it as:

> “The standard way an agent calls _any_ model, whether it’s local GPU, a remote GPU box, or a cloud LLM.”

You don’t need to invent anything here. Just define a small interface in your code like:

```csharp
interface ILLMClient {
    Task<ChatResponse> ChatAsync(ChatRequest request, CancellationToken ct);
}
```

…where the underlying client talks **OpenAI-style HTTP** to whatever machine actually hosts the model.

* * *

2\. MCP – Model Context Protocol (for tools / agents / skills)
--------------------------------------------------------------

Where the OpenAI API is about _models_, **MCP** is about _context + tools_:

*   It defines how a **client** (like an IDE / orchestrator / agent) talks to a **server** (tool/agent) that exposes:
    *   **tools** (functions you can call)
    *   **resources** (files, URLs, data sources)
    *   **prompts / templates**
*   It uses **JSON-RPC** over:
    *   stdio (local)
    *   or **WebSocket** (remote-friendly out of the box)

That makes MCP a very strong candidate for:

> “Standard protocol my agents use to talk to remote tools/skills/other agents.”

Because:

*   Each “agent” can expose an MCP server:
    *   Tools = its capabilities (`plan`, `execute`, `analyze_logs`, `update_beliefs`, …)
    *   Resources = its local knowledge (`/memory/beliefs`, `/env/metrics`, etc.)
*   An orchestrator (or other agents) connect to it over **WebSocket** even if it’s on another machine.
*   You get:
    *   discovery (`tools/list`, `resources/list`)
    *   structured parameters
    *   well-defined error handling

Roughly, an MCP call looks like a JSON-RPC request:

```json
{
  "jsonrpc": "2.0",
  "id": "42",
  "method": "tools/call",
  "params": {
    "name": "analyze_log_chunk",
    "arguments": {
      "path": "/logs/agent1.log",
      "startOffset": 0,
      "maxBytes": 4096
    }
  }
}
```

Over WebSocket this is just a message in each direction.  
Over SSH tunnel → WebSocket → remote MCP server? Still the same protocol.

So MCP gives you:

*   A **standard AI-native protocol** (not generic HTTP)
*   Built-in support for:
    *   tools
    *   resources
    *   prompts
*   A transport that’s naturally **remote-capable** (WebSocket) but also works local (stdio).

* * *

3\. How to combine them sanely
------------------------------

You can think of it like this:

*   **OpenAI-style HTTP** = “How I call models (LLMs, embedding models) on any machine.”
*   **MCP** = “How I call _agents/tools_ on any machine.”

A concrete architecture:

1.  **Each remote machine** runs:
    *   One or more **MCP servers** exposing its tools/agents/resources
    *   Optionally a **model server** exposing OpenAI-compatible HTTP
2.  **Your orchestrator / root agent**:
    *   Has an **MCP client** that connects via WebSocket to those remote MCP servers
    *   Has an **LLM client** that speaks OpenAI-style HTTP to remote model endpoints
3.  Inside your own code you don’t care if something is “local vs remote”; you only care:
    *   “Is this a _tool/agent_?” → MCP
    *   “Is this a _model_?” → OpenAI-style HTTP

You now have a **standardized, remote-safe interface** for both.

* * *

4\. What about old agent standards like FIPA / KQML?
----------------------------------------------------

There _are_ older multi-agent standards:

*   **FIPA ACL**
*   **KQML**

They define _speech acts_ (`inform`, `request`, `propose`, etc.) and message semantics. Historically interesting, but:

*   XML-ish / clunky encodings
*   Not widely adopted in modern LLM ecosystems
*   Little out-of-the-box tooling today

For a **2025 LLM + tools + remote machines** world, they’re more academic relics than practical standards. MCP + OpenAI-style HTTP are far more alive and evolving.

* * *

5\. TL;DR recommendation
------------------------

If you want a _standard AI protocol_ that works across remote machines and doesn’t paint you into a corner:

*   Use **OpenAI-compatible HTTP(S)** for:
    *   _Any model calls_ (chat, completions, embeddings)
*   Use **MCP over WebSocket** for:
    *   _Any agent/tool/resource calls_ between machines

And layer your own conversation semantics (ASK/TELL/PROPOSE/etc.) _on top_ of those, instead of inventing a brand-new wire protocol. That way you’re swimming with the current of the ecosystem instead of against it.

## Prompt:
Aside from MCP, what are the emerging standards?

## Response:
The landscape is still wild and evolving, but a few **proto-standards are clearly emerging** around multi-agent AI—even if none are “official ISO specs” yet.  
Think of them like the TCP/IP, REST, and gRPC equivalents slowly crystallizing for agent ecosystems.

Here are the ones that matter right now.

* * *

**1\. LSP-style Protocols for Agents (Language Server Protocol lineage)**
=========================================================================

**LSP** is not an “AI protocol,” but it has quietly become a template for emerging agent protocols:

*   JSON-RPC
*   Declarative capabilities
*   Push/pull notifications
*   Diagnostics, documents, requests, workspace folders

Many new AI frameworks are now explicitly basing themselves on the **LSP pattern**:

### Emerging systems in this family:

*   **SPP – System Prompt Protocol** (unofficial but adopted in several toolchains)
*   **Agent Server Protocols** inside JetBrains AI Assistant (undocumented but LSP-like)
*   **HuggingFace Smoltools Protocol** (Toy-sized LSP for “tools”)
*   **OpenAI’s ReAct-style tool routers** built as LSP-like JSON-RPC

These are converging on the idea:

> “Agents should behave like language servers, not black boxes.”

LSP is becoming the _structural ancestor_ of the modern AI agent protocol.

* * *

**2\. OpenAI “Actions Schema” (JSON Tooling Schema)**
=====================================================

This is what powers:

*   OpenAI function-calling
*   OpenAI ReAct agents
*   Anthropic tool-use
*   Gemini tool definitions

They all use **the same shape**:

```json
{
  "name": "summarize",
  "description": "Summarizes a document",
  "parameters": { "type": "object", "properties": { ... } }
}
```

This JSON Schema shape is becoming a **de facto standard** for:

*   describing agent capabilities
*   generating structured tool calls
*   validating arguments
*   cross-model tool portability

### Why it matters

Any agent infrastructure that uses tool-calling will support this schema.  
It’s universal across OpenAI, Anthropic, Google, xAI, Mistral, Cohere, and local models (Ollama, vLLM, llama.cpp).

That makes it the **closest thing to a universal agent capability schema** so far.

* * *

**3\. OpenAI-Compatible REST APIs (vLLM / Ollama / llama.cpp / Groq / etc.)**
=============================================================================

This is _boring but real_: the entire AI world is converging around **OpenAI’s REST shape** for:

*   Chat Completion
*   Embeddings
*   Audio
*   Images

Even enterprise stacks (Microsoft Azure AI, Snowflake Cortex, Databricks Mosaic) now say:

> “OpenAI-Compatible API”

You can treat this as the **model protocol layer** in a multi-agent architecture.  
It’s stable, vendor-neutral in practice, and universally supported.

* * *

**4\. JAXON / JIO (The Agentic OS Work)**
=========================================

This is early but important.

**Jaxon** (AI OS research) defines:

*   agent identity
*   agent capabilities
*   planning messages
*   memory vector spec
*   streaming event protocol

**JIO (Jaxon IO)** is emerging as:

*   a structured interface for multi-agent planning
*   a machine-readable workflow graph protocol
*   a “semantic circuit board” for agents

Not widely adopted _yet_, but it has legs because:

*   it formalizes agent _interfaces_
*   not tied to any model vendor
*   sits above LLMs, below “agent platforms”

This is likely to influence future agent dialects.

* * *

**5\. Agentic Graph Protocols (GraphRAG lineage)**
==================================================

GraphRAG, MemGPT, LangGraph, AutoGen—they all independently converged on:

*   **graph-based message passing**
*   nodes as agents
*   edges as message queues
*   serializable “steps”
*   streaming JSON messages

These aren’t formally called protocols, but function as one.

Expect a future RFC from the community in this space.

* * *

**6\. FLYTE / Temporal / Cadence Workflow RPC**
===============================================

These aren’t AI protocols _per se_, but they are quickly becoming:

> “The control-plane protocol for reliable agent orchestration.”

They provide:

*   durable steps
*   retry semantics
*   typed payloads
*   timelines
*   saga patterns
*   cancellation
*   scheduled triggers

In many enterprise-scale AI systems, this becomes the **agent orchestration protocol**.

Temporal + JSON = “AI Distributed Workflow Bus.”

* * *

**7\. WASI + Component Model (AI Containers)**
==============================================

This is new and accelerating fast.

The **WASI component model** is becoming:

*   a cross-language binary interface
*   a portable module system
*   a sandboxed environment
*   a way to package “AI tools” as WebAssembly components

Cloudflare Workers, Fermyon, Lunatic, Wasmer, Wasmtime ecosystems are adopting WASI.

This will likely evolve into:

> “The binary plugin protocol for high-performance AI agents.”

* * *

**8\. Agent-to-Agent RPC over WebSockets (JSON-RPC / MsgPack-RPC)**
===================================================================

This is where most cutting-edge frameworks are heading:

*   AutoGen
*   LangGraph
*   CrewAI
*   AG2
*   python-agents
*   Small-Team Agents
*   Mojo Agents

All of them converge on:

*   **WebSocket streams**
*   **JSON-RPC or MsgPack-RPC**
*   **bidirectional streaming**
*   **agent identity + routing**

This is essentially “MCP without the MCP spec.”

* * *

**So what should _you_ adopt for TARS?**
========================================

Given your ambitions (TARS as a self-evolving, multi-agent OS), the smartest move is:

**Use a three-layer hybrid that matches where the standards are heading**
-------------------------------------------------------------------------

### **1\. Model protocol = OpenAI-compatible REST**

Universal, easy to proxy, stable.

### **2\. Agent protocol = JSON-RPC or MsgPack-RPC over WebSockets**

This is the emerging de facto standard.  
Even MCP is “JSON-RPC over WebSocket + schema.”

### **3\. Capability schema = OpenAI Tool Schema (JSON Schema)**

This is the most interoperable, model-friendly format for agent functions.

### **Optional 4: WASI Component Model for performance-critical modules**

This gives TARS a long-term upgrade path beyond Python/C#/F#.

* * *

**The distilled recommendation**
================================

If you want the most future-proof emerging-standard stack **aside from MCP**, choose:

### **Transport**

*   **WebSocket** (bidirectional, low-latency)

### **RPC format**

*   **JSON-RPC or MsgPack-RPC** (lightweight, standard-ish)

### **Capability definition**

*   **OpenAI Tool JSON Schema**

### **Model calls**

*   **OpenAI-compatible REST**

### **Containers**

*   **WASI Component Model (“AI plugins”)**

These 5 pieces are where the entire ecosystem is converging.

* * *

If you want, I can generate a **canonical minimal protocol** that TARS could adopt—cleaner than MCP, more modern than FIPA, and fully aligned with emerging standards.



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)