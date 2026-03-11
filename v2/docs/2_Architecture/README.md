# TARS v2 Architecture

This directory contains the architectural documentation for TARS v2.

## Core Components

### Agent System

- **Event Bus**: Asynchronous message passing between agents
- **Agent Registry**: Thread-safe agent lifecycle management  
- **Agent Factory**: Creation and initialization of agents

### Memory Systems

- **Vector Store**: Semantic search and retrieval
- **Knowledge Graph**: Relational knowledge representation
- **Episodic Memory**: Trace-based experience replay

### DSL & Execution

- **Workflow-of-Thought (WoT) DSL**: Primary workflow language with typed AST, compiler, traces, and promotion pipeline (`Tars.DSL.Wot` + `Tars.Core.WorkflowOfThought`)
- **Metascript Engine**: Lightweight scripting mode for ad-hoc workflows (`Tars.Metascript`). See [DSL Strategy](./DSL_Strategy.md) for the relationship between Metascripts and WoT.
- **Graph Runtime**: Agent state machine orchestration (`Tars.Graph`)
- **Budget Governor**: Resource and cost management
- **Promotion Pipeline**: Compound Engineering 7-step loop for evolving patterns into DSL constructs (`Tars.Evolution`)

### Capabilities & Tooling

- **Tool Registry**: Dynamic discovery and registration of capabilities (`Tars.Tools`)
- **Epistemic Governor**: Knowledge integrity and cognitive regulation (`Tars.Cortex`)
- **Resilience**: Functional circuit breakers and retry policies (`Tars.Core`)

## Key Design Decisions

### F#-First Architecture

TARS v2 is built primarily in F#, leveraging:

- **Immutability** for safer concurrent operations
- **Pattern Matching** for expressive control flow
- **Type Safety** to catch errors at compile time
- **Computation Expressions** for clean async/result handling

### Event-Driven Design

- Agents communicate through semantic messages
- Loose coupling enables dynamic agent composition
- Event sourcing for debugging and replay

### Hexagonal (Ports & Adapters) Pattern

- Core logic isolated from infrastructure
- Easy to swap LLM providers, vector stores, etc.
- Facilitates testing with mock implementations

## Architecture Diagrams

*(TODO: Add system diagrams here)*

## Further Reading

- **DSL Strategy**: [`DSL_Strategy.md`](./DSL_Strategy.md) -- Metascripts vs WoT, promotion pipeline, migration path
- **Metascript Specification**: [`Metascript_Specification.md`](./Metascript_Specification.md)
- **Memory Architecture**: [`../4_Research/Architecture/Memory_Architecture.md`](../4_Research/Architecture/Memory_Architecture.md)
