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

- **Metascript Engine**: Recursive, macro-capable workflow engine (`Tars.Metascript`)
- **Graph Runtime**: Agent state machine orchestration (`Tars.Graph`)
- **Budget Governor**: Resource and cost management
- **Macro System**: Reusable workflow definitions stored in JSON

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

- **Metascript Specification**: [`Metascript_Specification.md`](./Metascript_Specification.md)
- **Memory Architecture**: [`../4_Research/Architecture/Memory_Architecture.md`](../4_Research/Architecture/Memory_Architecture.md)
