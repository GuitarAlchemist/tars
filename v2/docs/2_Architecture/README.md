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

- **Metascript Engine**: Custom workflow DSL execution
- **Graph Runtime**: Agent state machine orchestration
- **Budget Governor**: Resource and cost management

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
- **Memory Architecture**: [`../4_Research/Memory_Architecture.md`](../4_Research/Memory_Architecture.md)
