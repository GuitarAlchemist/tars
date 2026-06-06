# Agent Protocols Analysis & Recommendations

**Date:** November 22, 2025
**Status:** Recommendation
**Context:** Selecting the standard protocols for TARS v2 to ensure interoperability, extensibility, and future-proofing.

---

## Executive Summary

The AI agent landscape is converging on a few key standards. For TARS v2, we recommend a **hybrid approach**:

1. **Model Context Protocol (MCP)** as the *primary* standard for Tool & Data integration (the "USB-C for AI").
2. **Agent-to-Agent (A2A)** for high-level inter-agent collaboration (if we federate later).
3. **Universal Tool Calling Protocol (UTCP)** as a lightweight fallback for simple, direct API calls where MCP is overkill.

**Recommendation:** Adopt **MCP** immediately as the core integration layer.

---

## Protocol Landscape Analysis

### 1. Model Context Protocol (MCP)

* **Steward**: Anthropic (Open Source)
* **Philosophy**: "Client-Server for AI". The LLM is the client; tools are servers.
* **Key Mechanism**: JSON-RPC 2.0 over Stdio/HTTP.
* **Pros**:
  * **Standardization**: One way to connect to Git, Postgres, Filesystem, etc.
  * **Ecosystem**: Rapidly growing library of pre-built servers.
  * **Security**: Clear separation of concerns (Agent doesn't need direct DB credentials; the MCP server handles it).
* **Cons**: "Wrapper Tax" (requires running a separate server process).
* **TARS Fit**: **Critical**. This solves the "how do I give the agent access to X" problem permanently.

### 2. Universal Tool Calling Protocol (UTCP)

* **Steward**: Community
* **Philosophy**: "Direct & Lightweight". No intermediate servers.
* **Key Mechanism**: Standardized JSON schemas for direct API calls.
* **Pros**:
  * **Zero Latency**: No middleman.
  * **Simplicity**: Just a schema.
* **Cons**: Less abstraction; the Agent must handle the raw API complexity.
* **TARS Fit**: **Secondary**. Use for internal, high-performance loops where running an MCP server is too heavy.

### 3. Agent-to-Agent (A2A) Protocol

* **Steward**: Linux Foundation (Google initiated)
* **Philosophy**: "The Internet of Agents".
* **Key Mechanism**: REST/SSE with "Agent Cards" (manifests).
* **Pros**:
  * **Discovery**: Agents can "introduce" themselves and their capabilities.
  * **Streaming**: Native support for long-running tasks.
* **Cons**: Enterprise-focused, potentially heavy for a local coding agent.
* **TARS Fit**: **Future**. Relevant when TARS needs to talk to *other* TARS instances or external enterprise agents.

---

## Detailed Recommendations for TARS V2

### A. The "Tooling" Layer: Adopt MCP

We should implement the **Model Context Protocol** as the backbone of the `Tars.Cortex.Tools` layer.

* **Why**: It allows us to plug in *any* MCP-compliant server (e.g., a "GitHub MCP Server" or "Postgres MCP Server") without writing custom TARS code.
* **Implementation**:
  * TARS acts as an **MCP Client**.
  * We build/use **MCP Servers** for: File System, Git, Terminal, Browser, Vector Store.

### B. The "Internal" Layer: Native F# Interfaces

For internal components (e.g., the `Memory` system), we don't need a protocol. We use **native F# interfaces** (`IMemoryStore`).

* *Don't over-engineer internal communication.*

### C. The "Evolution" Layer: Agent0 Patterns

While not a "protocol" in the standard sense, the **Agent0** pattern (Curriculum/Executor loop) should be the *behavioral protocol* for the Self-Improvement loop.

---

## Implementation Roadmap

1. **Phase 1 (Foundation)**: Implement `McpClient` in `Tars.Cortex.Tools`.
2. **Phase 2 (Integration)**: Connect the standard `filesystem-server` and `git-server` via MCP.
3. **Phase 3 (Expansion)**: Allow users to configure *custom* MCP servers in `tars_config.yaml`.

---

## References

* [Model Context Protocol Docs](https://modelcontextprotocol.io/docs/getting-started/intro)
* [Universal Tool Calling Protocol (UTCP)](https://github.com/universal-tool-calling-protocol/utcp-mcp)
* [Agent-to-Agent (A2A) Protocol](https://github.com/google/agent-to-agent)
