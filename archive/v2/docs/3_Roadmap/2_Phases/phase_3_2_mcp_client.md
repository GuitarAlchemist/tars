# Phase 3.2: Model Context Protocol (MCP) Client

## Overview

Implement a client for the Model Context Protocol (MCP) to allow TARS to connect to external MCP servers. This enables the agent to access external tools, resources, and prompts in a standardized way, expanding its capabilities without hardcoding integrations.

## Goals

1. **Protocol Support**: Implement the core MCP JSON-RPC protocol (v1.0).
2. **Transport**: Support Stdio transport (primary) for local servers.
3. **Client**: Create a robust `McpClient` to manage connections and requests.
4. **Integration**: Adapt MCP tools to TARS `IAgentTool` interface.
5. **CLI**: Add commands to inspect and use MCP servers.

## Architecture

### Components

* **Tars.Connectors.Mcp**:
  * `McpTypes.fs`: JSON-RPC and MCP domain types (Tool, Resource, etc.).
  * `McpTransport.fs`: `IMcpTransport` interface and `StdioTransport` implementation.
  * `McpClient.fs`: High-level client for `CallTool`, `ListTools`, `ReadResource`.
* **Tars.Tools**:
  * `McpToolAdapter.fs`: Converts `McpTool` -> `IAgentTool`.

### Dependencies

* `System.Text.Json` for serialization.
* `StreamJsonRpc` (optional, or custom lightweight implementation). *Decision: Custom lightweight implementation to avoid heavy dependencies if possible, or use `StreamJsonRpc` if it simplifies things significantly. Let's start with custom as MCP is simple JSON-RPC.*

## Implementation Steps

### Step 1: Types & Protocol

Define the DTOs for JSON-RPC 2.0 and MCP specific payloads (Initialize, ListTools, CallTool, etc.).

### Step 2: Stdio Transport

Implement a transport layer that spawns a process and communicates via Stdio.

* Needs to handle creating the process.
* Needs to read stdout (messages from server) and write to stdin (messages to server).
* Needs to handle stderr (logging).

### Step 3: McpClient

Implement the client logic:

* Handshake (`initialize`).
* Request/Response correlation.
* Error handling.

### Step 4: Tool Integration

Map MCP tools to TARS tools.

* MCP `name` -> TARS `Name`.
* MCP `inputSchema` -> TARS `Schema`.
* MCP `call` -> TARS `ExecuteAsync`.

### Step 5: Testing

* Unit tests for serialization/deserialization.
* Integration test with a simple MCP server (e.g., a dummy python script or node script).

## Acceptance Criteria

* [ ] Can connect to a local MCP server (e.g., `npx -y @modelcontextprotocol/server-filesystem`).
* [ ] Can list tools from the server.
* [ ] Can execute a tool on the server.
* [ ] Can read a resource from the server.
