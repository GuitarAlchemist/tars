# TARS Docker Auto-Coding

This document explains how to use Docker for TARS auto-coding capabilities.

## Overview

TARS can auto-code itself using Docker containers. This approach provides several benefits:

1. **Isolation**: The auto-coding process runs in isolated containers, preventing conflicts with the host system.
2. **Reproducibility**: The Docker containers ensure consistent behavior across different environments.
3. **Scalability**: Multiple TARS instances can run in parallel, each focusing on different tasks.
4. **Resource Management**: Docker provides resource management capabilities, preventing TARS from consuming too many resources.

## Prerequisites

- Docker Desktop 4.40.0 or later
- TARS CLI built and available
- Docker network created for TARS

## Setup

### 1. Create Docker Network

If you haven't already created a Docker network for TARS, run:

```bash
docker network create tars-network
```

### 2. Start MCP Agent

The MCP (Model Context Protocol) agent is the core component for auto-coding. It provides the API endpoints for code generation and execution.

To start the MCP agent, run:

```bash
.\Scripts\Start-McpAgent.ps1
```

This script builds and starts the MCP agent container, which exposes the MCP API on port 8999.

### 3. Test Auto-Coding

To test the auto-coding capabilities, run:

```bash
.\Scripts\Test-AutoCodingDocker.ps1
```

This script creates a test file and sends an auto-coding request to the MCP agent. The MCP agent then improves the code and saves the result.

## Architecture

The Docker auto-coding architecture consists of the following components:

1. **MCP Agent**: The core component that provides the API endpoints for code generation and execution.
2. **Model Runner**: The Ollama container that runs the language model for code generation.
3. **TARS CLI**: The command-line interface for interacting with TARS.

## API Endpoints

The MCP agent exposes the following API endpoints:

- `POST /api/execute`: Execute a command
  - `command`: The command to execute (e.g., "code", "run", "status")
  - `target`: The target for the command (e.g., code description, file path)

## Examples

### Generate Code

```bash
curl -X POST http://localhost:8999/api/execute -H "Content-Type: application/json" -d '{
  "command": "code",
  "target": "Implement a Calculator class with Add, Subtract, Multiply, and Divide methods"
}'
```

### Execute Command

```bash
curl -X POST http://localhost:8999/api/execute -H "Content-Type: application/json" -d '{
  "command": "execute",
  "target": "dotnet build"
}'
```

## Troubleshooting

### MCP Agent Not Starting

If the MCP agent fails to start, check the Docker logs:

```bash
docker logs tars-mcp-agent
```

### Auto-Coding Not Working

If auto-coding is not working, check the following:

1. Ensure the MCP agent is running: `docker ps | grep tars-mcp-agent`
2. Check the MCP agent logs: `docker logs tars-mcp-agent`
3. Verify the test file exists and is writable
4. Check the network connectivity between the containers

## Conclusion

TARS Docker auto-coding provides a powerful and flexible way to enable TARS to improve itself. By leveraging Docker containers, TARS can run multiple instances in parallel, each focusing on different tasks, while maintaining isolation and reproducibility.
