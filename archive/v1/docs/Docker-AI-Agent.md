# Docker AI Agent Integration with TARS

This document explains how to use Docker AI Agent with TARS for enhanced AI capabilities.

## Overview

Docker AI Agent is a feature introduced in Docker Desktop 4.40.0 that provides:

1. **Local LLM Execution**: Run curated LLM models locally through Docker Hub
2. **Shell Command Execution**: Execute shell commands with AI assistance
3. **Security Scanning**: Scan containers for security vulnerabilities
4. **MCP Integration**: Expose capabilities as MCP Servers

TARS integrates with Docker AI Agent to leverage these capabilities for autonomous self-improvement and code generation.

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

### 2. Start Docker AI Agent

Use the TARS CLI to start Docker AI Agent:

```bash
tarscli docker-ai-agent start
```

This command starts the Docker AI Agent container and connects it to the TARS network.

### 3. Bridge with MCP

To enable TARS to use Docker AI Agent capabilities through MCP:

```bash
tarscli docker-ai-agent bridge --mcp-url http://localhost:8999/
```

## Usage

### Available Commands

The TARS CLI provides the following commands for interacting with Docker AI Agent:

- `tarscli docker-ai-agent start`: Start Docker AI Agent
- `tarscli docker-ai-agent stop`: Stop Docker AI Agent
- `tarscli docker-ai-agent status`: Check Docker AI Agent status
- `tarscli docker-ai-agent run-model <model>`: Run a specific model
- `tarscli docker-ai-agent generate <prompt>`: Generate text using Docker AI Agent
- `tarscli docker-ai-agent shell <command>`: Execute a shell command using Docker AI Agent
- `tarscli docker-ai-agent list-models`: List available models
- `tarscli docker-ai-agent bridge --mcp-url <url>`: Bridge Docker AI Agent with MCP

### Example: Text Generation

Generate text using Docker AI Agent:

```bash
tarscli docker-ai-agent generate "Write a function to calculate Fibonacci numbers in C#"
```

### Example: Shell Command Execution

Execute a shell command using Docker AI Agent:

```bash
tarscli docker-ai-agent shell "docker ps"
```

## Integration with TARS Self-Improvement

Docker AI Agent enhances TARS self-improvement capabilities by providing:

1. **Local LLM Execution**: Run models locally for code generation and analysis
2. **Container Management**: Create and manage TARS replica containers
3. **Security Scanning**: Ensure TARS containers are secure
4. **MCP Bridge**: Enable communication between TARS and Docker AI Agent

## Architecture

The Docker AI Agent integration with TARS consists of:

1. **Docker AI Agent Container**: Runs Docker AI Agent with TARS integration
2. **Model Runner Container**: Runs LLM models for Docker AI Agent
3. **TARS MCP Bridge**: Enables communication between TARS and Docker AI Agent
4. **TARS CLI Commands**: Provides commands for interacting with Docker AI Agent

## Troubleshooting

### Docker AI Agent Not Starting

If Docker AI Agent fails to start:

1. Check Docker Desktop is running
2. Ensure Docker Desktop version is 4.40.0 or later
3. Verify Docker network exists: `docker network ls`
4. Check Docker AI Agent logs: `docker logs tars-docker-ai-agent`

### Model Not Found

If a model is not found:

1. List available models: `tarscli docker-ai-agent list-models`
2. Pull the model: `tarscli docker-ai-agent run-model <model>`

### MCP Bridge Not Working

If the MCP bridge is not working:

1. Check MCP is running: `tarscli mcp status`
2. Verify Docker AI Agent is running: `tarscli docker-ai-agent status`
3. Try bridging again: `tarscli docker-ai-agent bridge --mcp-url http://localhost:8999/`

## References

- [Docker AI Agent Documentation](https://docs.docker.com/desktop/ai-agent/)
- [Docker Model Hub](https://hub.docker.com/search?q=&type=image&category=ai)
- [TARS MCP Documentation](features/model-context-protocol.md)
