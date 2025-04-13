# TARS Docker AI Agent Integration

This integration enables TARS to leverage Docker AI Agent capabilities for enhanced AI-powered operations.

## Overview

Docker AI Agent is a feature introduced in Docker Desktop 4.40.0 that provides:

1. **Local LLM Execution**: Run curated LLM models locally through Docker Hub
2. **Shell Command Execution**: Execute shell commands with AI assistance
3. **Security Scanning**: Scan containers for security vulnerabilities
4. **MCP Integration**: Expose capabilities as MCP Servers

TARS integrates with Docker AI Agent to leverage these capabilities for autonomous self-improvement and code generation.

## Getting Started

### Prerequisites

- Docker Desktop 4.40.0 or later
- TARS CLI built and available
- Docker network created for TARS

### Setup

1. Create Docker network:
   ```bash
   docker network create tars-network
   ```

2. Start Docker AI Agent:
   ```bash
   tarscli docker-ai-agent start
   ```

3. Bridge with MCP:
   ```bash
   tarscli docker-ai-agent bridge --mcp-url http://localhost:8999/
   ```

## Demo

Run the Docker AI Agent demo to see the integration in action:

```bash
tarscli demo docker-ai-agent
```

Or use the PowerShell script:

```powershell
.\Scripts\Run-DockerAIAgent.ps1
```

## Commands

The TARS CLI provides the following commands for interacting with Docker AI Agent:

- `tarscli docker-ai-agent start`: Start Docker AI Agent
- `tarscli docker-ai-agent stop`: Stop Docker AI Agent
- `tarscli docker-ai-agent status`: Check Docker AI Agent status
- `tarscli docker-ai-agent run-model <model>`: Run a specific model
- `tarscli docker-ai-agent generate <prompt>`: Generate text using Docker AI Agent
- `tarscli docker-ai-agent shell <command>`: Execute a shell command using Docker AI Agent
- `tarscli docker-ai-agent list-models`: List available models
- `tarscli docker-ai-agent bridge --mcp-url <url>`: Bridge Docker AI Agent with MCP

## Architecture

The Docker AI Agent integration with TARS consists of:

1. **Docker AI Agent Container**: Runs Docker AI Agent with TARS integration
2. **Model Runner Container**: Runs LLM models for Docker AI Agent
3. **TARS MCP Bridge**: Enables communication between TARS and Docker AI Agent
4. **TARS CLI Commands**: Provides commands for interacting with Docker AI Agent

## Files

- `Dockerfile.docker-ai-agent`: Dockerfile for the Docker AI Agent container
- `docker-compose-docker-ai-agent.yml`: Docker Compose file for the Docker AI Agent setup
- `TarsCli\Services\DockerAIAgentService.cs`: Service for interacting with Docker AI Agent
- `TarsCli\Commands\DockerAIAgentCommand.cs`: CLI commands for Docker AI Agent
- `TarsCli\Commands\DockerAIAgentDemoCommand.cs`: Demo command for Docker AI Agent
- `Scripts\Run-DockerAIAgent.ps1`: PowerShell script for running the Docker AI Agent demo
- `Scripts\Run-DockerAIAgent.bat`: Batch file for running the Docker AI Agent demo
- `docs\Docker-AI-Agent.md`: Documentation for Docker AI Agent integration

## Documentation

For more detailed information, see the [Docker AI Agent documentation](docs/Docker-AI-Agent.md).

## Troubleshooting

If you encounter issues with the Docker AI Agent integration:

1. Check Docker Desktop is running
2. Ensure Docker Desktop version is 4.40.0 or later
3. Verify Docker network exists: `docker network ls`
4. Check Docker AI Agent logs: `docker logs tars-docker-ai-agent`

## References

- [Docker AI Agent Documentation](https://docs.docker.com/desktop/ai-agent/)
- [Docker Model Hub](https://hub.docker.com/search?q=&type=image&category=ai)
- [TARS MCP Documentation](docs/MCP-Protocol.md)
