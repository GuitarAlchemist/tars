# MCP Swarm

The MCP Swarm is a feature of TARS that enables the deployment and management of multiple MCP agents in Docker containers. These agents can work together to perform complex tasks, such as self-improvement.

## Overview

The MCP Swarm consists of the following components:

1. **TarsMcpSwarmService**: A service for managing a swarm of TARS MCP servers in Docker containers.
2. **SwarmSelfImprovementService**: A service for self-improvement using a swarm of MCP agents.
3. **McpSwarmCommand**: A command-line interface for managing the MCP swarm.
4. **SwarmSelfImprovementCommand**: A command-line interface for managing the swarm self-improvement process.

## Architecture

The MCP Swarm architecture is designed to be scalable and flexible:

- Each agent runs in its own Docker container with its own MCP server.
- Agents communicate with each other through the MCP protocol.
- The swarm manager coordinates the agents and distributes tasks.
- The self-improvement service uses the swarm to analyze and improve code.

## Agent Roles

Agents in the swarm can have different roles:

- **Code Analyzer**: Analyzes code for potential improvements.
- **Code Generator**: Generates improved code based on analysis.
- **Test Generator**: Generates and runs tests for the improved code.
- **Documentation Generator**: Generates and updates documentation.
- **Project Manager**: Manages tasks and prioritizes improvements.

## Using the MCP Swarm

### Creating and Managing Agents

To create and manage agents in the swarm, use the `mcp-swarm` command:

```bash
# Create a new agent
tarscli mcp-swarm create <name> <role> [--capabilities <capabilities>] [--metadata <metadata>]

# Start an agent
tarscli mcp-swarm start <id>

# Stop an agent
tarscli mcp-swarm stop <id>

# Remove an agent
tarscli mcp-swarm remove <id>

# List all agents
tarscli mcp-swarm list

# Start all agents
tarscli mcp-swarm start-all

# Stop all agents
tarscli mcp-swarm stop-all

# Send a request to an agent
tarscli mcp-swarm send <id> <action> <operation> [--content <content>] [--file <file>]
```

### Self-Improvement with the MCP Swarm

To use the MCP swarm for self-improvement, use the `swarm-improve` command:

```bash
# Start the self-improvement process
tarscli swarm-improve start --target <directories> [--agent-count <count>] [--model <model>]

# Stop the self-improvement process
tarscli swarm-improve stop

# Get the status of the self-improvement process
tarscli swarm-improve status
```

### Running the MCP Swarm Demo

To see the MCP swarm in action, run the demo:

```bash
tarscli demo mcp-swarm-demo
```

## Configuration

The MCP swarm is configured in the `appsettings.json` file:

```json
"Tars": {
  "McpSwarm": {
    "ConfigPath": "config/mcp-swarm.json",
    "DockerComposeTemplatePath": "templates/docker-compose-mcp-agent.yml",
    "DockerComposeOutputDir": "docker/mcp-agents"
  },
  "SelfImprovement": {
    "AutoApply": false,
    "DefaultModel": "llama3",
    "TargetDirectories": ["TarsCli", "TarsEngine", "TarsEngine.SelfImprovement"]
  }
}
```

## Implementation Details

### TarsMcpSwarmService

The `TarsMcpSwarmService` is responsible for managing the swarm of MCP agents. It provides methods for:

- Creating agents
- Starting agents
- Stopping agents
- Removing agents
- Getting agent status
- Sending requests to agents

### SwarmSelfImprovementService

The `SwarmSelfImprovementService` uses the MCP swarm for self-improvement. It:

1. Creates a swarm of agents with different roles
2. Analyzes code for potential improvements
3. Generates improvements
4. Applies improvements
5. Tests the improved code

### Docker Integration

The MCP swarm uses Docker to run agents in containers. Each agent has its own Docker Compose file and container. The swarm manager creates and manages these containers.

## Future Enhancements

Planned enhancements for the MCP swarm include:

1. **Agent Specialization**: More specialized agent roles for specific tasks.
2. **Learning and Adaptation**: Agents that learn from their experiences and adapt their behavior.
3. **Collaborative Problem Solving**: Agents that work together to solve complex problems.
4. **Distributed Processing**: Agents that can run on multiple machines for better scalability.
5. **Autonomous Improvement**: A fully autonomous self-improvement process that can run without human intervention.

## Conclusion

The MCP Swarm is a powerful feature of TARS that enables distributed, collaborative AI. By deploying multiple agents in Docker containers, TARS can perform complex tasks and improve itself autonomously.
