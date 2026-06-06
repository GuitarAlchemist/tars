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

### 2. Start Ollama Container

The Ollama container is the core component for auto-coding. It provides the language model for code generation.

To start the Ollama container, run:

```bash
docker-compose -f docker-compose-simple.yml up -d
```

### 3. Test Auto-Coding

To test the auto-coding capabilities, run:

```bash
.\Scripts\Run-AutoCoding-Simple.ps1
```

This script creates a test file and improves the code using TARS auto-coding capabilities.

## Scripts

The following scripts are available for auto-coding:

- `Run-AutoCoding-Simple.ps1`: A simple script that demonstrates auto-coding by improving a test file.
- `Run-DockerAIAgent-Direct.ps1`: A script that runs the Docker AI Agent directly to improve a test file.
- `Run-AutoCoding.ps1`: A script that finds TODOs in the codebase and improves them.

## Architecture

The Docker auto-coding architecture consists of the following components:

1. **Ollama Container**: The container that runs the language model for code generation.
2. **Docker AI Agent**: The container that provides the API endpoints for code generation and execution.
3. **TARS CLI**: The command-line interface for interacting with TARS.

## Troubleshooting

### Ollama Container Not Starting

If the Ollama container fails to start, check the Docker logs:

```bash
docker logs tars-model-runner
```

### Auto-Coding Not Working

If auto-coding is not working, check the following:

1. Ensure the Ollama container is running: `docker ps | grep ollama`
2. Check the Ollama container logs: `docker logs tars-model-runner`
3. Verify the test file exists and is writable
4. Check the network connectivity between the containers

## Conclusion

TARS Docker auto-coding provides a powerful and flexible way to enable TARS to improve itself. By leveraging Docker containers, TARS can run multiple instances in parallel, each focusing on different tasks, while maintaining isolation and reproducibility.

The auto-coding capabilities demonstrated in this document are just the beginning. With further development, TARS can become fully autonomous, implementing its own TODOs, fixing bugs, and adding new features without human intervention.
