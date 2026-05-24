# TARS Auto-Coding with Docker

This document explains how to use TARS Auto-Coding with Docker.

## Overview

TARS Auto-Coding is a feature that allows TARS to automatically improve code by analyzing it and generating improved versions. When used with Docker, TARS can leverage containerized language models for code generation without requiring local installation of these models.

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

The Ollama container provides the language model for code generation. To start it, run:

```bash
docker-compose -f docker-compose-auto-coding.yml up -d
```

This will start an Ollama container with the necessary configuration for auto-coding.

### 3. Pull the Required Model

Pull the llama3 model (or another model of your choice):

```bash
docker exec -it tars-model-runner ollama pull llama3
```

## Usage

### Improve a Single File

To improve a single file using auto-coding with Docker:

```bash
# Set environment variables for Docker
$env:OLLAMA_USE_DOCKER = "true"
$env:OLLAMA_BASE_URL = "http://localhost:8080"

# Run the auto-coding command
dotnet TarsCli/bin/Debug/net9.0/tarscli.dll self-code improve <file-path> --model llama3 --auto-apply
```

Replace `<file-path>` with the path to the file you want to improve.

### Using the Script

For convenience, you can use the provided PowerShell script:

```bash
.\Scripts\Run-AutoCoding-Docker.ps1
```

This script will:
1. Check if Docker is running
2. Create a Docker network if it doesn't exist
3. Start the Ollama container
4. Pull the llama3 model
5. Run the auto-coding command on the example file

## Configuration

You can configure the auto-coding feature by setting the following environment variables:

- `OLLAMA_USE_DOCKER`: Set to "true" to use Docker for Ollama
- `OLLAMA_BASE_URL`: The base URL for the Ollama API (default: "http://localhost:8080" when using Docker)

## Troubleshooting

### Ollama Container Not Starting

If the Ollama container fails to start, check the Docker logs:

```bash
docker logs tars-model-runner
```

### Model Not Found

If you get an error about the model not being found, make sure you've pulled the model:

```bash
docker exec -it tars-model-runner ollama pull llama3
```

### Connection Refused

If you get a "connection refused" error, make sure the Ollama container is running:

```bash
docker ps | Select-String "tars-model-runner"
```

## Restoring Original Files

After testing auto-coding, you can restore the original example file:

```bash
.\Scripts\Restore-AutoCodingExample.ps1
```

## Advanced Usage

For more advanced usage, including batch processing and custom models, see the main Auto-Coding documentation.
