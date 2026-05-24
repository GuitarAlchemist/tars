# Docker Model Runner Setup

TARS supports multiple model providers for inference, including Ollama and Docker Model Runner. This document explains how to set up and use Docker Model Runner with TARS.

## Overview

Docker Model Runner is a containerized service that provides an API for running language models. It's designed to be a flexible and scalable solution for model inference, allowing you to run models in a containerized environment.

In the current implementation, Docker Model Runner uses Ollama as the backend, providing an OpenAI-compatible API for model inference.

## Prerequisites

- Docker Desktop installed and running
- Docker Compose installed
- Ollama installed (optional, as it's included in the Docker setup)

## Setup

### 1. Create a Docker Compose file

Create a file named `docker-compose-model-runner.yml` with the following content:

```yaml
version: '3.8'

services:
  model-runner:
    image: ollama/ollama:latest
    ports:
      - "8080:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

volumes:
  ollama_data:
```

This configuration:
- Uses the official Ollama Docker image
- Maps port 8080 on the host to port 11434 in the container
- Creates a volume for Ollama data
- Enables GPU acceleration if available

### 2. Start the Docker Model Runner

Run the following command to start the Docker Model Runner:

```bash
docker-compose -f docker-compose-model-runner.yml up -d
```

### 3. Configure TARS to use Docker Model Runner

The Docker Model Runner configuration is in the `appsettings.json` file:

```json
"DockerModelRunner": {
  "BaseUrl": "http://localhost:8080",
  "DefaultModel": "llama3:8b"
}
```

## Usage

Once the Docker Model Runner is set up, TARS can use it as a model provider. The Docker Model Runner will automatically pull models as needed.

You can test the Docker Model Runner with the following command:

```bash
dotnet run --project TarsCli/TarsCli.csproj -- demo model-providers
```

This will show the available model providers and generate text using each provider.

## Troubleshooting

### No models available

If the Docker Model Runner shows "No models available", you can pull a model manually:

```bash
docker exec -it tars-model-runner-1 ollama pull llama3
```

### Connection refused

If you get a "Connection refused" error, make sure the Docker Model Runner is running:

```bash
docker ps
```

If it's not running, start it with:

```bash
docker-compose -f docker-compose-model-runner.yml up -d
```

## Implementation Details

The Docker Model Runner service in TARS is implemented in `TarsCli/Services/DockerModelRunnerService.cs`. It provides methods for:

- Checking if Docker Model Runner is available
- Getting available models
- Generating completions
- Generating chat completions

The service automatically pulls models if they're not available, and handles streaming responses from the Ollama API.
