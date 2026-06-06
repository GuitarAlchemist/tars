# TARS Swarm Auto-Coding

This document explains how to use the TARS Swarm Auto-Coding feature, which enables TARS to auto-code itself using a swarm of Docker containers.

## Overview

The TARS Swarm Auto-Coding feature uses a swarm of specialized Docker containers to analyze, improve, and test code. The process is as follows:

1. **Analysis**: Analyze code for potential improvements
2. **Generation**: Generate improved code
3. **Testing**: Test the improved code to ensure it works correctly
4. **Application**: Apply the improvements to the codebase

The swarm consists of the following containers:

- **Coordinator**: Manages the swarm and coordinates the auto-coding process
- **Analyzer**: Analyzes code for potential improvements
- **Generator**: Generates improved code
- **Tester**: Builds and tests the improved code
- **Model Runner**: Provides the language model for code generation

## Prerequisites

- Docker Desktop 4.40.0 or later
- TARS CLI built and available
- .NET SDK 9.0 or later

## Setup

### 1. Create Docker Network

If you haven't already created a Docker network for TARS, run:

```bash
docker network create tars-network
```

### 2. Build TARS CLI

Make sure the TARS CLI is built and available:

```bash
dotnet build TarsCli/TarsCli.csproj
```

## Usage

### Running the Swarm Auto-Coding Process

To run the swarm auto-coding process, use the `Run-TarsAutoCodeSwarm.ps1` script:

```bash
.\Scripts\Run-TarsAutoCodeSwarm.ps1 -TargetDirectories TarsCli,TarsEngine -AgentCount 3 -Model llama3 -AutoApply
```

Parameters:

- `-TargetDirectories`: Directories to target for improvement (default: TarsCli, TarsEngine)
- `-AgentCount`: Number of agents to create (default: 3)
- `-Model`: Model to use for improvement (default: llama3)
- `-AutoApply`: Automatically apply improvements (default: false)
- `-SkipTests`: Skip testing the improved code (default: false)

### Applying Improvements Manually

If you don't use the `-AutoApply` parameter, you can apply the improvements manually after the process completes:

```bash
.\Scripts\Apply-SwarmImprovements.ps1
```

This script will:

1. Create backups of the original files
2. Apply the improvements
3. Build and test the solution
4. Roll back changes if the build or tests fail

### Testing Improvements

To test the improvements without applying them:

```bash
.\Scripts\Test-SwarmImprovements.ps1
```

### Building TARS in Docker

To build the TARS solution in Docker:

```bash
.\Scripts\Build-TarsInDocker.ps1
```

## Architecture

The TARS Swarm Auto-Coding architecture consists of the following components:

### 1. Docker Containers

- **tars-coordinator**: Manages the swarm and coordinates the auto-coding process
- **tars-analyzer**: Analyzes code for potential improvements
- **tars-generator**: Generates improved code
- **tars-tester**: Builds and tests the improved code
- **tars-model-runner**: Provides the language model for code generation

### 2. Shared Volumes

- **tars-shared**: Shared volume for exchanging data between containers
- **workspace**: Mount of the host workspace for accessing the codebase

### 3. Network

- **tars-network**: Docker network for communication between containers

## Process Flow

1. **Initialization**:
   - Start the Docker containers
   - Initialize the swarm

2. **Analysis**:
   - Analyze code for potential improvements
   - Identify files that can be improved

3. **Generation**:
   - Generate improved code
   - Store improvements in the shared volume

4. **Testing**:
   - Build the solution with the improvements
   - Run tests to ensure the improvements don't break existing functionality

5. **Application**:
   - Create backups of the original files
   - Apply the improvements
   - Build and test the solution
   - Roll back changes if the build or tests fail

## Troubleshooting

### Swarm Not Starting

If the swarm doesn't start, check the Docker logs:

```bash
docker logs tars-coordinator
```

### Improvements Not Being Applied

If improvements aren't being applied, check the following:

1. Make sure the tests pass
2. Check the Docker logs for errors
3. Make sure the files are writable

### Build or Tests Failing

If the build or tests fail after applying improvements, the changes will be automatically rolled back. Check the logs for details on what failed.

## Conclusion

The TARS Swarm Auto-Coding feature provides a powerful way for TARS to improve itself using a swarm of Docker containers. By analyzing, improving, and testing code in isolation, TARS can safely apply improvements to its own codebase without breaking existing functionality.
