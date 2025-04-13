# TARS Auto-Coding

This document provides an overview of the TARS Auto-Coding feature, which enables TARS to auto-code itself using Docker containers.

## Overview

TARS Auto-Coding is a feature that allows TARS to analyze, improve, and test its own code without human intervention. This feature is implemented using Docker containers, which provide isolation, reproducibility, and scalability.

## Auto-Coding Approaches

TARS supports two approaches to auto-coding:

1. **Docker Auto-Coding**: A simple approach that uses a single Docker container to improve code.
2. **Swarm Auto-Coding**: A more advanced approach that uses a swarm of specialized Docker containers to analyze, improve, and test code.

### Docker Auto-Coding

The Docker Auto-Coding approach uses a single Docker container to improve code. This approach is simpler and easier to set up, but it doesn't provide the same level of isolation and scalability as the Swarm Auto-Coding approach.

For more information, see the [Docker Auto-Coding documentation](Docker/README-Docker-Auto-Coding-Final.md).

### Swarm Auto-Coding

The Swarm Auto-Coding approach uses a swarm of specialized Docker containers to analyze, improve, and test code. This approach provides better isolation, reproducibility, and scalability, but it's more complex to set up and manage.

For more information, see the [Swarm Auto-Coding documentation](Swarm/README-Swarm-Auto-Coding-Final.md).

## Scripts

TARS provides several scripts for auto-coding:

### Docker Auto-Coding Scripts

- `Run-AutoCoding.ps1`: Finds TODOs in the codebase and improves them.
- `Run-AutoCoding-Simple.ps1`: A simple script that demonstrates auto-coding by improving a test file.
- `Test-AutoCoding.ps1`: Tests the auto-coding capabilities.
- `Test-AutoCodingDocker.ps1`: Tests the auto-coding capabilities using Docker.

### Swarm Auto-Coding Scripts

- `Run-SwarmAutoCode.ps1`: Manages the swarm and synchronizes files between Docker and the host.
- `Run-TarsAutoCodeSwarm.ps1`: Orchestrates the entire swarm auto-coding process.
- `Apply-SwarmImprovements.ps1`: Applies the improvements to the host with backup and rollback capabilities.
- `Test-SwarmImprovements.ps1`: Tests the improvements in Docker.

### Demo Scripts

- `Run-SwarmAutoCode-Test.ps1`: Tests the swarm auto-coding process with a simple example.
- `Test-SwarmAutoCode-Simple.ps1`: A simple script that demonstrates swarm auto-coding.

## Getting Started

To get started with TARS Auto-Coding, follow these steps:

1. Install Docker Desktop 4.40.0 or later.
2. Build the TARS CLI: `dotnet build TarsCli/TarsCli.csproj`.
3. Create a Docker network: `docker network create tars-network`.
4. Run one of the auto-coding scripts:
   - For Docker Auto-Coding: `.\Scripts\AutoCoding\Docker\Run-AutoCoding-Simple.ps1`.
   - For Swarm Auto-Coding: `.\Scripts\AutoCoding\Swarm\Run-TarsAutoCodeSwarm.ps1`.

## Conclusion

TARS Auto-Coding is a powerful feature that enables TARS to improve itself without human intervention. By using Docker containers, TARS can analyze, improve, and test its own code in a safe and reproducible way.
