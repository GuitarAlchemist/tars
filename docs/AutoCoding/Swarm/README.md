# TARS Swarm Auto-Coding

This document provides an overview of the TARS Swarm Auto-Coding feature, which enables TARS to auto-code itself using a swarm of Docker containers.

## Overview

TARS Swarm Auto-Coding is a feature that allows TARS to analyze, improve, and test its own code without human intervention. This feature is implemented using a swarm of specialized Docker containers, which provide isolation, reproducibility, and scalability.

## Documentation

- [Swarm Auto-Coding Documentation](README-Swarm-Auto-Coding-Final.md): Comprehensive documentation for the swarm auto-coding feature.
- [Swarm Auto-Coding Status](../TARS-Auto-Coding-Status.md): Current status of the swarm auto-coding feature.

## Scripts

TARS provides several scripts for swarm auto-coding:

- `Run-SwarmAutoCode.ps1`: Manages the swarm and synchronizes files between Docker and the host.
- `Run-TarsAutoCodeSwarm.ps1`: Orchestrates the entire swarm auto-coding process.
- `Apply-SwarmImprovements.ps1`: Applies the improvements to the host with backup and rollback capabilities.
- `Test-SwarmImprovements.ps1`: Tests the improvements in Docker.

## Getting Started

To get started with TARS Swarm Auto-Coding, follow these steps:

1. Install Docker Desktop 4.40.0 or later.
2. Build the TARS CLI: `dotnet build TarsCli/TarsCli.csproj`.
3. Create a Docker network: `docker network create tars-network`.
4. Build the Docker images: `docker-compose -f docker-compose-swarm.yml build`.
5. Run the swarm auto-coding process: `.\Scripts\AutoCoding\Swarm\Run-TarsAutoCodeSwarm.ps1`.

## Conclusion

TARS Swarm Auto-Coding is a powerful feature that enables TARS to improve itself without human intervention. By using a swarm of specialized Docker containers, TARS can analyze, improve, and test its own code in a safe and reproducible way.
