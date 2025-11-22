# TARS Docker Auto-Coding

This document provides an overview of the TARS Docker Auto-Coding feature, which enables TARS to auto-code itself using Docker containers.

## Overview

TARS Docker Auto-Coding is a feature that allows TARS to analyze, improve, and test its own code without human intervention. This feature is implemented using Docker containers, which provide isolation, reproducibility, and scalability.

## Documentation

- [Docker Auto-Coding Documentation](README-Docker-Auto-Coding-Final.md): Comprehensive documentation for the Docker auto-coding feature.
- [Docker Auto-Coding Status](../TARS-Auto-Coding-Status.md): Current status of the Docker auto-coding feature.

## Scripts

TARS provides several scripts for Docker auto-coding:

- `Run-AutoCoding.ps1`: Finds TODOs in the codebase and improves them.
- `Run-AutoCoding-Simple.ps1`: A simple script that demonstrates auto-coding by improving a test file.
- `Test-AutoCoding.ps1`: Tests the auto-coding capabilities.
- `Test-AutoCodingDocker.ps1`: Tests the auto-coding capabilities using Docker.

## Getting Started

To get started with TARS Docker Auto-Coding, follow these steps:

1. Install Docker Desktop 4.40.0 or later.
2. Build the TARS CLI: `dotnet build TarsCli/TarsCli.csproj`.
3. Create a Docker network: `docker network create tars-network`.
4. Run one of the auto-coding scripts: `.\Scripts\AutoCoding\Docker\Run-AutoCoding-Simple.ps1`.

## Conclusion

TARS Docker Auto-Coding is a powerful feature that enables TARS to improve itself without human intervention. By using Docker containers, TARS can analyze, improve, and test its own code in a safe and reproducible way.
