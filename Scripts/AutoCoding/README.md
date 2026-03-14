# TARS Auto-Coding Scripts

This directory contains scripts for TARS Auto-Coding.

## Overview

TARS Auto-Coding Scripts are PowerShell scripts that enable TARS to auto-code itself using Docker containers. These scripts provide a simple and easy-to-use interface for the auto-coding feature.

## Directories

- [Docker](Docker/): Scripts for Docker auto-coding.
- [Swarm](Swarm/): Scripts for Swarm auto-coding.
- [Demos](Demos/): Demo scripts for auto-coding.

## Getting Started

To get started with TARS Auto-Coding, follow these steps:

1. Install Docker Desktop 4.40.0 or later.
2. Build the TARS CLI: `dotnet build TarsCli/TarsCli.csproj`.
3. Create a Docker network: `docker network create tars-network`.
4. Run one of the auto-coding scripts:
   - For Docker Auto-Coding: `.\Scripts\AutoCoding\Docker\Run-AutoCoding-Simple.ps1`.
   - For Swarm Auto-Coding: `.\Scripts\AutoCoding\Swarm\Run-TarsAutoCodeSwarm.ps1`.

## Conclusion

TARS Auto-Coding Scripts provide a simple and easy-to-use interface for the auto-coding feature. By using these scripts, you can enable TARS to analyze, improve, and test its own code without human intervention.
