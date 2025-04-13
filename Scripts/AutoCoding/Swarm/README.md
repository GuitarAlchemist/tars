# TARS Swarm Auto-Coding Scripts

This directory contains scripts for TARS Swarm Auto-Coding.

## Overview

TARS Swarm Auto-Coding Scripts are PowerShell scripts that enable TARS to auto-code itself using a swarm of Docker containers. These scripts provide a more advanced and scalable interface for the auto-coding feature.

## Scripts

- `Run-SwarmAutoCode.ps1`: Manages the swarm and synchronizes files between Docker and the host.
- `Run-TarsAutoCodeSwarm.ps1`: Orchestrates the entire swarm auto-coding process.
- `Apply-SwarmImprovements.ps1`: Applies the improvements to the host with backup and rollback capabilities.
- `Test-SwarmImprovements.ps1`: Tests the improvements in Docker.

## Running the Scripts

To run a script, simply execute it from the command line:

```powershell
.\Scripts\AutoCoding\Swarm\Run-TarsAutoCodeSwarm.ps1
```

## Prerequisites

- Docker Desktop 4.40.0 or later
- TARS CLI built and available
- Docker network created for TARS

## Conclusion

TARS Swarm Auto-Coding Scripts provide a more advanced and scalable interface for the auto-coding feature. By using these scripts, you can enable TARS to analyze, improve, and test its own code without human intervention using a swarm of specialized Docker containers.
