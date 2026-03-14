# Getting Started with TARS v2

Welcome! This guide will help you get started with TARS v2.

## Prerequisites

- **.NET 10.0 SDK** or later
- **Docker** (optional, for containerized deployment)
- **CUDA-capable GPU** (optional, for accelerated inference)

## Quick Start

### 1. Build the Project

```bash
cd v2
dotnet build
```

### 2. Run Your First Command

```bash
# Run the demo ping command
dotnet run --project src/Tars.Interface.Cli -- demo-ping

# Start interactive chat
dotnet run --project src/Tars.Interface.Cli -- chat
```

### 3. Explore Examples

Check out the example metascripts in `/examples`:

- Basic agent workflows
- Multi-agent coordination
- Knowledge retrieval patterns

## Next Steps

- **Architecture**: Learn about the [system design](../2_Architecture/)
- **Roadmap**: See what's being [built next](../3_Roadmap/1_Plans/implementation_plan.md)
- **Testing**: Run the [test suite](../5_Quality/Testing_Tips.md)

## Need Help?

- Review [Troubleshooting](../6_Maintenance/)
- Check the [API Reference](../7_Reference/)
