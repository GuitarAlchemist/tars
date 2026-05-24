# Maintenance & Operations

Troubleshooting, known issues, and operational guides for TARS v2.

## Troubleshooting

Common issues and their solutions. *(Check subdirectories for specific guides)*

## Cargo Cult Code

**[Cargo Cult Analysis](./cargo_cult_analysis.md)**: Documentation of legacy code that has been removed or flagged for cleanup in TARS v2.

### What is Cargo Cult Code?

Code that is carried over from previous versions without a clear understanding of its purpose or necessity. This often includes:

- Over-engineered abstractions
- Unused or redundant components
- Speculative features for future use

### Removal Log

See [`cargo_cult_analysis.md`](./cargo_cult_analysis.md) for the complete list of removed components and the rationale.

## Known Issues

*(TODO: Maintain a list of known issues and workarounds)*

## Build & Deployment

### Common Build Issues

- **Lock file conflicts**: Delete `bin/` and `obj/` directories
- **Missing dependencies**: Run `dotnet restore`
- **Package cache**: Clear with `dotnet nuget locals all --clear`

### Environment Setup

- **.NET 10.0 SDK**: Required for building
- **CUDA Toolkit**: Optional, for GPU acceleration
- **Docker**: Optional, for containerized deployment

## Monitoring

*(TODO: Add monitoring and observability guides)*

## Backup & Recovery

*(TODO: Add data backup procedures)*
