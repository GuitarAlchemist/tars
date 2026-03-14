# F# Migration Documentation

This directory contains documentation for the F# migration of the TARS engine.

## Table of Contents

1. [Migration Guide](MigrationGuide.md)
2. [Implementation Comparison](ImplementationComparison.md)
3. [Code Examples](CodeExamples.md)
4. [Best Practices](BestPractices.md)
5. [Roadmap](Roadmap.md)

## Overview

The TARS engine is being migrated from a mixed C#/F# architecture to a cleaner architecture with a dedicated F# core and C# adapters. This migration addresses several issues with the current implementation:

- Namespace collisions between C# and F# code
- Type mismatches between different parts of the codebase
- Compilation errors in the existing F# code
- Lack of proper separation between the F# core and C# code

The new architecture consists of:

- `TarsEngine.FSharp.Core`: A clean F# implementation of core functionality
- `TarsEngine.FSharp.Adapters`: C# adapters that allow C# code to use the F# implementation
- `TarsEngine.Services.Abstractions`: C# interfaces that define the contract between C# and F# code

## Getting Started

If you're new to the F# migration, start with the [Migration Guide](MigrationGuide.md). It provides an overview of the migration process and explains the differences between the old and new implementations.

If you're looking for examples of how to use the new F# implementation, check out the [Code Examples](CodeExamples.md). It provides detailed examples of how to use the new F# implementation from both F# and C# code.

If you're interested in the differences between the old and new implementations, check out the [Implementation Comparison](ImplementationComparison.md). It provides a detailed comparison of the old and new implementations.

If you're looking for best practices for using the new F# implementation, check out the [Best Practices](BestPractices.md). It provides best practices for using the new F# implementation from both F# and C# code.

If you're interested in the roadmap for the F# migration, check out the [Roadmap](Roadmap.md). It provides a phased approach to migration, with each phase building on the previous one.

## Contributing

If you'd like to contribute to the F# migration, please follow these steps:

1. Read the [Migration Guide](MigrationGuide.md) to understand the migration process.
2. Read the [Best Practices](BestPractices.md) to understand how to write good F# code.
3. Check the [Roadmap](Roadmap.md) to see what needs to be done.
4. Pick a task from the roadmap and implement it.
5. Submit a pull request with your changes.

## Questions and Feedback

If you have questions or feedback about the F# migration, please open an issue on the GitHub repository or contact the development team.
