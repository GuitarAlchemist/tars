# F# Migration Guide

This guide explains how to migrate code from the old F# implementation in `TarsEngine.TreeOfThought` to the new F# implementation in `TarsEngine.FSharp.Core`.

## Table of Contents

1. [Overview](#overview)
2. [Key Differences](#key-differences)
3. [Migration Steps](#migration-steps)
4. [Examples](#examples)
5. [Troubleshooting](#troubleshooting)

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

## Key Differences

### Namespace Structure

**Old Structure**:
```
TarsEngine.FSharp.MetascriptToT
TarsEngine.FSharp.MetascriptGeneration
TarsEngine.FSharp.MetascriptExecution
```

**New Structure**:
```
TarsEngine.FSharp.Core.TreeOfThought
TarsEngine.FSharp.Core.Core
TarsEngine.FSharp.Adapters.TreeOfThought
```

### Type Names

**Old Types**:
```fsharp
type MetascriptThoughtNode = { ... }
type MetascriptEvaluationMetrics = { ... }
```

**New Types**:
```fsharp
type ThoughtNode = { ... }
type EvaluationMetrics = { ... }
```

### Function Names

**Old Functions**:
```fsharp
let createNode thought = ...
let addChild parent child = ...
let evaluateNode node metrics = ...
```

**New Functions**:
```fsharp
let createNode thought = ...
let addChild parent child = ...
let evaluateNode node metrics = ...
```

The function names are largely the same, but they operate on different types and are in different namespaces.

## Migration Steps

1. **Identify Dependencies**: Identify all code that depends on the old F# implementation.

2. **Update References**: Update references from `TarsEngine.TreeOfThought` to `TarsEngine.FSharp.Core` and `TarsEngine.FSharp.Adapters`.

3. **Update Type Usage**: Update code to use the new types instead of the old types.

4. **Use Adapters**: Use the adapter classes to convert between the old and new types if necessary.

5. **Update Function Calls**: Update function calls to use the new functions instead of the old functions.

6. **Test**: Test the migrated code to ensure it works correctly.

## Examples

### Example 1: Creating a Thought Node

**Old Code**:
```fsharp
open TarsEngine.FSharp.MetascriptToT

let node = createNode "My thought"
```

**New Code (F#)**:
```fsharp
open TarsEngine.FSharp.Core.TreeOfThought

let node = ThoughtNode.createNode "My thought"
```

**New Code (C#)**:
```csharp
using TarsEngine.FSharp.Adapters.TreeOfThought;

var node = ThoughtNodeAdapter.CreateNode("My thought");
```

### Example 2: Adding a Child Node

**Old Code**:
```fsharp
open TarsEngine.FSharp.MetascriptToT

let parent = createNode "Parent"
let child = createNode "Child"
let updatedParent = addChild parent child
```

**New Code (F#)**:
```fsharp
open TarsEngine.FSharp.Core.TreeOfThought

let parent = ThoughtNode.createNode "Parent"
let child = ThoughtNode.createNode "Child"
let updatedParent = ThoughtNode.addChild parent child
```

**New Code (C#)**:
```csharp
using TarsEngine.FSharp.Adapters.TreeOfThought;

var parent = ThoughtNodeAdapter.CreateNode("Parent");
var child = ThoughtNodeAdapter.CreateNode("Child");
var updatedParent = parent.AddChild(child);
```

### Example 3: Evaluating a Node

**Old Code**:
```fsharp
open TarsEngine.FSharp.MetascriptToT

let node = createNode "My thought"
let metrics = { Correctness = 0.8; Efficiency = 0.7; Robustness = 0.6; Maintainability = 0.5; Overall = 0.65 }
let evaluatedNode = evaluateNode node metrics
```

**New Code (F#)**:
```fsharp
open TarsEngine.FSharp.Core.TreeOfThought

let node = ThoughtNode.createNode "My thought"
let metrics = ThoughtNode.createMetrics 0.8 0.7 0.6 0.5
let evaluatedNode = ThoughtNode.evaluateNode node metrics
```

**New Code (C#)**:
```csharp
using TarsEngine.FSharp.Adapters.TreeOfThought;

var node = ThoughtNodeAdapter.CreateNode("My thought");
var metrics = EvaluationMetricsAdapter.CreateMetrics(0.8, 0.7, 0.6, 0.5);
var evaluatedNode = node.EvaluateNode(metrics);
```

### Example 4: Using the Tree-of-Thought Service

**Old Code**:
```csharp
using TarsEngine.Services.TreeOfThought;

var service = new SimpleTreeOfThoughtService();
var root = await service.CreateThoughtTreeAsync("Problem", options);
```

**New Code**:
```csharp
using TarsEngine.Services.Abstractions.TreeOfThought;
using TarsEngine.FSharp.Adapters.TreeOfThought;

var service = new FSharpTreeOfThoughtService(logger);
var root = await service.CreateThoughtTreeAsync("Problem", options);
```

## Troubleshooting

### Type Mismatches

If you encounter type mismatches, make sure you're using the correct types from the new implementation. The old and new types have different names and are in different namespaces.

### Function Not Found

If you encounter "function not found" errors, make sure you're using the correct namespace and module. The new functions are in different modules than the old functions.

### Adapter Errors

If you encounter errors when using the adapters, make sure you're using the correct adapter for the type you're working with. The adapters are designed to convert between the old and new types, but they need to be used correctly.

### Compilation Errors

If you encounter compilation errors, make sure you've updated all references to the old implementation. The new implementation has a different structure and different types, so all references need to be updated.

## Conclusion

Migrating from the old F# implementation to the new one requires careful attention to detail, but the benefits are worth it. The new implementation is cleaner, more maintainable, and more robust. By following this guide, you can successfully migrate your code to the new implementation.
