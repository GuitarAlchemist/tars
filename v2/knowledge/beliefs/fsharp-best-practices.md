---
id: fsharp-best
title: F# Best Practices
category: beliefs
confidence: high
source: observed
tags: fsharp, coding, best-practices
created: 2024-01-01T00:00:00Z
updated: 2024-01-01T00:00:00Z
---

# F# Best Practices (TARS Beliefs)

## Type Design
- Prefer discriminated unions over inheritance
- Use single-case DUs for type safety (e.g., `AgentId of Guid`)
- Make illegal states unrepresentable

## Function Design
- Keep functions small and composable
- Use `|>` for pipelines
- Prefer `Option` over null
- Use `Result` for operations that can fail

## Module Organization
- Group related functions in modules
- Keep modules focused on one responsibility
- Use `[<AutoOpen>]` sparingly

## Async/Task
- Use `task { }` for .NET interop
- Use `async { }` for F#-native async
- Avoid blocking with `.Result` or `.Wait()`

## Testing
- Pure functions are easy to test
- Use property-based testing for invariants
- Test edge cases explicitly

