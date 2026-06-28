# Codebase Design Vocabulary for TARS

This document provides a shared vocabulary for deep modules, small interfaces, seams, adapters, locality, leverage, and deletion tests, adapted for the TARS F# functional-first context. Agents should use these terms when applying the `anti-ball-of-mud` skill to articulate architectural friction and propose solutions.

## Module (Deep vs. Shallow)
- **Deep Module:** A module that hides significant complexity behind a small, simple interface. In F#, this often looks like a module exposing a few well-typed `Result<T, E>` returning functions while keeping complex algorithms, state management, or IO encapsulated.
- **Shallow Module:** A module whose interface is nearly as complex as its implementation. Shallow modules often just rename complexity without reducing the cognitive load for the caller.

## Interface
The boundary between components. In F#, interfaces are typically defined by function signatures, record types representing inputs/outputs, and Discriminated Unions representing possible states or errors.
- **Small Interface:** An interface that minimizes the surface area of interaction. It should be easy to understand and hard to use incorrectly (make illegal states unrepresentable).

## Seam
A place where you can alter behavior in your program without editing in that place.
- **Narrow Seam:** A carefully designed boundary that isolates different concerns (e.g., separating core logic from side effects like API calls or file system access). In functional programming, seams are often created by passing functions as arguments (higher-order functions) or returning data that describes side effects rather than executing them directly.

## Adapter
A layer that translates one interface to another.
- **Purpose:** Adapters are crucial for isolating the core domain logic from external dependencies (e.g., GitHub API, LLM providers, file system). The core domain should depend on abstractions (interfaces or function signatures) that the adapter implements.

## Locality
The principle that code that changes together should live together, and code that needs to be understood together should be located close to each other.
- **High Locality:** Minimizes bouncing across multiple files to understand a single concept.

## Leverage
The impact of a change relative to its size.
- **High Leverage:** A small change in a core contract or a well-placed adapter that significantly simplifies multiple other areas of the codebase.

## Deletion Test
A thought experiment: "How hard would it be to delete this feature or dependency?"
- **Good Design:** Features or dependencies can be removed by deleting a localized set of files and removing a few registrations, without cascading changes throughout the entire codebase. This implies well-defined seams and loose coupling.
