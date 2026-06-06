# Functional Programming Patterns - Extraction Proposal

**Status**: 📋 Proposal  
**Phase**: 7.x (Post-Phase 6 Completion)  
**Priority**: Medium  
**Author**: AI Assistant  
**Date**: 2025-11-29

---

## Executive Summary

This proposal outlines a systematic approach to extracting common functional programming patterns (monads, applicatives, semigroups, etc.) from the TARS v2 codebase into reusable, composable modules. The goal is to reduce code duplication, improve type safety, and provide a consistent functional programming vocabulary across the project.

---

## Current State Analysis

TARS v2 already demonstrates excellent functional design:

### Existing Patterns

1. **Computation Expressions**
   - `AgentWorkflow` CE (bind, return, zero, combine)
   - Location: `src/Tars.Core/AgentWorkflow.fs`
   - Status: ✅ Well-designed, production-ready

2. **Discriminated Unions**
   - `ExecutionOutcome<'T>` (Success/PartialSuccess/Failure)
   - `ParsedResponse` (ToolCall/MultiToolCall/TextResponse)
   - `CognitiveMode` (Exploratory/Convergent/Critical)
   - Status: ✅ Domain-appropriate, type-safe

3. **Result Types**
   - F# built-in `Result<'T,'E>`
   - Custom `PartialFailure` type
   - Status: ✅ Used throughout, but could be more consistent

4. **Option Types**
   - Extensive use of `Option<'T>`
   - Pattern matching on Some/None
   - Status: ✅ Idiomatic F# usage

### Recurring Patterns (Not Yet Extracted)

1. **Validation Logic** - Multiple errors accumulation (applicative style)
2. **Context Threading** - Passing `AgentWorkflowContext` (reader monad style)
3. **Logging/Metrics** - Side effects alongside computation (writer monad style)
4. **State Management** - Agent state transitions (state monad style)
5. **Async Composition** - Nested async blocks (monad transformers)

---

## Proposed Extraction Strategy

### Phase 1: Core Algebraic Structures

**Goal**: Extract fundamental algebraic types into `Tars.Core.Functional`

#### 1.1 Semigroup & Monoid

```fsharp
namespace Tars.Core.Functional

/// Semigroup: combine two values
type ISemigroup<'T> =
    abstract Combine: 'T -> 'T -> 'T

/// Monoid: semigroup with identity element
type IMonoid<'T> =
    inherit ISemigroup<'T>
    abstract Empty: 'T

// Instances
module Semigroups =
    let listSemigroup<'T> : ISemigroup<'T list> =
        { new ISemigroup<'T list> with
            member _.Combine x y = x @ y }
    
    let stringSemigroup : ISemigroup<string> =
        { new ISemigroup<string> with
            member _.Combine x y = x + y }

module Monoids =
    let listMonoid<'T> : IMonoid<'T list> =
        { new IMonoid<'T list> with
            member _.Combine x y = x @ y
            member _.Empty = [] }
```

**Use Cases**:

- Combining partial results in `AgentWorkflow`
- Accumulating warnings in `ExecutionOutcome`
- Merging metric maps

---

#### 1.2 Functor

```fsharp
/// Functor: map over a context
type IFunctor<'F> =
    abstract Map: ('A -> 'B) -> 'F<'A> -> 'F<'B>

// Instances
module Functors =
    let optionFunctor : IFunctor<Option> =
        { new IFunctor<Option> with
            member _.Map f opt = Option.map f opt }
    
    let resultFunctor : IFunctor<Result<*, 'E>> =
        { new IFunctor<Result> with
            member _.Map f res = Result.map f res }
    
    let executionOutcomeFunctor : IFunctor<ExecutionOutcome> =
        { new IFunctor<ExecutionOutcome> with
            member _.Map f outcome =
                match outcome with
                | Success v -> Success (f v)
                | PartialSuccess (v, w) -> PartialSuccess (f v, w)
                | Failure e -> Failure e }
```

**Use Cases**:

- Transform success values without handling errors
- Pipeline transformations
- Generic utility functions

---

#### 1.3 Applicative

```fsharp
/// Applicative: apply functions in a context
type IApplicative<'F> =
    inherit IFunctor<'F>
    abstract Pure: 'A -> 'F<'A>
    abstract Apply: 'F<'A -> 'B> -> 'F<'A> -> 'F<'B>

// Helper for validation
type ValidationResult<'T,'E> =
    | Valid of 'T
    | Invalid of 'E list  // Note: list for accumulation

module Applicatives =
    let validationApplicative<'E> : IApplicative<ValidationResult<*, 'E>> =
        { new IApplicative<ValidationResult> with
            member _.Pure x = Valid x
            
            member _.Map f v =
                match v with
                | Valid x -> Valid (f x)
                | Invalid e -> Invalid e
            
            member _.Apply fv v =
                match fv, v with
                | Valid f, Valid x -> Valid (f x)
                | Invalid e1, Invalid e2 -> Invalid (e1 @ e2)  // Accumulate errors!
                | Invalid e, _ -> Invalid e
                | _, Invalid e -> Invalid e }
```

**Use Cases**:

- **Validate multiple fields** and collect all errors (not fail-fast)
- Parallel computation with error accumulation
- Form validation in UI (future)

---

#### 1.4 Monad

```fsharp
/// Monad: bind (flatMap) operations
type IMonad<'M> =
    inherit IApplicative<'M>
    abstract Bind: 'M<'A> -> ('A -> 'M<'B>) -> 'M<'B>

module Monads =
    let optionMonad : IMonad<Option> =
        { new IMonad<Option> with
            member _.Pure x = Some x
            member _.Map f opt = Option.map f opt
            member _.Apply fopt opt =
                match fopt, opt with
                | Some f, Some x -> Some (f x)
                | _ -> None
            member _.Bind m f = Option.bind f m }
    
    let resultMonad<'E> : IMonad<Result<*, 'E>> =
        { new IMonad<Result> with
            member _.Pure x = Ok x
            member _.Map f res = Result.map f res
            member _.Apply fres res =
                match fres, res with
                | Ok f, Ok x -> Ok (f x)
                | Error e, _ -> Error e
                | _, Error e -> Error e
            member _.Bind m f = Result.bind f m }
```

**Use Cases**:

- Sequential operations that may fail
- Early short-circuiting on error
- Chaining async operations

---

### Phase 2: Monad Transformers

**Goal**: Combine multiple monads (e.g., Async + Result + Reader)

#### 2.1 Reader Monad (Context Threading)

```fsharp
/// Reader monad: thread context through computation
type Reader<'Env,'T> = Reader of ('Env -> 'T)

module Reader =
    let run (Reader f) env = f env
    
    let map f (Reader m) = Reader (fun env -> f (m env))
    
    let bind (Reader m) f = 
        Reader (fun env ->
            let a = m env
            let (Reader m') = f a
            m' env)
    
    /// Ask for the environment
    let ask<'Env> : Reader<'Env,'Env> = Reader id
    
    /// Run with a modified environment
    let local f (Reader m) = Reader (fun env -> m (f env))

// Computation expression
type ReaderBuilder() =
    member _.Return x = Reader (fun _ -> x)
    member _.Bind(m, f) = Reader.bind m f
    member _.ReturnFrom m = m

let reader = ReaderBuilder()
```

**Use Cases**:

- Thread `AgentWorkflowContext` without explicit parameters
- Dependency injection
- Configuration access

**Example**:

```fsharp
let workflowUsingReader = reader {
    let! ctx = Reader.ask  // Get context
    let! agent = Reader (fun ctx -> ctx.Self)
    // ... use ctx and agent without passing them around
    return result
}
```

---

#### 2.2 Writer Monad (Logging/Metrics)

```fsharp
/// Writer monad: accumulate values alongside computation
type Writer<'W,'T> = Writer of ('T * 'W)

module Writer =
    let run (Writer (v, w)) = (v, w)
    
    let map f (Writer (v, w)) = Writer (f v, w)
    
    let bind (Writer (v, w)) f =
        let (Writer (v', w')) = f v
        Writer (v', w @ w')  // Assuming 'W is a list (monoid)
    
    /// Write a log entry
    let tell w = Writer ((), [w])
    
    /// Listen to accumulated logs
    let listen (Writer (v, w)) = Writer ((v, w), w)

type WriterBuilder<'W when 'W :> ISemigroup<'W>>() =
    member _.Return x = Writer (x, Monoid.empty)
    member _.Bind(m, f) = Writer.bind m f
    member _.ReturnFrom m = m

let writer = WriterBuilder()
```

**Use Cases**:

- **Collect metrics alongside computation** (already doing this manually)
- Structured logging
- Audit trails

**Example**:

```fsharp
let workflowWithMetrics = writer {
    do! Writer.tell (Metric("start", DateTime.UtcNow))
    let! result = performComputation()
    do! Writer.tell (Metric("end", DateTime.UtcNow))
    return result
}
```

---

#### 2.3 State Monad (Agent State Management)

```fsharp
/// State monad: thread state through computation
type State<'S,'T> = State of ('S -> 'T * 'S)

module State =
    let run (State f) initialState = f initialState
    
    let map f (State m) =
        State (fun s ->
            let (a, s') = m s
            (f a, s'))
    
    let bind (State m) f =
        State (fun s ->
            let (a, s') = m s
            let (State m') = f a
            m' s')
    
    /// Get current state
    let get<'S> : State<'S,'S> = State (fun s -> (s, s))
    
    /// Set new state
    let put s : State<'S,unit> = State (fun _ -> ((), s))
    
    /// Modify state
    let modify f : State<'S,unit> =
        State (fun s -> ((), f s))

type StateBuilder() =
    member _.Return x = State (fun s -> (x, s))
    member _.Bind(m, f) = State.bind m f
    member _.ReturnFrom m = m

let state = StateBuilder()
```

**Use Cases**:

- **Agent state transitions** (currently using mutable record updates)
- Game state management
- Compiler phases

**Example**:

```fsharp
let agentStateTransition = state {
    let! currentState = State.get
    do! State.put (Thinking currentState.Memory)
    return! nextStep
}
```

---

#### 2.4 Async Monad Transformers

```fsharp
/// Combine Async + Result
type AsyncResult<'T,'E> = Async<Result<'T,'E>>

module AsyncResult =
    let map f (ar: AsyncResult<'T,'E>) : AsyncResult<'U,'E> =
        async {
            let! result = ar
            return Result.map f result
        }
    
    let bind (ar: AsyncResult<'T,'E>) (f: 'T -> AsyncResult<'U,'E>) : AsyncResult<'U,'E> =
        async {
            let! result = ar
            match result with
            | Ok v -> return! f v
            | Error e -> return Error e
        }
    
    let retn x : AsyncResult<'T,'E> = async { return Ok x }

type AsyncResultBuilder() =
    member _.Return x = AsyncResult.retn x
    member _.Bind(m, f) = AsyncResult.bind m f
    member _.ReturnFrom m = m

let asyncResult = AsyncResultBuilder()
```

**Use Cases**:

- **Combine async I/O with error handling** (currently using nested matches)
- LLM calls with Result wrapping
- Database operations

---

### Phase 3: Domain-Specific Combinators

#### 3.1 ExecutionOutcome Combinators

```fsharp
module ExecutionOutcome =
    /// Map over successful value
    let map f outcome =
        match outcome with
        | Success v -> Success (f v)
        | PartialSuccess (v, w) -> PartialSuccess (f v, w)
        | Failure e -> Failure e
    
    /// Bind (flatMap)
    let bind outcome f =
        match outcome with
        | Success v -> f v
        | PartialSuccess (v, w) ->
            match f v with
            | Success v' -> PartialSuccess (v', w)
            | PartialSuccess (v', w') -> PartialSuccess (v', w @ w')
            | Failure e -> Failure (w @ e)
        | Failure e -> Failure e
    
    /// Apply (for parallel composition)
    let apply foutcome voutcome =
        match foutcome, voutcome with
        | Success f, Success v -> Success (f v)
        | Success f, PartialSuccess (v, w) -> PartialSuccess (f v, w)
        | PartialSuccess (f, w1), Success v -> PartialSuccess (f v, w1)
        | PartialSuccess (f, w1), PartialSuccess (v, w2) -> PartialSuccess (f v, w1 @ w2)
        | Failure e1, Failure e2 -> Failure (e1 @ e2)
        | Failure e, _ -> Failure e
        | _, Failure e -> Failure e
    
    /// Sequence a list of outcomes
    let sequence outcomes =
        List.fold (fun acc outcome ->
            apply (map (fun list v -> v :: list) acc) outcome
        ) (Success []) outcomes
        |> map List.rev
    
    /// Traverse (map + sequence)
    let traverse f list =
        list |> List.map f |> sequence
```

**Use Cases**:

- Compose multiple agent actions
- Parallel tool execution
- Batch operations with partial failures

---

#### 3.2 AgentWorkflow Utilities

```fsharp
module AgentWorkflow =
    /// Lift a pure value into workflow
    let retn x : AgentWorkflow<'T> =
        fun _ -> async { return Success x }
    
    /// Sequence workflows
    let sequence workflows =
        fun ctx ->
            async {
                let! results = 
                    workflows
                    |> List.map (fun wf -> wf ctx)
                    |> Async.Sequential
                return ExecutionOutcome.sequence (Array.toList results)
            }
    
    /// Retry with exponential backoff
    let retry maxAttempts (workflow: AgentWorkflow<'T>) : AgentWorkflow<'T> =
        fun ctx ->
            let rec loop attempt backoff =
                async {
                    let! result = workflow ctx
                    match result with
                    | Success _ -> return result
                    | PartialSuccess _ when attempt >= maxAttempts -> return result
                    | Failure _ when attempt >= maxAttempts -> return result
                    | _ ->
                        do! Async.Sleep backoff
                        return! loop (attempt + 1) (backoff * 2)
                }
            loop 1 1000
```

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

- [ ] Create `src/Tars.Core/Functional/` directory
- [ ] Implement Semigroup, Monoid interfaces
- [ ] Implement Functor, Applicative, Monad interfaces
- [ ] Add unit tests for algebraic laws
- [ ] Document with examples

### Phase 2: Monad Transformers (Week 3-4)

- [ ] Implement Reader monad + builder
- [ ] Implement Writer monad + builder
- [ ] Implement State monad + builder
- [ ] Implement AsyncResult transformer
- [ ] Add integration tests

### Phase 3: Domain Utilities (Week 5-6)

- [ ] Extract ExecutionOutcome combinators
- [ ] Extract AgentWorkflow utilities
- [ ] Refactor existing code to use new combinators
- [ ] Performance benchmarks

### Phase 4: Documentation & Training (Week 7-8)

- [ ] Write comprehensive guide: "Functional Patterns in TARS"
- [ ] Add cookbook examples
- [ ] Create migration guide for existing code
- [ ] Internal workshop/presentation

---

## Benefits

1. **Code Reusability**: Common patterns extracted into reusable modules
2. **Type Safety**: Algebraic laws enforced at compile time
3. **Composability**: Small functions combined into larger workflows
4. **Testability**: Algebraic laws provide automatic test cases
5. **Readability**: Consistent functional vocabulary across codebase
6. **Performance**: Optimizations in one place benefit all usage
7. **Onboarding**: New developers learn F# idioms through well-documented patterns

---

## Trade-offs & Considerations

### Advantages

✅ Reduced code duplication  
✅ More composable abstractions  
✅ Type-safe transformations  
✅ Easier to reason about control flow  

### Disadvantages

⚠️ Learning curve for developers unfamiliar with category theory  
⚠️ Potential over-abstraction if not disciplined  
⚠️ Performance overhead from excessive wrapping (mitigated with inlining)  
⚠️ Increased cognitive load for simple operations  

### Mitigation Strategies

1. **Start simple**: Only extract patterns with 3+ use sites
2. **Document heavily**: Include real-world examples
3. **Training**: Provide workshops and pair programming
4. **Pragmatism**: Don't force functional purity everywhere
5. **Performance**: Benchmark and optimize hot paths

---

## Success Criteria

1. **Adoption Rate**: 80% of new code uses extracted patterns within 3 months
2. **Code Reduction**: 20% reduction in duplicated logic
3. **Bug Reduction**: 15% fewer runtime errors in refactored modules
4. **Developer Satisfaction**: Positive feedback from team survey
5. **Performance**: No regressions in critical paths

---

## Examples of Refactoring Opportunities

### Before (Current)

```fsharp
let workflow = agent {
    let! context = getContext()
    match context.Budget with
    | Some governor ->
        match governor.TryConsume cost with
        | Ok _ ->
            let! result = performAction()
            match result with
            | Some v -> return Success v
            | None -> return Failure [Error "Action failed"]
        | Error e -> return Failure [Error $"Budget: {e}"]
    | None ->
        let! result = performAction()
        match result with
        | Some v -> return Success v
        | None -> return Failure [Error "Action failed"]
}
```

### After (With Patterns)

```fsharp
let workflow = reader {
    let! governor = Reader (fun ctx -> ctx.Budget)
    let! result = Reader.lift (
        asyncResult {
            do! checkBudget governor cost
            return! performAction()
        })
    return result
}
```

---

## Conclusion

Extracting functional programming patterns from TARS v2 will:

1. **Reduce complexity** through composable abstractions
2. **Improve correctness** via type-safe transformations
3. **Enhance maintainability** with consistent idioms
4. **Accelerate development** by reusing proven patterns

This proposal should be considered **post-Phase 6**, once the current architecture stabilizes.

---

**Recommendation**: **APPROVE** for Phase 7.x implementation, pending stakeholder review and team training plan.
