# F# Migration Best Practices

This document provides best practices for using the new F# implementation in `TarsEngine.FSharp.Core` and its C# adapters in `TarsEngine.FSharp.Adapters`.

## Table of Contents

1. [General Best Practices](#general-best-practices)
2. [F# Best Practices](#f-best-practices)
3. [C# Best Practices](#c-best-practices)
4. [Interoperability Best Practices](#interoperability-best-practices)
5. [Testing Best Practices](#testing-best-practices)

## General Best Practices

### Use the Right Tool for the Job

- Use F# for functional code, especially code that involves pattern matching, immutability, and type inference.
- Use C# for object-oriented code, especially code that involves inheritance, interfaces, and LINQ.
- Use the adapter layer to bridge the gap between F# and C# code.

### Follow the Functional Programming Paradigm

- Prefer immutable data structures over mutable ones.
- Prefer pure functions over functions with side effects.
- Use pattern matching to handle different cases.
- Use composition to build complex functions from simple ones.

### Document Your Code

- Document all public functions and types.
- Explain the purpose of each function and type.
- Provide examples of how to use the code.
- Document any assumptions or constraints.

### Write Tests

- Write tests for all public functions and types.
- Test edge cases and error conditions.
- Use property-based testing for functions with well-defined properties.
- Test interoperability between F# and C# code.

## F# Best Practices

### Use Modules to Organize Code

- Use modules to group related functions and types.
- Use nested modules to create a hierarchy of functions and types.
- Use the `[<AutoOpen>]` attribute sparingly, only for truly ubiquitous functions.

```fsharp
// Good
module TarsEngine.FSharp.Core.TreeOfThought

type ThoughtNode = { ... }

let createNode thought = ...
let addChild parent child = ...
let evaluateNode node metrics = ...

// Bad
module TarsEngine.FSharp.Core

type ThoughtNode = { ... }

let createNode thought = ...
let addChild parent child = ...
let evaluateNode node metrics = ...
```

### Use Type Inference Wisely

- Let the compiler infer types when it's clear what the types should be.
- Provide explicit type annotations for public functions and types.
- Use type annotations to document the expected types.

```fsharp
// Good
let createNode (thought: string) : ThoughtNode = ...

// Also good for internal functions
let private calculateScore metrics = ...
```

### Use Pattern Matching

- Use pattern matching to handle different cases.
- Use pattern matching to extract values from complex types.
- Use pattern matching to handle optional values.

```fsharp
// Good
match node.Evaluation with
| Some eval -> eval.Overall
| None -> 0.0

// Bad
if node.Evaluation.IsSome then
    node.Evaluation.Value.Overall
else
    0.0
```

### Use Railway-Oriented Programming for Error Handling

- Use the `Result` type to represent success or failure.
- Use the `Option` type to represent the presence or absence of a value.
- Use the `AsyncResult` type for asynchronous operations that can fail.

```fsharp
// Good
let tryFindNode thought node =
    match findNode thought node with
    | Some n -> Ok n
    | None -> Error $"Node with thought '{thought}' not found"

// Bad
let tryFindNode thought node =
    let n = findNode thought node
    if n.IsSome then
        n.Value
    else
        failwith $"Node with thought '{thought}' not found"
```

## C# Best Practices

### Use the Adapter Layer

- Use the adapter layer to interact with F# code.
- Don't try to use F# types directly from C# code.
- Use the adapter methods to convert between F# and C# types.

```csharp
// Good
var node = ThoughtNodeAdapter.CreateNode("My thought");
var child = ThoughtNodeAdapter.CreateNode("Child thought");
var updatedNode = node.AddChild(child);

// Bad
var node = new FSharp.Core.TreeOfThought.ThoughtNode.ThoughtNode("My thought", new List<FSharp.Core.TreeOfThought.ThoughtNode.ThoughtNode>(), FSharpOption<FSharp.Core.TreeOfThought.ThoughtNode.EvaluationMetrics>.None, false, new Map<string, object>());
```

### Use the Service Interface

- Use the `ITreeOfThoughtService` interface to interact with the Tree-of-Thought functionality.
- Don't try to use the adapter layer directly in application code.
- Use dependency injection to get an instance of the service.

```csharp
// Good
public class MyClass
{
    private readonly ITreeOfThoughtService _totService;

    public MyClass(ITreeOfThoughtService totService)
    {
        _totService = totService;
    }

    public async Task DoSomethingAsync()
    {
        var root = await _totService.CreateThoughtTreeAsync("Problem", options);
        // ...
    }
}

// Bad
public class MyClass
{
    public async Task DoSomethingAsync()
    {
        var node = ThoughtNodeAdapter.CreateNode("Problem");
        // ...
    }
}
```

### Use Immutable Objects

- Treat the objects returned by the adapter layer as immutable.
- Don't try to modify the properties of these objects directly.
- Use the adapter methods to create new objects with the desired changes.

```csharp
// Good
var updatedNode = node.AddChild(child);

// Bad
node.Children.Add(child); // This won't work because Children is a read-only list
```

### Handle F# Option Types Correctly

- Use the adapter methods to handle F# option types.
- Don't try to access the value of an option directly.
- Use null checks for nullable reference types.

```csharp
// Good
var evaluation = node.Evaluation;
if (evaluation != null)
{
    var score = evaluation.Overall;
    // ...
}

// Bad
var score = ((FSharpOption<FSharp.Core.TreeOfThought.ThoughtNode.EvaluationMetrics>)node.FSharpNode.Evaluation).Value.Overall;
```

## Interoperability Best Practices

### Use the Adapter Pattern

- Use the adapter pattern to bridge the gap between F# and C# code.
- Create adapter classes that wrap F# types and provide a C#-friendly interface.
- Use extension methods to add C#-friendly methods to F# types.

```csharp
// Good
public class ThoughtNodeAdapter
{
    private readonly FSharp.Core.TreeOfThought.ThoughtNode.ThoughtNode _fsharpNode;

    public ThoughtNodeAdapter(FSharp.Core.TreeOfThought.ThoughtNode.ThoughtNode fsharpNode)
    {
        _fsharpNode = fsharpNode;
    }

    public string Thought => _fsharpNode.Thought;
    // ...
}

// Bad
public static class ThoughtNodeExtensions
{
    public static string GetThought(this FSharp.Core.TreeOfThought.ThoughtNode.ThoughtNode node)
    {
        return node.Thought;
    }
    // ...
}
```

### Handle F# Types Correctly

- Use the appropriate C# types to represent F# types.
- Use `IReadOnlyList<T>` to represent F# lists.
- Use `IReadOnlyDictionary<K, V>` to represent F# maps.
- Use nullable reference types to represent F# options.

```csharp
// Good
public IReadOnlyList<ThoughtNodeAdapter> Children { get { ... } }
public EvaluationMetricsAdapter? Evaluation { get { ... } }
public IReadOnlyDictionary<string, object> Metadata { get { ... } }

// Bad
public List<ThoughtNodeAdapter> Children { get { ... } }
public EvaluationMetricsAdapter Evaluation { get { ... } }
public Dictionary<string, object> Metadata { get { ... } }
```

### Use F# Functions from C#

- Use the `FSharpFunc<T, TResult>` class to represent F# functions in C#.
- Use the `Invoke` method to call F# functions from C#.
- Use lambda expressions to create F# functions in C#.

```csharp
// Good
var fsharpFunc = new FSharpFunc<FSharp.Core.TreeOfThought.ThoughtNode.ThoughtNode, bool>(
    fsharpNode => predicate(new ThoughtNodeAdapter(fsharpNode)));

// Bad
Func<FSharp.Core.TreeOfThought.ThoughtNode.ThoughtNode, bool> fsharpFunc = 
    fsharpNode => predicate(new ThoughtNodeAdapter(fsharpNode));
```

## Testing Best Practices

### Test F# Code with F# Tests

- Use F# to test F# code.
- Use property-based testing for functions with well-defined properties.
- Use unit tests for functions with specific behavior.
- Use integration tests for functions that interact with external systems.

```fsharp
// Good
[<Fact>]
let ``createNode should create a node with the given thought`` () =
    let thought = "Test thought"
    let node = ThoughtNode.createNode thought
    Assert.Equal(thought, node.Thought)
    Assert.Empty(node.Children)
    Assert.True(Option.isNone node.Evaluation)
    Assert.False(node.Pruned)
    Assert.Empty(node.Metadata)
```

### Test C# Code with C# Tests

- Use C# to test C# code.
- Use unit tests for classes with specific behavior.
- Use integration tests for classes that interact with external systems.
- Use mocks to isolate the code being tested.

```csharp
// Good
[Fact]
public void CreateNode_ShouldCreateNodeWithGivenThought()
{
    // Arrange
    var thought = "Test thought";

    // Act
    var node = ThoughtNodeAdapter.CreateNode(thought);

    // Assert
    Assert.Equal(thought, node.Thought);
    Assert.Empty(node.Children);
    Assert.Null(node.Evaluation);
    Assert.False(node.Pruned);
    Assert.Empty(node.Metadata);
}
```

### Test Interoperability

- Test that F# code can be called from C# code.
- Test that C# code can be called from F# code.
- Test that data can be passed between F# and C# code.
- Test that errors are handled correctly across the language boundary.

```csharp
// Good
[Fact]
public void FSharpFunction_ShouldBeCallableFromCSharp()
{
    // Arrange
    var thought = "Test thought";

    // Act
    var node = ThoughtNodeAdapter.CreateNode(thought);

    // Assert
    Assert.Equal(thought, node.Thought);
}
```

```fsharp
// Good
[<Fact>]
let ``C# function should be callable from F#`` () =
    // Arrange
    let thought = "Test thought"
    
    // Act
    let node = ThoughtNodeAdapter.CreateNode(thought)
    
    // Assert
    Assert.Equal(thought, node.Thought)
```

## Conclusion

By following these best practices, you can ensure that your code is maintainable, robust, and efficient. The new F# implementation provides a solid foundation for building complex applications, and the adapter layer makes it easy to use from C# code.
