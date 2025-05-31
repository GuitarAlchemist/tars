# Functional Programming Guide for TARS

This guide provides an overview of functional programming patterns and how to use them effectively in the TARS codebase.

## Table of Contents

1. [Introduction](#introduction)
2. [Monads](#monads)
   - [Option Monad](#option-monad)
   - [Result Monad](#result-monad)
   - [AsyncResult Monad](#asyncresult-monad)
3. [Discriminated Unions](#discriminated-unions)
   - [Either](#either)
   - [Validation](#validation)
4. [Best Practices](#best-practices)
5. [Examples](#examples)

## Introduction

Functional programming is a programming paradigm that treats computation as the evaluation of mathematical functions and avoids changing state and mutable data. In C#, we can adopt functional programming patterns to write more robust, maintainable, and testable code.

Key benefits of functional programming in TARS:

- **Type Safety**: Catch errors at compile time rather than runtime
- **Immutability**: Prevent unexpected state changes
- **Composability**: Build complex operations from simple ones
- **Testability**: Pure functions are easier to test
- **Readability**: Express intent clearly

## Monads

Monads are a design pattern that allows us to chain operations while handling side effects like nullable values, errors, or asynchronous operations.

### Option Monad

The `EnhancedOption<T>` monad represents a value that may or may not be present. Use it instead of null references.

```csharp
// Instead of:
string name = GetName();
if (name != null)
{
    Console.WriteLine($"Hello, {name}!");
}
else
{
    Console.WriteLine("Hello, anonymous user!");
}

// Use:
EnhancedOption<string> nameOption = GetNameOption();
string greeting = nameOption.Match(
    some: name => $"Hello, {name}!",
    none: () => "Hello, anonymous user!"
);
Console.WriteLine(greeting);
```

Key operations:

- `EnhancedOption.Some(value)`: Create an option with a value
- `EnhancedOption.None<T>()`: Create an option with no value
- `option.Match(some, none)`: Pattern match on the option
- `option.Map(mapper)`: Transform the value if present
- `option.Bind(binder)`: Chain operations that return options
- `option.ValueOr(defaultValue)`: Get the value or a default
- `option.IfSome(action)`: Perform an action if the value is present
- `option.IfNone(action)`: Perform an action if the value is not present

Extension methods for collections:

- `source.TryFind(predicate)`: Find an element or return None
- `source.TryFirst()`: Get the first element or None
- `source.Choose()`: Filter out None values from a sequence of options

### Result Monad

The `Result<T, TError>` monad represents an operation that might fail. Use it instead of throwing exceptions.

```csharp
// Instead of:
try
{
    int result = Divide(10, 0);
    Console.WriteLine($"Result: {result}");
}
catch (Exception ex)
{
    Console.WriteLine($"Error: {ex.Message}");
}

// Use:
Result<int, Exception> result = DivideResult(10, 0);
string message = result.Match(
    success: value => $"Result: {value}",
    failure: ex => $"Error: {ex.Message}"
);
Console.WriteLine(message);
```

Key operations:

- `Result<T, TError>.Success(value)`: Create a successful result
- `Result<T, TError>.Failure(error)`: Create a failed result
- `result.Match(success, failure)`: Pattern match on the result
- `result.Map(mapper)`: Transform the value if successful
- `result.MapError(mapper)`: Transform the error if failed
- `result.Bind(binder)`: Chain operations that return results
- `result.IfSuccess(action)`: Perform an action if successful
- `result.IfFailure(action)`: Perform an action if failed

### AsyncResult Monad

The `EnhancedAsyncResult<T>` monad represents an asynchronous operation. Use it to handle async operations in a functional way.

```csharp
// Instead of:
public async Task<bool> DeactivateAsync()
{
    if (!_isActive)
    {
        _logger.LogInformation("Connection discovery is already inactive");
        return true;
    }
    
    try
    {
        _logger.LogInformation("Deactivating connection discovery");
        _isActive = false;
        return true;
    }
    catch (Exception ex)
    {
        _logger.LogError(ex, "Error deactivating connection discovery");
        return false;
    }
}

// Use:
public EnhancedAsyncResult<bool> DeactivateAsync()
{
    if (!_isActive)
    {
        _logger.LogInformation("Connection discovery is already inactive");
        return EnhancedAsyncResult.FromResult(true);
    }
    
    return EnhancedAsyncResult.TryAsync(async () =>
    {
        _logger.LogInformation("Deactivating connection discovery");
        _isActive = false;
        return true;
    });
}
```

Key operations:

- `EnhancedAsyncResult.FromResult(value)`: Create an async result from a value
- `EnhancedAsyncResult.FromTask(task)`: Create an async result from a task
- `asyncResult.RunAsync()`: Run the async operation and get the result
- `asyncResult.Map(mapper)`: Transform the result when it completes
- `asyncResult.Bind(binder)`: Chain operations that return async results
- `asyncResult.Do(action)`: Perform an action when the operation completes

## Discriminated Unions

Discriminated unions are a type that can be one of several cases, each with its own data. In C#, we can implement them using abstract record types and pattern matching.

### Either

The `Either<TLeft, TRight>` type represents a value of one of two possible types. By convention, Left is used for failure and Right is used for success.

```csharp
Either<string, int> ParseInt(string input)
{
    if (int.TryParse(input, out var result))
    {
        return Either<string, int>.Right(result);
    }
    else
    {
        return Either<string, int>.Left($"Cannot parse '{input}' as an integer");
    }
}

var result = ParseInt("42");
string message = result.Match(
    leftFunc: error => $"Error: {error}",
    rightFunc: value => $"Value: {value}"
);
Console.WriteLine(message); // Output: Value: 42
```

Key operations:

- `Either<TLeft, TRight>.Left(value)`: Create a Left instance
- `Either<TLeft, TRight>.Right(value)`: Create a Right instance
- `either.Match(leftFunc, rightFunc)`: Pattern match on the either
- `either.Map(mapper)`: Transform the Right value
- `either.MapLeft(mapper)`: Transform the Left value
- `either.Bind(binder)`: Chain operations that return eithers
- `EitherExtensions.Try(func)`: Try to execute a function and return an Either

### Validation

The `Validation<T, TError>` type represents a validation result that can be either valid with a value or invalid with a list of errors.

```csharp
Validation<string, string> ValidateUsername(string username) =>
    string.IsNullOrWhiteSpace(username)
        ? Validation<string, string>.Invalid("Username cannot be empty")
        : username.Length < 3
            ? Validation<string, string>.Invalid("Username must be at least 3 characters")
            : username.Length > 20
                ? Validation<string, string>.Invalid("Username cannot be longer than 20 characters")
                : Validation<string, string>.Valid(username);

var result = ValidateUsername("john_doe");
result.Match(
    validFunc: username => Console.WriteLine($"Valid username: {username}"),
    invalidFunc: errors => Console.WriteLine($"Validation errors: {string.Join(", ", errors)}")
);
// Output: Valid username: john_doe
```

Key operations:

- `Validation<T, TError>.Valid(value)`: Create a valid validation result
- `Validation<T, TError>.Invalid(error)`: Create an invalid validation result with a single error
- `Validation<T, TError>.Invalid(errors)`: Create an invalid validation result with multiple errors
- `validation.Match(validFunc, invalidFunc)`: Pattern match on the validation result
- `validation.Map(mapper)`: Transform the value if valid
- `validation.MapError(mapper)`: Transform the errors if invalid
- `validation.Bind(binder)`: Chain operations that return validations
- `ValidationExtensions.Sequence(validations)`: Combine multiple validation results
- `ValidationExtensions.Map2(validation1, validation2, mapper)`: Combine two validation results
- `ValidationExtensions.Map3(validation1, validation2, validation3, mapper)`: Combine three validation results

## Best Practices

1. **Prefer monads over null checks and exceptions**:
   - Use `EnhancedOption<T>` instead of nullable references
   - Use `Result<T, TError>` instead of throwing exceptions
   - Use `EnhancedAsyncResult<T>` for asynchronous operations

2. **Use pattern matching for control flow**:
   - Use `Match` methods to handle different cases
   - Use C# 9+ pattern matching syntax for complex conditions

3. **Favor immutability**:
   - Use records for data types
   - Avoid mutating state
   - Return new instances instead of modifying existing ones

4. **Compose operations**:
   - Use `Map`, `Bind`, and other combinators to chain operations
   - Break complex operations into smaller, reusable functions

5. **Handle errors explicitly**:
   - Use `Result<T, TError>` to make error handling explicit
   - Collect all errors with `Validation<T, TError>`
   - Propagate errors with `Either<TLeft, TRight>`

6. **Use modern C# features**:
   - Records for immutable data types
   - Pattern matching for control flow
   - Target-typed new expressions
   - Init-only properties

## Examples

For complete examples, see the `TarsEngine.Examples.FunctionalProgrammingExamples` class.

### Combining Multiple Patterns

```csharp
// Define a function that might return null
EnhancedOption<string> GetUsername(int userId) =>
    userId switch
    {
        1 => EnhancedOption.Some("john_doe"),
        2 => EnhancedOption.Some("jane_smith"),
        _ => EnhancedOption.None<string>()
    };

// Define a function that might fail
async Task<Result<int, Exception>> GetUserAgeAsync(string username)
{
    await Task.Delay(100); // Simulate API call
    return username switch
    {
        "john_doe" => Result<int, Exception>.Success(25),
        "jane_smith" => Result<int, Exception>.Success(30),
        _ => Result<int, Exception>.Failure(new ArgumentException($"User not found: {username}"))
    };
}

// Define a validation function
Validation<int, string> ValidateAge(int age) =>
    age < 18
        ? Validation<int, string>.Invalid("You must be at least 18 years old")
        : age > 120
            ? Validation<int, string>.Invalid("Invalid age")
            : Validation<int, string>.Valid(age);

// Combine all patterns
int userId = 1;

// 1. Get the username (Option)
var usernameOption = GetUsername(userId);

// 2. Convert to Either for error handling
var usernameEither = usernameOption.ToEither($"User not found with ID: {userId}");

// 3. Use AsyncResultError for async operation
var ageResult = await usernameEither.Match(
    leftFunc: error => Task.FromResult(Result<int, Exception>.Failure(new Exception(error))),
    rightFunc: username => GetUserAgeAsync(username)
);

// 4. Convert to Validation for validation
var ageValidation = ageResult.Match(
    success: age => ValidateAge(age),
    failure: ex => Validation<int, string>.Invalid(ex.Message)
);

// 5. Handle the final result
ageValidation.Match(
    validFunc: age => Console.WriteLine($"User is {age} years old and eligible"),
    invalidFunc: errors => Console.WriteLine($"Validation failed: {string.Join(", ", errors)}")
);
// Output: User is 25 years old and eligible
```

This example demonstrates how to combine multiple functional programming patterns to handle a complex workflow with nullable values, asynchronous operations, error handling, and validation.
