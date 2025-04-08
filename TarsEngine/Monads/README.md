# TARS Monads Library

This library provides a set of monadic types to help write more functional and robust code in C#. It addresses common issues like handling nullable references, asynchronous operations, and error handling in a consistent and composable way.

## Available Monads

### Option<T>

Represents an optional value that may or may not be present. Used to handle nullable references in a functional way.

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
Option<string> nameOption = GetNameOption();
string greeting = nameOption.Match(
    some: name => $"Hello, {name}!",
    none: () => "Hello, anonymous user!"
);
Console.WriteLine(greeting);
```

### Result<T, TError>

Represents the result of an operation that might fail. Used to handle errors in a functional way.

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

### AsyncResult<T>

Represents an asynchronous operation that will produce a result. Used to handle async operations in a functional way.

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
public Task<bool> DeactivateAsync()
{
    if (!_isActive)
    {
        _logger.LogInformation("Connection discovery is already inactive");
        return Task.FromResult(true);
    }
    
    try
    {
        _logger.LogInformation("Deactivating connection discovery");
        _isActive = false;
        return Task.FromResult(true);
    }
    catch (Exception ex)
    {
        _logger.LogError(ex, "Error deactivating connection discovery");
        return Task.FromResult(false);
    }
}

// Or with AsyncResult monad:
public AsyncResult<bool> DeactivateAsyncMonad()
{
    if (!_isActive)
    {
        _logger.LogInformation("Connection discovery is already inactive");
        return AsyncResult<bool>.FromResult(true);
    }
    
    try
    {
        _logger.LogInformation("Deactivating connection discovery");
        _isActive = false;
        return AsyncResult<bool>.FromResult(true);
    }
    catch (Exception ex)
    {
        _logger.LogError(ex, "Error deactivating connection discovery");
        return AsyncResult<bool>.FromResult(false);
    }
}
```

### AsyncOption<T>

Represents an asynchronous operation that will produce an optional value. Used to handle async operations that might return null in a functional way.

```csharp
// Use:
public AsyncOption<User> GetUserAsync(string userId)
{
    return AsyncOption<User>.FromValueTask(
        _userRepository.GetUserByIdAsync(userId)
    );
}

// Usage:
AsyncOption<User> userOption = GetUserAsync("123");
await userOption.RunAsync().ContinueWith(t => {
    t.Result.Match(
        some: user => Console.WriteLine($"Found user: {user.Name}"),
        none: () => Console.WriteLine("User not found")
    );
});
```

### AsyncResultError<T, TError>

Represents an asynchronous operation that might fail. Used to handle async operations that might fail in a functional way.

```csharp
// Use:
public AsyncResultError<User, Exception> GetUserAsync(string userId)
{
    return AsyncResultError.TryAsync(async () => {
        var user = await _userRepository.GetUserByIdAsync(userId);
        if (user == null)
            throw new NotFoundException($"User {userId} not found");
        return user;
    });
}

// Usage:
AsyncResultError<User, Exception> userResult = GetUserAsync("123");
await userResult.RunAsync().ContinueWith(t => {
    t.Result.Match(
        success: user => Console.WriteLine($"Found user: {user.Name}"),
        failure: ex => Console.WriteLine($"Error: {ex.Message}")
    );
});
```

### Printable<T>

Represents a value that can be converted to a string representation. Used for consistent string formatting across different types.

```csharp
// Instead of:
DateTime now = DateTime.Now;
Console.WriteLine($"Current time: {now:yyyy-MM-dd HH:mm:ss}");

// Use:
DateTime now = DateTime.Now;
Printable<DateTime> printableNow = Printable<DateTime>.Create(now, dt => dt.ToString("yyyy-MM-dd HH:mm:ss"));
Console.WriteLine($"Current time: {printableNow}");

// Or with extension method:
Console.WriteLine($"Current time: {now.ToPrintable(dt => dt.ToString("yyyy-MM-dd HH:mm:ss"))}");

// Or with predefined printer:
Console.WriteLine($"Current time: {now.ToPrintable(Printers.DateTime)}");
```

## Adapters

The library also includes adapters to help with ambiguous references between different types:

### KnowledgeItemAdapter

Converts between `TarsEngine.Models.KnowledgeItem` and `TarsEngine.Services.Interfaces.KnowledgeItem`.

```csharp
// Instead of:
var modelItem = new ModelKnowledgeItem
{
    Id = item.Id,
    Title = item.Title,
    Content = item.Content,
    Type = (ModelKnowledgeType)(int)item.Type,
    Tags = item.Tags,
    Metadata = item.Metadata,
    CreatedAt = DateTime.UtcNow,
    UpdatedAt = DateTime.UtcNow
};

// Use:
var modelItem = KnowledgeItemAdapter.ToModel(item);
```

### ValidationIssueAdapter

Converts between `TarsEngine.Models.ValidationIssue` and `TarsEngine.Services.Interfaces.ValidationIssue`.

```csharp
// Instead of:
var modelIssue = new ModelValidationIssue
{
    Description = issue.Description,
    Severity = (ModelIssueSeverity)(int)issue.Severity,
    Location = issue.Location
};

// Use:
var modelIssue = ValidationIssueAdapter.ToModel(issue);
```

## Combining Monads

The library supports combining monads for complex scenarios:

```csharp
// Example of a complex operation that:
// 1. Might return null (Option)
// 2. Might fail with an exception (Result)
// 3. Is asynchronous (AsyncResult)
// 4. Needs custom formatting (Printable)

AsyncResultError<Option<DateTime>, Exception> result = GetLastLoginTimeAsync("user123");

await result.RunAsync().ContinueWith(t => {
    string message = t.Result.Match(
        success: option => option.Match(
            some: time => $"Last login: {time.ToPrintable(Printers.DateTime)}",
            none: () => "No previous login found"
        ),
        failure: ex => $"Error retrieving login time: {ex.Message}"
    );
    
    Console.WriteLine(message);
});
```

## Benefits

- **Type Safety**: Monads provide compile-time type checking for operations that might fail or return null.
- **Composability**: Monads can be composed together to build complex operations from simple ones.
- **Readability**: Monads make code more readable by making the flow of data explicit.
- **Maintainability**: Monads make code more maintainable by separating concerns and reducing side effects.
- **Testability**: Monads make code more testable by making dependencies explicit.

## Getting Started

To use the monads library, simply add a reference to the `TarsEngine.Monads` namespace:

```csharp
using TarsEngine.Monads;
```

For convenience, you can also use the static `Monad` class:

```csharp
using static TarsEngine.Monads.Monad;
```

This allows you to write code like:

```csharp
Option<string> nameOption = Some("John");
Result<int, Exception> result = Success(42);
AsyncResult<bool> asyncResult = AsyncFromResult(true);
```
