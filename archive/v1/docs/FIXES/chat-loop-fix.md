# Chat Loop Task Disposal Fix

## Issue Description

The TARS chat functionality was experiencing an error when running the chat loop:

```
Error in chat loop: A task may only be disposed if it is in a completion state (RanToCompletion, Faulted or Canceled).
```

This error occurred because the typing indicator task was being disposed while it was still running, which is not allowed in .NET. A task can only be disposed after it has completed, faulted, or been canceled.

## Solution

The fix involved modifying the `RunChatLoop` method in `CliSupport.cs` to properly handle the typing indicator task:

1. Added a `CancellationTokenSource` to signal the task to stop
2. Modified the typing indicator task to check for cancellation
3. Added proper cancellation and waiting for the task to complete before proceeding
4. Enhanced error handling to provide more detailed information

### Code Changes

```csharp
// Before
var typingTask = Task.Run(async () =>
{
    var typingChars = new[] { '|', '/', '-', '\\' };
    var typingIndex = 0;

    while (true)
    {
        Console.Write(typingChars[typingIndex]);
        await Task.Delay(100);
        Console.Write("\b");
        typingIndex = (typingIndex + 1) % typingChars.Length;
    }
});

// Get response from chat bot
var response = await chatBotService.SendMessageAsync(input);

// Stop typing indicator
typingTask.Dispose();
```

```csharp
// After
var typingCts = new CancellationTokenSource();

var typingTask = Task.Run(async () =>
{
    var typingChars = new[] { '|', '/', '-', '\\' };
    var typingIndex = 0;

    try
    {
        while (!typingCts.Token.IsCancellationRequested)
        {
            Console.Write(typingChars[typingIndex]);
            await Task.Delay(100, typingCts.Token);
            Console.Write("\b");
            typingIndex = (typingIndex + 1) % typingChars.Length;
        }
    }
    catch (OperationCanceledException)
    {
        // Expected when cancellation is requested
    }
}, typingCts.Token);

// Get response from chat bot
var response = await chatBotService.SendMessageAsync(input);

// Stop typing indicator
typingCts.Cancel();
try
{
    await typingTask; // Wait for the task to complete
}
catch (OperationCanceledException)
{
    // Expected when cancellation is requested
}
```

## Testing

The fix was tested by running the chat command and verifying that:

1. The typing indicator appears and disappears correctly
2. The chat bot responds to messages properly
3. No errors occur during the chat session
4. The chat session can be exited cleanly

## Lessons Learned

1. Always use a `CancellationTokenSource` when you need to cancel a long-running task
2. Never dispose a task that is still running
3. Always wait for a task to complete after canceling it
4. Provide detailed error information to help diagnose issues

## Related Documentation

- [Task Cancellation in .NET](https://docs.microsoft.com/en-us/dotnet/standard/parallel-programming/task-cancellation)
- [Task Disposal](https://docs.microsoft.com/en-us/dotnet/api/system.threading.tasks.task.dispose)
