# JSON Escaping Fix for Ollama API

## Issue Description

During the implementation of the TARS self-improvement system, we encountered an issue with the Ollama API calls. When sending prompts containing newlines, quotes, or other special characters, the API would return a 400 Bad Request error with the message:

```
{"error":"invalid character '\\n' in string literal"}
```

This occurred because the JSON string was not properly escaped when constructing the request body.

## Root Cause

The original implementation constructed the JSON request body using string interpolation without properly escaping special characters:

```fsharp
let requestBody =
    sprintf "{\"model\": \"%s\", \"prompt\": \"%s\", \"stream\": false}"
        model prompt
```

This approach fails when the `prompt` contains characters that need to be escaped in JSON strings, such as:
- Newlines (`\n`)
- Carriage returns (`\r`)
- Tabs (`\t`)
- Quotes (`"`)
- Backslashes (`\`)

## Solution

We implemented a proper JSON string escaping function to handle all special characters:

```fsharp
// Properly escape the prompt for JSON
let escapedPrompt = 
    prompt
    |> fun s -> s.Replace("\\", "\\\\")
    |> fun s -> s.Replace("\"", "\\\"")
    |> fun s -> s.Replace("\n", "\\n")
    |> fun s -> s.Replace("\r", "\\r")
    |> fun s -> s.Replace("\t", "\\t")

let requestBody =
    sprintf "{\"model\": \"%s\", \"prompt\": \"%s\", \"stream\": false}"
        model escapedPrompt
```

This solution:
1. First escapes backslashes (to avoid double-escaping other characters)
2. Escapes quotes
3. Escapes newlines, carriage returns, and tabs
4. Uses the escaped string in the JSON construction

## Implementation

The fix was applied to both the `SelfAnalyzer` and `SelfImprover` modules in the `TarsEngine.SelfImprovement` library:

```fsharp
// In Library.fs
// For the SelfAnalyzer module
let escapedPrompt = 
    prompt
    |> fun s -> s.Replace("\\", "\\\\")
    |> fun s -> s.Replace("\"", "\\\"")
    |> fun s -> s.Replace("\n", "\\n")
    |> fun s -> s.Replace("\r", "\\r")
    |> fun s -> s.Replace("\t", "\\t")
```

## Verification

After implementing the fix, we successfully ran the self-analyze command on a test file:

```
dotnet run --project TarsCli/TarsCli.csproj -- self-analyze --file test_code.cs --model llama2
```

The command completed successfully, identifying issues in the test code:
- Magic numbers on multiple lines
- Inefficient string concatenation in a loop

## Lessons Learned

1. Always properly escape strings when constructing JSON manually
2. Consider using a JSON serialization library for more complex JSON construction
3. Test API calls with inputs containing special characters early in development

## Future Improvements

For future development, we should consider:

1. Using a proper JSON serialization library like `Newtonsoft.Json` or `System.Text.Json`
2. Creating a dedicated helper function for Ollama API calls
3. Adding more robust error handling for API communication issues
