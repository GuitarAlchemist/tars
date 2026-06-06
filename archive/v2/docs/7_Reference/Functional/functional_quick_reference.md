# TARS Functional Programming - Quick Reference

## Validation (Error Accumulation)

### When to Use

- Form validation (collect ALL errors)
- Configuration parsing
- Multi-field validation

### Basic Usage

```fsharp
open Tars.Core

// Define your error type
type ValidationError =
    | EmailInvalid
    | PasswordTooShort
    | AgeTooYoung

// Validate each field
let validateEmail email =
    if email.Contains "@" then Validation.valid email
    else Validation.invalid EmailInvalid

let validatePassword pwd =
    if pwd.Length >= 8 then Validation.valid pwd
    else Validation.invalid PasswordTooShort

let validateAge age =
    if age >= 18 then Validation.valid age
    else Validation.invalid AgeTooYoung

// Combine validations - ALL errors are collected!
let result = 
    match validateEmail "bad", validatePassword "short", validateAge 10 with
    | Valid e, Valid p, Valid a -> Valid { Email = e; Password = p; Age = a }
    | _ -> Invalid [EmailInvalid; PasswordTooShort; AgeTooYoung]
// Returns: Invalid [EmailInvalid; PasswordTooShort; AgeTooYoung]
```

### With Validators Module

```fsharp
open Tars.Core.Validators

let validateUser email password age =
    Validation.apply
        (Validation.apply
            (Validation.map (fun e p a -> { Email = e; Password = p; Age = a })
                (notEmpty EmailInvalid email))
            (satisfies (fun p -> p.Length >= 8) PasswordTooShort password))
        (inRange 18 120 AgeTooYoung age)
```

---

## AsyncResult (Async + Result)

### When to Use

- Database operations
- HTTP API calls
- File I/O
- LLM calls

### Basic Usage

```fsharp
open Tars.Core

// Individual operations
let loadConfig () : AsyncResult<Config, string> = 
    asyncResult {
        let! file = readFileAsync "config.json"
        let! parsed = parseConfig file
        return parsed
    }

let queryDatabase config : AsyncResult<Data, string> =
    asyncResult {
        let! conn = connectToDb config
        let! data = executeQuery conn "SELECT * FROM agents"
        return data
    }

// Compose them
let workflow () : AsyncResult<ProcessedData, string> =
    asyncResult {
        let! config = loadConfig()
        let! data = queryDatabase config
        let! processed = processData data
        return processed
    }

// Run it
let main () =
    async {
        let! result = workflow()
        match result with
        | Ok data -> printfn "Success: %A" data
        | Error err -> printfn "Error: %s" err
    }
    |> Async.RunSynchronously
```

### Converting to/from

```fsharp
// Result -> AsyncResult
let ar = AsyncResult.ofResult (Ok 42)

// Async -> AsyncResult
let ar2 = AsyncResult.ofAsync (async { return 42 })

// AsyncResult -> extract
async {
    let! result = myAsyncResult
    match result with
    | Ok value -> // use value
    | Error err -> // handle error
}
```

---

## Functional Operators

### Option

```fsharp
open Tars.Core.FunctionalOps

// Map: transform the value if Some
let doubled = Some 21 <!> (fun x -> x * 2)  // Some 42

// Bind: chain optional operations
let result = 
    Some "hello"
    >>= (fun s -> if s.Length > 0 then Some (s.ToUpper()) else None)
    >>= (fun s -> Some (s + "!"))
// Result: Some "HELLO!"
```

### Result

```fsharp
// Map: transform Ok values
let doubled = Ok 21 <!^ (fun x -> x * 2)  // Ok 42

// Bind: chain Result operations
let result =
    Ok "42"
    >>=^ (fun s -> try Ok (int s) with _ -> Error "Not a number")
    >>=^ (fun n -> if n > 0 then Ok n else Error "Must be positive")
// Result: Ok 42

// Kleisli composition (compose two Result-returning functions)
let parse : string -> Result<int, string> = 
    fun s -> try Ok (int s) with _ -> Error "Parse error"

let validate : int -> Result<int, string> =
    fun n -> if n > 0 then Ok n else Error "Must be positive"

let parseAndValidate = parse >=> validate
let result = parseAndValidate "42"  // Ok 42
```

---

## Real-World Examples

### Example 1: LLM Call with Error Handling

```fsharp
let callLlm (prompt: string) : AsyncResult<string, string> =
    asyncResult {
        let! response = llmService.Generate(prompt)  // AsyncResult
        let! validated = validateResponse response    // AsyncResult
        return validated
    }

// Use it
async {
    let! result = callLlm "What is TARS?"
    match result with
    | Ok answer -> printfn "LLM says: %s" answer
    | Error err -> printfn "LLM failed: %s" err
}
```

### Example 2: Agent Configuration Validation

```fsharp
type ConfigError =
    | MissingName
    | InvalidModel
    | InvalidPort

let validateAgentConfig name model port =
    let validName = notEmpty MissingName name
    let validModel = satisfies (fun m -> ["gpt-4"; "claude"].Contains m) InvalidModel model
    let validPort = inRange 1 65535 InvalidPort port
    
    match validName, validModel, validPort with
    | Valid n, Valid m, Valid p -> Valid { Name = n; Model = m; Port = p }
    | _ -> 
        // Collect all errors
        [validName; validModel; validPort]
        |> List.choose (function Invalid es -> Some es | _ -> None)
        |> List.concat
        |> Invalid

// Example usage
let result = validateAgentConfig "" "unknown-model" 99999
// Returns: Invalid [MissingName; InvalidModel; InvalidPort]
```

### Example 3: Batch Processing with Error Collection

```fsharp
let processAgents (agents: Agent list) : AsyncResult<Agent list, string> =
    agents
    |> List.map (fun agent -> asyncResult {
        let! validated = validateAgent agent
        let! processed = processAgent validated
        return processed
    })
    |> AsyncResult.sequence  // Combine all AsyncResults

// If ANY agent fails, you get the first error
// If you want ALL errors, use Validation instead of Result
```

---

## Common Patterns

### Pattern: Validated Builder

```fsharp
type UserRegistration = {
    Email: string
    Password: string
    Age: int
}

let createUser email password age =
    let vEmail = validateEmail email
    let vPassword = validatePassword password
    let vAge = validateAge age
    
    match vEmail, vPassword, vAge with
    | Valid e, Valid p, Valid a -> Valid { Email = e; Password = p; Age = a }
    | _ ->
        [vEmail; vPassword; vAge]
        |> List.choose (function Invalid es -> Some es | _ -> None)
        |> List.concat
        |> Invalid
```

### Pattern: Railway-Oriented Programming

```fsharp
let pipeline input =
    input
    |> parseInput     // Result<ParsedData, Error>
    >>=^ validateData  // Result<ValidData, Error>
    >>=^ transformData // Result<Transformed, Error>
    >>=^ saveData      // Result<Saved, Error>

// Each step either continues on "success track" or jumps to "error track"
```

### Pattern: Async Pipeline

```fsharp
let asyncPipeline input =
    asyncResult {
        let! parsed = parseInputAsync input
        let! validated = validateAsync parsed
        let! transformed = transformAsync validated
        let! saved = saveAsync transformed
        return saved
    }
```

---

## Tips & Best Practices

### DO

✅ Use Validation for multi-field error collection
✅ Use AsyncResult for async I/O with error handling
✅ Use operators for clean pipelines
✅ Pattern match to handle all cases

### ❌ DON'T

❌ Mix Validation and Result (they have different semantics)
❌ Use Validation for fail-fast scenarios (use Result)
❌ Over-use operators (readability matters)
❌ Ignore type errors (they're your friend!)

### Performance Notes

- Validation creates lists: use for small numbers of fields
- AsyncResult has minimal overhead (<1% typical)
- Operators are inlined by compiler
- Pattern matching is optimized

---

## Troubleshooting

### "The type 'Validation' is not defined"

```fsharp
// Add at top of file:
open Tars.Core
```

### "Expected AsyncResult but got Async<Result>"

```fsharp
// They're the same! AsyncResult<'T,'E> = Async<Result<'T,'E>>
// Use it directly:
let myFunction () : AsyncResult<int, string> = async { return Ok 42 }
```

### "How do I convert between types?"

```fsharp
// Result -> Validation
Validation.ofResult myResult

// Validation -> Result  
Validation.toResult myValidation

// Async -> AsyncResult
AsyncResult.ofAsync myAsync

// Result -> AsyncResult
AsyncResult.ofResult myResult
```

---

## References

- **Implementation**: `src/Tars.Core/Functional.fs`
- **Examples**: `tests/Tars.Tests/FunctionalTests.fs` (TODO)
- **Proposal**: `docs/3_Roadmap/functional_patterns_proposal.md`
- **F# Docs**: <https://fsharp.org/>
- **Railway Oriented Programming**: <https://fsharpforfunandprofit.com/rop/>
