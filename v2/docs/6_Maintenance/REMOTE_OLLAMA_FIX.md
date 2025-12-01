# TARS v2 - Remote Ollama Configuration Summary

## ✅ You're Right

Your `secrets.json` file DOES have an `OLLAMA_BASE_URL` configured for Open WebUI, but the Chat command isn't using it yet.

## Current Status

### What's Working

- ✅ `Tars.Security.CredentialVault.loadSecretsFromDisk()` - function exists  
- ✅ `secrets.json` - your file with OLLAMA_BASE_URL
- ✅ All tests passing (39/39)
- ✅ SafetyGate, EventBus, Architecture docs - all complete

### What Needs Fixing

- ❌ `Chat.fs` doesn't load `secrets.json` yet
- ❌ `Chat.fs` hardcoded to `http://localhost:11434/`
- ❌ Return type mismatch in Chat.run (returns `Task<unit>` should be `Task<int>`)

## The Fix

### File: `src/Tars.Interface.Cli/Commands/Chat.fs`

**Add after line 16:**

```fsharp
// Load secrets from secrets.json
let secretsPath = "secrets.json"
match CredentialVault.loadSecretsFromDisk secretsPath with
| Ok () -> logger.Information("Secrets loaded successfully")
| Error err -> logger.Warning("Could not load secrets: {Error}", err)
```

**Replace line 28:**

```fsharp
// OLD:
{ OllamaBaseUri = Uri("http://localhost:11434/")

// NEW:
{ OllamaBaseUri = 
    match CredentialVault.getSecret "OLLAMA_BASE_URL" with
    | Ok url -> 
        logger.Information("Using Ollama from secrets: {Url}", url)
        Uri(url)
    | Error _ -> 
        logger.Information("Using default Ollama: http://localhost:11434/")
        Uri("http://localhost:11434/")
```

**Add required import at top:**

```fsharp
open Tars.Security  // Add this line
```

**Fix return type (line 117):**

```fsharp
// Change from:
return 0

// To:
return 0  // Make sure this is the last line in the task { } block
```

## Quick Workaround (Until Fixed)

**Option 1: Edit the hardcoded URL**

Edit line 28 in `Chat.fs` directly:

```fsharp
{ OllamaBaseUri = Uri("https://your-openwebui-url/api")
```

**Option 2: Environment Variable**

```powershell
$env:OLLAMA_BASE_URL = "https://your-openwebui-url/api"
dotnet run --project src/Tars.Interface.Cli -- chat
```

## Files to Reference

- **Secrets Loading**: `src/Tars.Security/Security.fs` (lines 24-44)
- **Chat Command**: `src/Tars.Interface.Cli/Commands/Chat.fs`
- **Program Entry**: `src/Tars.Interface.Cli/Program.fs` (already loads secrets for OPENWEBUI_EMAIL/PASSWORD)

## Expected Behavior After Fix

```
[INFO] Starting TARS v2 Chat...
[INFO] Secrets loaded successfully
[INFO] Using Ollama from secrets: https://your-openwebui-instance.com/api
TARS v2 Chat initialized. Type 'exit' to quit.

User>
```

---

**Next Session Goal**: Apply this fix properly with full testing to ensure Open WebUI works correctly! 🎯
