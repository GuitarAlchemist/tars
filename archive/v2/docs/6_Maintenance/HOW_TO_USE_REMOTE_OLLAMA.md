# Using Remote Ollama/Open WebUI with TARS v2

## Current Issue

TARS chat is currently hardcoded to use `http://localhost:11434/` for Ollama.

Your `secrets.json` file contains `OLLAMA_BASE_URL` for the remote Open WebUI connection, but the Chat command isn't loading it yet.

## Quick Fix

### Option 1: Edit Chat.fs Directly (Manual)

Edit `src/Tars.Interface.Cli/Commands/Chat.fs` and change line 28:

**Before:**

```fsharp
{ OllamaBaseUri = Uri("http://localhost:11434/")
```

**After:** (replace with your Open WebUI URL from secrets.json)

```fsharp
{ OllamaBaseUri = Uri("YOUR_OPENWEBUI_URL_HERE")  // e.g., "https://openwebui.example.com/api"
```

### Option 2: Use Environment Variable

Set an environment variable before running:

```powershell
$env:OLLAMA_BASE_URL = "https://your-openwebui-url-here/api"
dotnet run --project src/Tars.Interface.Cli -- chat
```

## Proper Fix (TODO)

The Chat command should be updated to:

1. Load `secrets.json` using `Tars.Security.CredentialVault.loadSecretsFromDisk`
2. Read `OLLAMA_BASE_URL` from secrets
3. Fall back to localhost if not found

### Code Pattern

```fsharp
// At the start of Chat.run:
let secretsPath = "secrets.json"
match CredentialVault.loadSecretsFromDisk secretsPath with
| Ok () -> logger.Information("Secrets loaded")
| Error err -> logger.Warning("Could not load secrets: {Error}", err)

// When setting up routing config:
let ollamaBaseUri =
    match CredentialVault.getSecret "OLLAMA_BASE_URL" with
    | Ok url -> Uri(url)
    | Error _ -> Uri("http://localhost:11434/")

let routingCfg: RoutingConfig =
    { OllamaBaseUri = ollamaBaseUri  // Use the secret
      ...
```

## Your secrets.json Format

Ensure your `secrets.json` looks like:

```json
{
  "OLLAMA_BASE_URL": "https://your-open-webui-instance.com/api",
  "OTHER_SECRET": "value"
}
```

## Testing

Once updated, run:

```powershell
cd c:\Users\spare\source\repos\tars\v2
dotnet run --project src/Tars.Interface.Cli -- chat
```

The logs should show:

```
[INFO] Secrets loaded successfully
[INFO] Using Ollama from secrets: https://your-url-here
```

---

**Note**: The proper fix with automatic secrets loading will be implemented in the next session to avoid introducing syntax errors during manual edits.
