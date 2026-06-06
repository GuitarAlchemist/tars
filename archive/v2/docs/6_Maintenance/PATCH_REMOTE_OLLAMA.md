# PATCH: Add Remote Ollama Support to Chat.fs

## File: src/Tars.Interface.Cli/Commands/Chat.fs

### Change 1: Add Tars.Security import

**After line 12 (`open Tars.Llm.LlmService`):**

```fsharp
open Tars.Security
```

### Change 2: Load secrets

**After line 17 (`logger.Information("Starting TARS v2 Chat...")`):**

```fsharp
        // Load secrets from secrets.json
        let secretsPath = "secrets.json"
        match CredentialVault.loadSecretsFromDisk secretsPath with
        | Ok () -> logger.Information("Secrets loaded successfully")
        | Error err -> logger.Warning("Could not load secrets: {Error}", err)
```

### Change 3: Use OLLAMA_BASE_URL from secrets

**Replace line 28 Only (`{ OllamaBaseUri = Uri("http://localhost:11434/")`):**

Replace with:

```fsharp
        //Get Ollama base URL from secrets, fall back to localhost
        let ollamaBaseUri =
            match CredentialVault.getSecret "OLLAMA_BASE_URL" with
            | Ok url -> 
                logger.Information("Using Ollama from secrets: {Url}", url)
                Uri(url)
            | Error _ -> 
                logger.Information("Using default Ollama: http://localhost:11434/")
                Uri("http://localhost:11434/")

        // Initialize LLM Service
        let routingCfg: RoutingConfig =
            { OllamaBaseUri = ollamaBaseUri
```

---

## **Apply via manual edit or use this command:**

```powershell
# Make a backup first
cp src/Tars.Interface.Cli/Commands/Chat.fs src/Tars.Interface.Cli/Commands/Chat.fs.bak

# Then manually apply the three changes above
```

## Test

```powershell
dotnet build Tars.sln
dotnet run --project src/Tars.Interface.Cli -- chat
```

Expected log output:

```
[INFO] Starting TARS v2 Chat...
[INFO] Secrets loaded successfully  
[INFO] Using Ollama from secrets: https://your-openwebui-url
```
