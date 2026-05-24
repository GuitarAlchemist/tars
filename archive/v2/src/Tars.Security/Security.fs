namespace Tars.Security

open System
open System.IO

module CredentialVault =
    open System.Collections.Concurrent

    let private secrets = ConcurrentDictionary<string, string>()

    /// Register a secret in the in-memory vault
    let registerSecret (key: string) (value: string) =
        secrets.AddOrUpdate(key, value, (fun _ _ -> value)) |> ignore

    /// Retrieve a secret from in-memory vault or environment variables
    let getSecret (key: string) =
        // Environment always wins to allow secure overrides
        match Environment.GetEnvironmentVariable(key) with
        | value when not (isNull value) -> Ok value
        | _ ->
            match secrets.TryGetValue(key) with
            | true, value -> Ok value
            | _ -> Error $"Secret '%s{key}' not found"

    /// Load secrets from a JSON file on disk
    /// The file should be a simple key-value map: {"KEY": "VALUE"}
    let loadSecretsFromDisk (filePath: string) =
        try
            if File.Exists(filePath) then
                let json = File.ReadAllText(filePath)
                // Simple parsing to avoid heavy dependencies for now, or use System.Text.Json
                // Assuming simple flat JSON object
                let dict =
                    System.Text.Json.JsonSerializer.Deserialize<System.Collections.Generic.Dictionary<string, string>>(
                        json
                    )

                for kvp in dict do
                    registerSecret kvp.Key kvp.Value

                Ok()
            else
                Error $"Secrets file not found: {filePath}"
        with ex ->
            Error $"Failed to load secrets from disk: {ex.Message}"

module FilesystemPolicy =
    /// Ensure a requested path is within a specific base directory
    let validatePath (basePath: string) (requestedPath: string) =
        try
            let fullBase = Path.GetFullPath(basePath)
            // Combine to handle relative paths correctly against the base
            let fullRequested = Path.GetFullPath(Path.Combine(fullBase, requestedPath))

            // Ensure the base path ends with a separator to prevent partial matches (e.g., /data vs /database)
            let safeBase =
                if fullBase.EndsWith(Path.DirectorySeparatorChar.ToString()) then
                    fullBase
                else
                    fullBase + Path.DirectorySeparatorChar.ToString()

            if fullRequested.StartsWith(safeBase, StringComparison.OrdinalIgnoreCase) then
                Ok fullRequested
            else
                Error $"Access denied: Path '%s{requestedPath}' is outside the allowed directory '%s{basePath}'."
        with ex ->
            Error $"Invalid path: %s{ex.Message}"
