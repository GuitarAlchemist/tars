namespace Tars.Security

open System
open System.IO

module CredentialVault =
    open System.Collections.Concurrent

    let private secrets = ConcurrentDictionary<string, string>()

    /// Register a secret in the in-memory vault
    let registerSecret (key: string) (value: string) =
        secrets.AddOrUpdate(key, value, (fun _ _ -> value)) |> ignore

    /// Retrieve a secret from environment variables or in-memory vault
    let getSecret (key: string) =
        match Environment.GetEnvironmentVariable(key) with
        | null -> 
            match secrets.TryGetValue(key) with
            | true, value -> Ok value
            | _ -> Error $"Secret '%s{key}' not found"
        | value -> Ok value

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
