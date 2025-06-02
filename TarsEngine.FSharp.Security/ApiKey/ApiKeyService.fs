namespace TarsEngine.FSharp.Security.ApiKey

open System
open System.Security.Cryptography
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Security.Core

/// <summary>
/// API Key service for TARS authentication
/// Handles API key generation, validation, and management
/// </summary>
type ApiKeyService(config: TarsSecurityConfig, logger: ILogger<ApiKeyService>) =
    
    /// Generate secure API key
    let generateApiKey() =
        use rng = RandomNumberGenerator.Create()
        let bytes = Array.zeroCreate 32
        rng.GetBytes(bytes)
        Convert.ToBase64String(bytes).Replace("+", "-").Replace("/", "_").TrimEnd('=')
    
    /// Hash API key for storage
    let hashApiKey (key: string) =
        use sha256 = SHA256.Create()
        let bytes = System.Text.Encoding.UTF8.GetBytes(key)
        let hash = sha256.ComputeHash(bytes)
        Convert.ToBase64String(hash)
    
    /// Create API key with prefix
    let createKeyWithPrefix (key: string) =
        match config.ApiKeyPrefix with
        | Some prefix -> $"{prefix}{key}"
        | None -> key
    
    /// Extract key from prefixed key
    let extractKeyFromPrefix (prefixedKey: string) =
        match config.ApiKeyPrefix with
        | Some prefix when prefixedKey.StartsWith(prefix) -> 
            prefixedKey.Substring(prefix.Length)
        | _ -> prefixedKey
    
    /// Generate new API key
    member this.GenerateApiKey(name: string, permissions: TarsPermission list, ?expirationDays: int) =
        try
            let key = generateApiKey()
            let keyWithPrefix = createKeyWithPrefix key
            let keyHash = hashApiKey key
            let expiration = 
                match expirationDays with
                | Some days -> Some (DateTime.UtcNow.AddDays(float days))
                | None -> None
            
            let apiKeyInfo = {
                KeyId = Guid.NewGuid().ToString()
                KeyHash = keyHash
                Name = name
                Permissions = permissions
                ExpiresAt = expiration
                IsActive = true
                CreatedAt = DateTime.UtcNow
                LastUsedAt = None
            }
            
            logger.LogInformation("API key generated: {Name} with {PermissionCount} permissions", name, permissions.Length)
            Ok (keyWithPrefix, apiKeyInfo)
            
        with
        | ex ->
            logger.LogError(ex, "Failed to generate API key for: {Name}", name)
            Error $"API key generation failed: {ex.Message}"
    
    /// Validate API key (placeholder - in production, use proper storage)
    member this.ValidateApiKey(apiKey: string) =
        try
            let key = extractKeyFromPrefix apiKey
            let keyHash = hashApiKey key
            
            // TODO: Replace with proper API key storage lookup
            // This is a simple implementation for demonstration
            let mockApiKeys = [
                {
                    KeyId = "demo-key-1"
                    KeyHash = keyHash
                    Name = "Demo API Key"
                    Permissions = [ReadService; ReadAgents; ReadTasks]
                    ExpiresAt = Some (DateTime.UtcNow.AddDays(30.0))
                    IsActive = true
                    CreatedAt = DateTime.UtcNow.AddDays(-1.0)
                    LastUsedAt = None
                }
            ]
            
            match mockApiKeys |> List.tryFind (fun k -> k.KeyHash = keyHash && k.IsActive) with
            | Some keyInfo ->
                // Check expiration
                match keyInfo.ExpiresAt with
                | Some expiry when expiry < DateTime.UtcNow ->
                    logger.LogWarning("API key expired: {KeyId}", keyInfo.KeyId)
                    Expired "API key has expired"
                | _ ->
                    let securityContext = {
                        UserId = $"apikey:{keyInfo.KeyId}"
                        Username = keyInfo.Name
                        Role = User // API keys default to User role
                        Permissions = keyInfo.Permissions
                        AuthType = ApiKey
                        IsAuthenticated = true
                        ExpiresAt = keyInfo.ExpiresAt
                        Claims = []
                    }
                    
                    logger.LogDebug("API key validated: {KeyId}", keyInfo.KeyId)
                    Valid securityContext
            | None ->
                logger.LogWarning("Invalid API key provided")
                Invalid "Invalid API key"
                
        with
        | ex ->
            logger.LogError(ex, "Error validating API key")
            Invalid $"API key validation error: {ex.Message}"
    
    /// Revoke API key (placeholder)
    member this.RevokeApiKey(keyId: string) =
        try
            // TODO: Implement API key revocation in storage
            logger.LogInformation("API key revoked: {KeyId}", keyId)
            Ok "API key revoked successfully"
        with
        | ex ->
            logger.LogError(ex, "Failed to revoke API key: {KeyId}", keyId)
            Error $"Failed to revoke API key: {ex.Message}"
    
    /// List API keys (placeholder)
    member this.ListApiKeys() =
        try
            // TODO: Implement API key listing from storage
            let mockKeys = [
                {
                    KeyId = "demo-key-1"
                    KeyHash = "***"
                    Name = "Demo API Key"
                    Permissions = [ReadService; ReadAgents; ReadTasks]
                    ExpiresAt = Some (DateTime.UtcNow.AddDays(30.0))
                    IsActive = true
                    CreatedAt = DateTime.UtcNow.AddDays(-1.0)
                    LastUsedAt = None
                }
            ]
            Ok mockKeys
        with
        | ex ->
            logger.LogError(ex, "Failed to list API keys")
            Error $"Failed to list API keys: {ex.Message}"
