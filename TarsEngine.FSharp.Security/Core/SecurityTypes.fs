namespace TarsEngine.FSharp.Security.Core

open System
open System.Security.Claims

/// <summary>
/// Authentication types supported by TARS
/// </summary>
type TarsAuthType =
    | None
    | JWT
    | ApiKey
    | Windows
    | OAuth2

/// <summary>
/// TARS user roles for authorization
/// </summary>
type TarsRole =
    | Anonymous
    | User
    | Agent
    | Administrator
    | System

/// <summary>
/// TARS permissions for fine-grained access control
/// </summary>
type TarsPermission =
    | ReadService
    | WriteService
    | ManageService
    | ReadAgents
    | WriteAgents
    | ManageAgents
    | ReadTasks
    | WriteTasks
    | ManageTasks
    | ReadRoadmaps
    | WriteRoadmaps
    | ManageRoadmaps
    | SystemAdmin

/// <summary>
/// TARS security context for authenticated requests
/// </summary>
type TarsSecurityContext = {
    UserId: string
    Username: string
    Role: TarsRole
    Permissions: TarsPermission list
    AuthType: TarsAuthType
    IsAuthenticated: bool
    ExpiresAt: DateTime option
    Claims: Claim list
}

/// <summary>
/// JWT token information
/// </summary>
type JwtTokenInfo = {
    Token: string
    RefreshToken: string option
    ExpiresAt: DateTime
    IssuedAt: DateTime
    Subject: string
    Issuer: string
    Audience: string
}

/// <summary>
/// API key information
/// </summary>
type ApiKeyInfo = {
    KeyId: string
    KeyHash: string
    Name: string
    Permissions: TarsPermission list
    ExpiresAt: DateTime option
    IsActive: bool
    CreatedAt: DateTime
    LastUsedAt: DateTime option
}

/// <summary>
/// Security configuration for TARS
/// </summary>
type TarsSecurityConfig = {
    EnableAuthentication: bool
    EnableAuthorization: bool
    DefaultAuthType: TarsAuthType
    RequireHttps: bool
    AllowAnonymous: bool
    
    // JWT Configuration
    JwtSecret: string
    JwtIssuer: string
    JwtAudience: string
    JwtExpirationMinutes: int
    JwtRefreshExpirationDays: int
    
    // API Key Configuration
    ApiKeyHeader: string
    ApiKeyPrefix: string option
    
    // Rate limiting
    EnableRateLimiting: bool
    RateLimitPerMinute: int
    RateLimitBurstSize: int
    
    // CORS
    EnableCors: bool
    AllowedOrigins: string list
    AllowedMethods: string list
    AllowedHeaders: string list
}

/// <summary>
/// Security validation result
/// </summary>
type SecurityValidationResult =
    | Valid of TarsSecurityContext
    | Invalid of string
    | Expired of string
    | Forbidden of string

/// <summary>
/// Security events for auditing
/// </summary>
type SecurityEvent =
    | AuthenticationSuccess of string * TarsAuthType
    | AuthenticationFailure of string * TarsAuthType * string
    | AuthorizationFailure of string * TarsPermission * string
    | TokenExpired of string
    | InvalidToken of string
    | RateLimitExceeded of string
    | SuspiciousActivity of string * string

/// <summary>
/// Module for working with TARS roles
/// </summary>
module TarsRole =
    
    /// Convert role to string
    let toString = function
        | Anonymous -> "Anonymous"
        | User -> "User"
        | Agent -> "Agent"
        | Administrator -> "Administrator"
        | System -> "System"
    
    /// Parse role from string
    let fromString = function
        | "Anonymous" -> Anonymous
        | "User" -> User
        | "Agent" -> Agent
        | "Administrator" -> Administrator
        | "System" -> System
        | _ -> Anonymous
    
    /// Get default permissions for role
    let getDefaultPermissions = function
        | Anonymous -> []
        | User -> [ReadService; ReadAgents; ReadTasks; ReadRoadmaps]
        | Agent -> [ReadService; WriteService; ReadAgents; WriteAgents; ReadTasks; WriteTasks; ReadRoadmaps; WriteRoadmaps]
        | Administrator -> [ReadService; WriteService; ManageService; ReadAgents; WriteAgents; ManageAgents; ReadTasks; WriteTasks; ManageTasks; ReadRoadmaps; WriteRoadmaps; ManageRoadmaps]
        | System -> [ReadService; WriteService; ManageService; ReadAgents; WriteAgents; ManageAgents; ReadTasks; WriteTasks; ManageTasks; ReadRoadmaps; WriteRoadmaps; ManageRoadmaps; SystemAdmin]

/// <summary>
/// Module for working with TARS permissions
/// </summary>
module TarsPermission =
    
    /// Convert permission to string
    let toString = function
        | ReadService -> "service:read"
        | WriteService -> "service:write"
        | ManageService -> "service:manage"
        | ReadAgents -> "agents:read"
        | WriteAgents -> "agents:write"
        | ManageAgents -> "agents:manage"
        | ReadTasks -> "tasks:read"
        | WriteTasks -> "tasks:write"
        | ManageTasks -> "tasks:manage"
        | ReadRoadmaps -> "roadmaps:read"
        | WriteRoadmaps -> "roadmaps:write"
        | ManageRoadmaps -> "roadmaps:manage"
        | SystemAdmin -> "system:admin"
    
    /// Parse permission from string
    let fromString = function
        | "service:read" -> Some ReadService
        | "service:write" -> Some WriteService
        | "service:manage" -> Some ManageService
        | "agents:read" -> Some ReadAgents
        | "agents:write" -> Some WriteAgents
        | "agents:manage" -> Some ManageAgents
        | "tasks:read" -> Some ReadTasks
        | "tasks:write" -> Some WriteTasks
        | "tasks:manage" -> Some ManageTasks
        | "roadmaps:read" -> Some ReadRoadmaps
        | "roadmaps:write" -> Some WriteRoadmaps
        | "roadmaps:manage" -> Some ManageRoadmaps
        | "system:admin" -> Some SystemAdmin
        | _ -> None
    
    /// Check if permission allows action
    let allows (required: TarsPermission) (available: TarsPermission list) =
        available |> List.contains required ||
        available |> List.contains SystemAdmin

/// <summary>
/// Module for creating default security configurations
/// </summary>
module TarsSecurityConfig =
    
    /// Create default security configuration
    let createDefault() = {
        EnableAuthentication = true
        EnableAuthorization = true
        DefaultAuthType = JWT
        RequireHttps = false // Set to true in production
        AllowAnonymous = false
        
        JwtSecret = "CHANGE_THIS_IN_PRODUCTION_TO_A_SECURE_SECRET_KEY_AT_LEAST_256_BITS"
        JwtIssuer = "TARS"
        JwtAudience = "TARS-API"
        JwtExpirationMinutes = 60
        JwtRefreshExpirationDays = 7
        
        ApiKeyHeader = "X-API-Key"
        ApiKeyPrefix = Some "tars_"
        
        EnableRateLimiting = true
        RateLimitPerMinute = 100
        RateLimitBurstSize = 20
        
        EnableCors = true
        AllowedOrigins = ["http://localhost:3000"; "http://localhost:8080"]
        AllowedMethods = ["GET"; "POST"; "PUT"; "DELETE"; "PATCH"; "OPTIONS"]
        AllowedHeaders = ["Content-Type"; "Authorization"; "X-API-Key"]
    }
    
    /// Create development configuration (less secure, more permissive)
    let createDevelopment() = 
        let config = createDefault()
        { config with
            RequireHttps = false
            AllowAnonymous = true
            EnableRateLimiting = false
            AllowedOrigins = ["*"]
        }
    
    /// Create production configuration (more secure)
    let createProduction() =
        let config = createDefault()
        { config with
            RequireHttps = true
            AllowAnonymous = false
            EnableRateLimiting = true
            AllowedOrigins = [] // Must be explicitly configured
        }
