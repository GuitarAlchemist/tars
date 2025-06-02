namespace TarsEngine.FSharp.Security.Core

open System
open Microsoft.Extensions.Configuration

/// <summary>
/// JWT configuration settings for TARS
/// </summary>
type JwtConfiguration = {
    Secret: string
    Issuer: string
    Audience: string
    ExpirationMinutes: int
    RefreshExpirationDays: int
    ClockSkewMinutes: int
    ValidateIssuer: bool
    ValidateAudience: bool
    ValidateLifetime: bool
    ValidateIssuerSigningKey: bool
}

/// <summary>
/// Module for working with JWT configuration
/// </summary>
module JwtConfiguration =
    
    /// Create default JWT configuration
    let createDefault() = {
        Secret = "CHANGE_THIS_IN_PRODUCTION_TO_A_SECURE_SECRET_KEY_AT_LEAST_256_BITS"
        Issuer = "TARS"
        Audience = "TARS-API"
        ExpirationMinutes = 60
        RefreshExpirationDays = 7
        ClockSkewMinutes = 5
        ValidateIssuer = true
        ValidateAudience = true
        ValidateLifetime = true
        ValidateIssuerSigningKey = true
    }
    
    /// Create development configuration (less secure)
    let createDevelopment() = 
        let config = createDefault()
        { config with
            ExpirationMinutes = 480 // 8 hours for development
            ClockSkewMinutes = 10
        }
    
    /// Create production configuration (more secure)
    let createProduction(secret: string) =
        let config = createDefault()
        { config with
            Secret = secret
            ExpirationMinutes = 30 // Shorter expiration for production
            ClockSkewMinutes = 2
        }
    
    /// Load configuration from IConfiguration
    let loadFromConfiguration (configuration: IConfiguration) =
        let section = configuration.GetSection("TarsSecurity")
        let config = createDefault()
        
        {
            Secret = section.["JwtSecret"] |> Option.ofObj |> Option.defaultValue config.Secret
            Issuer = section.["JwtIssuer"] |> Option.ofObj |> Option.defaultValue config.Issuer
            Audience = section.["JwtAudience"] |> Option.ofObj |> Option.defaultValue config.Audience
            ExpirationMinutes = 
                match section.["JwtExpirationMinutes"] |> Option.ofObj with
                | Some value when Int32.TryParse(value) |> fst -> Int32.Parse(value)
                | _ -> config.ExpirationMinutes
            RefreshExpirationDays = 
                match section.["JwtRefreshExpirationDays"] |> Option.ofObj with
                | Some value when Int32.TryParse(value) |> fst -> Int32.Parse(value)
                | _ -> config.RefreshExpirationDays
            ClockSkewMinutes = config.ClockSkewMinutes
            ValidateIssuer = config.ValidateIssuer
            ValidateAudience = config.ValidateAudience
            ValidateLifetime = config.ValidateLifetime
            ValidateIssuerSigningKey = config.ValidateIssuerSigningKey
        }
    
    /// Validate JWT configuration
    let validate (config: JwtConfiguration) =
        let errors = ResizeArray<string>()
        
        if String.IsNullOrWhiteSpace(config.Secret) then
            errors.Add("JWT secret cannot be empty")
        elif config.Secret.Length < 32 then
            errors.Add("JWT secret should be at least 32 characters long")
        elif config.Secret = "CHANGE_THIS_IN_PRODUCTION_TO_A_SECURE_SECRET_KEY_AT_LEAST_256_BITS" then
            errors.Add("JWT secret must be changed from default value")
        
        if String.IsNullOrWhiteSpace(config.Issuer) then
            errors.Add("JWT issuer cannot be empty")
        
        if String.IsNullOrWhiteSpace(config.Audience) then
            errors.Add("JWT audience cannot be empty")
        
        if config.ExpirationMinutes <= 0 then
            errors.Add("JWT expiration minutes must be greater than 0")
        
        if config.RefreshExpirationDays <= 0 then
            errors.Add("JWT refresh expiration days must be greater than 0")
        
        if config.ClockSkewMinutes < 0 then
            errors.Add("JWT clock skew minutes cannot be negative")
        
        if errors.Count = 0 then
            Ok "JWT configuration is valid"
        else
            Error (String.concat "; " errors)
    
    /// Convert to TarsSecurityConfig
    let toSecurityConfig (jwtConfig: JwtConfiguration) =
        {
            EnableAuthentication = true
            EnableAuthorization = true
            DefaultAuthType = JWT
            RequireHttps = false
            AllowAnonymous = false
            
            JwtSecret = jwtConfig.Secret
            JwtIssuer = jwtConfig.Issuer
            JwtAudience = jwtConfig.Audience
            JwtExpirationMinutes = jwtConfig.ExpirationMinutes
            JwtRefreshExpirationDays = jwtConfig.RefreshExpirationDays
            
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
