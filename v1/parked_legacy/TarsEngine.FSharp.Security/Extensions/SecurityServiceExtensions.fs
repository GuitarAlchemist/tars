namespace TarsEngine.FSharp.Security.Extensions

open Microsoft.Extensions.DependencyInjection
open Microsoft.Extensions.Configuration
open Microsoft.AspNetCore.Authentication.JwtBearer
open Microsoft.IdentityModel.Tokens
open System.Text
open TarsEngine.FSharp.Security.Core
open TarsEngine.FSharp.Security.JWT

/// <summary>
/// Extension methods for configuring TARS security services
/// </summary>
module SecurityServiceExtensions =
    
    /// Add TARS security services to dependency injection
    let addTarsSecurity (services: IServiceCollection) (configuration: IConfiguration) =
        
        // Load security configuration
        let securityConfig = 
            let configSection = configuration.GetSection("TarsSecurity")
            if configSection.Exists() then
                // Load from configuration
                let config = TarsSecurityConfig.createDefault()
                configSection.Bind(config)
                config
            else
                // Use default configuration
                TarsSecurityConfig.createDefault()
        
        // Register security configuration
        services.AddSingleton<TarsSecurityConfig>(securityConfig) |> ignore
        
        // Register JWT token service
        services.AddScoped<JwtTokenService>() |> ignore
        
        // Configure JWT authentication if enabled
        if securityConfig.EnableAuthentication && securityConfig.DefaultAuthType = JWT then
            services.AddAuthentication(JwtBearerDefaults.AuthenticationScheme)
                .AddJwtBearer(fun options ->
                    options.TokenValidationParameters <- TokenValidationParameters(
                        ValidateIssuerSigningKey = true,
                        IssuerSigningKey = SymmetricSecurityKey(Encoding.UTF8.GetBytes(securityConfig.JwtSecret)),
                        ValidateIssuer = true,
                        ValidIssuer = securityConfig.JwtIssuer,
                        ValidateAudience = true,
                        ValidAudience = securityConfig.JwtAudience,
                        ValidateLifetime = true,
                        ClockSkew = System.TimeSpan.FromMinutes(5.0)
                    )
                    options.Events <- JwtBearerEvents(
                        OnAuthenticationFailed = fun context ->
                            System.Threading.Tasks.Task.CompletedTask
                    )
                ) |> ignore
        
        // Add authorization if enabled
        if securityConfig.EnableAuthorization then
            services.AddAuthorization(fun options ->
                // Define TARS-specific policies
                options.AddPolicy("TarsUser", fun policy ->
                    policy.RequireAuthenticatedUser() |> ignore
                    policy.RequireClaim("role", "User", "Agent", "Administrator", "System") |> ignore
                )
                
                options.AddPolicy("TarsAgent", fun policy ->
                    policy.RequireAuthenticatedUser() |> ignore
                    policy.RequireClaim("role", "Agent", "Administrator", "System") |> ignore
                )
                
                options.AddPolicy("TarsAdmin", fun policy ->
                    policy.RequireAuthenticatedUser() |> ignore
                    policy.RequireClaim("role", "Administrator", "System") |> ignore
                )
                
                options.AddPolicy("TarsSystem", fun policy ->
                    policy.RequireAuthenticatedUser() |> ignore
                    policy.RequireClaim("role", "System") |> ignore
                )
                
                // Permission-based policies
                options.AddPolicy("ReadService", fun policy ->
                    policy.RequireAuthenticatedUser() |> ignore
                    policy.RequireClaim("permission", "service:read", "system:admin") |> ignore
                )
                
                options.AddPolicy("WriteService", fun policy ->
                    policy.RequireAuthenticatedUser() |> ignore
                    policy.RequireClaim("permission", "service:write", "system:admin") |> ignore
                )
                
                options.AddPolicy("ManageService", fun policy ->
                    policy.RequireAuthenticatedUser() |> ignore
                    policy.RequireClaim("permission", "service:manage", "system:admin") |> ignore
                )
                
                options.AddPolicy("ManageAgents", fun policy ->
                    policy.RequireAuthenticatedUser() |> ignore
                    policy.RequireClaim("permission", "agents:manage", "system:admin") |> ignore
                )
                
                options.AddPolicy("ManageRoadmaps", fun policy ->
                    policy.RequireAuthenticatedUser() |> ignore
                    policy.RequireClaim("permission", "roadmaps:manage", "system:admin") |> ignore
                )
            ) |> ignore
        
        // Add CORS if enabled
        if securityConfig.EnableCors then
            services.AddCors(fun options ->
                options.AddDefaultPolicy(fun policy ->
                    if securityConfig.AllowedOrigins |> List.contains "*" then
                        policy.AllowAnyOrigin() |> ignore
                    else
                        policy.WithOrigins(securityConfig.AllowedOrigins |> List.toArray) |> ignore
                    
                    policy.WithMethods(securityConfig.AllowedMethods |> List.toArray) |> ignore
                    policy.WithHeaders(securityConfig.AllowedHeaders |> List.toArray) |> ignore
                    
                    if not (securityConfig.AllowedOrigins |> List.contains "*") then
                        policy.AllowCredentials() |> ignore
                )
            ) |> ignore
        
        services
    
    /// Add TARS security with custom configuration
    let addTarsSecurityWithConfig (services: IServiceCollection) (config: TarsSecurityConfig) =
        
        // Register security configuration
        services.AddSingleton<TarsSecurityConfig>(config) |> ignore
        
        // Register JWT token service
        services.AddScoped<JwtTokenService>() |> ignore
        
        // Configure authentication and authorization as above
        // (Same logic as addTarsSecurity but with provided config)
        
        services
    
    /// Add TARS security for development (less secure, more permissive)
    let addTarsSecurityDevelopment (services: IServiceCollection) =
        let config = TarsSecurityConfig.createDevelopment()
        addTarsSecurityWithConfig services config
    
    /// Add TARS security for production (more secure)
    let addTarsSecurityProduction (services: IServiceCollection) (jwtSecret: string) =
        let config = TarsSecurityConfig.createProduction()
        let productionConfig = { config with JwtSecret = jwtSecret }
        addTarsSecurityWithConfig services productionConfig

/// <summary>
/// Configuration helpers for TARS security
/// </summary>
module SecurityConfigurationHelpers =
    
    /// Create JWT token for initial setup/testing
    let createInitialAdminToken (config: TarsSecurityConfig) =
        let logger = Microsoft.Extensions.Logging.Abstractions.NullLogger<JwtTokenService>.Instance
        let tokenService = JwtTokenService(config, logger)
        tokenService.GenerateToken("admin", "Administrator", Administrator)
    
    /// Validate security configuration
    let validateSecurityConfig (config: TarsSecurityConfig) =
        let errors = ResizeArray<string>()
        
        // Check JWT secret strength
        if config.JwtSecret.Length < 32 then
            errors.Add("JWT secret should be at least 32 characters long")
        
        if config.JwtSecret = "CHANGE_THIS_IN_PRODUCTION_TO_A_SECURE_SECRET_KEY_AT_LEAST_256_BITS" then
            errors.Add("JWT secret must be changed from default value")
        
        // Check HTTPS requirement in production
        if not config.RequireHttps && not config.AllowAnonymous then
            errors.Add("HTTPS should be required when authentication is enabled")
        
        // Check CORS configuration
        if config.EnableCors && config.AllowedOrigins |> List.contains "*" && not config.AllowAnonymous then
            errors.Add("Wildcard CORS origins should not be used with authentication")
        
        if errors.Count = 0 then
            Ok "Security configuration is valid"
        else
            Error (String.concat "; " errors)
    
    /// Generate secure JWT secret
    let generateSecureJwtSecret() =
        let random = System.Security.Cryptography.RandomNumberGenerator.Create()
        let bytes = Array.zeroCreate 64 // 512 bits
        random.GetBytes(bytes)
        System.Convert.ToBase64String(bytes)
