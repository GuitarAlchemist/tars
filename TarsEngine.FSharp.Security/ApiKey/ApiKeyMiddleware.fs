namespace TarsEngine.FSharp.Security.ApiKey

open System
open System.Threading.Tasks
open Microsoft.AspNetCore.Http
open Microsoft.Extensions.Logging
open Microsoft.Extensions.DependencyInjection
open TarsEngine.FSharp.Security.Core

/// <summary>
/// API Key authentication middleware for TARS endpoints
/// Validates API keys and sets security context
/// </summary>
type ApiKeyMiddleware(next: RequestDelegate, logger: ILogger<ApiKeyMiddleware>) =
    
    /// Extract API key from request
    let extractApiKey (context: HttpContext) (config: TarsSecurityConfig) =
        // Try configured header first
        match context.Request.Headers.TryGetValue(config.ApiKeyHeader) with
        | true, values when values.Count > 0 -> Some values.[0]
        | _ ->
            // Try query parameter as fallback
            match context.Request.Query.TryGetValue("api_key") with
            | true, values when values.Count > 0 -> Some values.[0]
            | _ -> None
    
    /// Set security context in HTTP context
    let setSecurityContext (context: HttpContext) (securityContext: TarsSecurityContext) =
        context.Items.["TarsSecurityContext"] <- securityContext
        
        // Also set user principal for ASP.NET Core compatibility
        let identity = System.Security.Claims.ClaimsIdentity(securityContext.Claims, "ApiKey")
        let principal = System.Security.Claims.ClaimsPrincipal(identity)
        context.User <- principal
    
    /// Set anonymous context
    let setAnonymousContext (context: HttpContext) =
        let anonymousContext = {
            UserId = "anonymous"
            Username = "Anonymous"
            Role = Anonymous
            Permissions = []
            AuthType = None
            IsAuthenticated = false
            ExpiresAt = None
            Claims = []
        }
        setSecurityContext context anonymousContext
    
    /// Check if endpoint allows anonymous access
    let allowsAnonymous (context: HttpContext) =
        let path = context.Request.Path.Value.ToLowerInvariant()
        path.Contains("/health") || 
        path.Contains("/swagger") || 
        path.Contains("/docs") ||
        path.Contains("/favicon.ico")
    
    /// Process the request
    member this.InvokeAsync(context: HttpContext) = task {
        try
            let config = context.RequestServices.GetRequiredService<TarsSecurityConfig>()
            
            // Skip authentication if disabled
            if not config.EnableAuthentication then
                setAnonymousContext context
                do! next.Invoke(context)
            else
                match extractApiKey context config with
                | Some apiKey ->
                    // Validate API key
                    let apiKeyService = context.RequestServices.GetRequiredService<ApiKeyService>()
                    match apiKeyService.ValidateApiKey(apiKey) with
                    | Valid securityContext ->
                        logger.LogDebug("API key authentication successful for: {UserId}", securityContext.UserId)
                        setSecurityContext context securityContext
                        do! next.Invoke(context)
                        
                    | Invalid reason ->
                        logger.LogWarning("API key authentication failed: {Reason}", reason)
                        context.Response.StatusCode <- 401
                        do! context.Response.WriteAsync($"Authentication failed: {reason}")
                        
                    | Expired reason ->
                        logger.LogWarning("API key expired: {Reason}", reason)
                        context.Response.StatusCode <- 401
                        do! context.Response.WriteAsync($"API key expired: {reason}")
                        
                    | Forbidden reason ->
                        logger.LogWarning("API key authentication forbidden: {Reason}", reason)
                        context.Response.StatusCode <- 403
                        do! context.Response.WriteAsync($"Access forbidden: {reason}")
                
                | None ->
                    // No API key provided - fall back to other authentication methods
                    // or allow anonymous if configured
                    if config.AllowAnonymous || allowsAnonymous context then
                        logger.LogDebug("No API key provided, allowing anonymous access for path: {Path}", context.Request.Path)
                        setAnonymousContext context
                        do! next.Invoke(context)
                    else
                        // Continue to next middleware (might be JWT or other auth)
                        do! next.Invoke(context)
                        
        with
        | ex ->
            logger.LogError(ex, "Error in API key middleware")
            context.Response.StatusCode <- 500
            do! context.Response.WriteAsync("Internal server error")
    }

/// <summary>
/// Extension methods for adding API key middleware
/// </summary>
module ApiKeyMiddlewareExtensions =
    
    open Microsoft.AspNetCore.Builder
    
    /// Add API key middleware to the pipeline
    let useApiKeyAuthentication (app: IApplicationBuilder) =
        app.UseMiddleware<ApiKeyMiddleware>()
