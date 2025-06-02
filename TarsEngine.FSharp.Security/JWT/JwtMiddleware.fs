namespace TarsEngine.FSharp.Security.JWT

open System
open System.Threading.Tasks
open Microsoft.AspNetCore.Http
open Microsoft.Extensions.Logging
open Microsoft.Extensions.DependencyInjection
open TarsEngine.FSharp.Security.Core

/// <summary>
/// JWT authentication middleware for TARS endpoints
/// Validates JWT tokens and sets security context
/// </summary>
type JwtMiddleware(next: RequestDelegate, logger: ILogger<JwtMiddleware>) =
    
    /// Extract JWT token from request
    let extractToken (context: HttpContext) =
        // Try Authorization header first
        match context.Request.Headers.TryGetValue("Authorization") with
        | true, values when values.Count > 0 ->
            let authHeader = values.[0]
            if authHeader.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase) then
                Some (authHeader.Substring(7))
            else
                None
        | _ ->
            // Try query parameter as fallback
            match context.Request.Query.TryGetValue("token") with
            | true, values when values.Count > 0 -> Some values.[0]
            | _ -> None
    
    /// Set security context in HTTP context
    let setSecurityContext (context: HttpContext) (securityContext: TarsSecurityContext) =
        context.Items.["TarsSecurityContext"] <- securityContext
        
        // Also set user principal for ASP.NET Core compatibility
        let identity = System.Security.Claims.ClaimsIdentity(securityContext.Claims, "JWT")
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
        // Check for [AllowAnonymous] attribute or similar
        // For now, allow anonymous access to health checks and documentation
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
                match extractToken context with
                | Some token ->
                    // Validate JWT token
                    let tokenService = context.RequestServices.GetRequiredService<JwtTokenService>()
                    match tokenService.ValidateToken(token) with
                    | Valid securityContext ->
                        logger.LogDebug("JWT authentication successful for user: {UserId}", securityContext.UserId)
                        setSecurityContext context securityContext
                        do! next.Invoke(context)
                        
                    | Invalid reason ->
                        logger.LogWarning("JWT authentication failed: {Reason}", reason)
                        context.Response.StatusCode <- 401
                        do! context.Response.WriteAsync($"Authentication failed: {reason}")
                        
                    | Expired reason ->
                        logger.LogWarning("JWT token expired: {Reason}", reason)
                        context.Response.StatusCode <- 401
                        context.Response.Headers.Add("WWW-Authenticate", "Bearer error=\"invalid_token\", error_description=\"Token expired\"")
                        do! context.Response.WriteAsync($"Token expired: {reason}")
                        
                    | Forbidden reason ->
                        logger.LogWarning("JWT authentication forbidden: {Reason}", reason)
                        context.Response.StatusCode <- 403
                        do! context.Response.WriteAsync($"Access forbidden: {reason}")
                
                | None ->
                    // No token provided
                    if config.AllowAnonymous || allowsAnonymous context then
                        logger.LogDebug("Anonymous access allowed for path: {Path}", context.Request.Path)
                        setAnonymousContext context
                        do! next.Invoke(context)
                    else
                        logger.LogWarning("No JWT token provided for protected endpoint: {Path}", context.Request.Path)
                        context.Response.StatusCode <- 401
                        context.Response.Headers.Add("WWW-Authenticate", "Bearer")
                        do! context.Response.WriteAsync("Authentication required")
                        
        with
        | ex ->
            logger.LogError(ex, "Error in JWT middleware")
            context.Response.StatusCode <- 500
            do! context.Response.WriteAsync("Internal server error")
    }

/// <summary>
/// Extension methods for adding JWT middleware
/// </summary>
module JwtMiddlewareExtensions =
    
    open Microsoft.AspNetCore.Builder
    
    /// Add JWT middleware to the pipeline
    let useJwtAuthentication (app: IApplicationBuilder) =
        app.UseMiddleware<JwtMiddleware>()

/// <summary>
/// Helper functions for working with security context
/// </summary>
module SecurityContext =
    
    /// Get TARS security context from HTTP context
    let getTarsContext (httpContext: HttpContext) =
        match httpContext.Items.TryGetValue("TarsSecurityContext") with
        | true, context -> Some (context :?> TarsSecurityContext)
        | _ -> None
    
    /// Check if user has required permission
    let hasPermission (permission: TarsPermission) (httpContext: HttpContext) =
        match getTarsContext httpContext with
        | Some context -> TarsPermission.allows permission context.Permissions
        | None -> false
    
    /// Check if user has required role
    let hasRole (role: TarsRole) (httpContext: HttpContext) =
        match getTarsContext httpContext with
        | Some context -> context.Role = role || context.Role = System
        | None -> false
    
    /// Get current user ID
    let getCurrentUserId (httpContext: HttpContext) =
        match getTarsContext httpContext with
        | Some context -> Some context.UserId
        | None -> None
    
    /// Check if user is authenticated
    let isAuthenticated (httpContext: HttpContext) =
        match getTarsContext httpContext with
        | Some context -> context.IsAuthenticated
        | None -> false
