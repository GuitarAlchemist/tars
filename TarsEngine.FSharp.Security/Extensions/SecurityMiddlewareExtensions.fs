namespace TarsEngine.FSharp.Security.Extensions

open Microsoft.AspNetCore.Builder
open Microsoft.Extensions.DependencyInjection
open TarsEngine.FSharp.Security.Core
open TarsEngine.FSharp.Security.JWT
open TarsEngine.FSharp.Security.ApiKey

/// <summary>
/// Extension methods for configuring TARS security middleware
/// </summary>
module SecurityMiddlewareExtensions =
    
    /// Add TARS security middleware to the pipeline
    let useTarsSecurity (app: IApplicationBuilder) =
        let serviceProvider = app.ApplicationServices
        let config = serviceProvider.GetRequiredService<TarsSecurityConfig>()
        
        // Add CORS if enabled
        if config.EnableCors then
            app.UseCors() |> ignore
        
        // Add security headers
        app.Use(fun context next ->
            // Add security headers
            context.Response.Headers.Add("X-Content-Type-Options", "nosniff")
            context.Response.Headers.Add("X-Frame-Options", "DENY")
            context.Response.Headers.Add("X-XSS-Protection", "1; mode=block")
            context.Response.Headers.Add("Referrer-Policy", "strict-origin-when-cross-origin")
            
            if config.RequireHttps then
                context.Response.Headers.Add("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
            
            next.Invoke()
        ) |> ignore
        
        // Add authentication middleware based on configuration
        match config.DefaultAuthType with
        | JWT ->
            app.UseMiddleware<JwtMiddleware>() |> ignore
        | ApiKey ->
            app.UseMiddleware<ApiKeyMiddleware>() |> ignore
        | _ ->
            // Use both JWT and API key middleware for flexibility
            app.UseMiddleware<ApiKeyMiddleware>() |> ignore
            app.UseMiddleware<JwtMiddleware>() |> ignore
        
        // Add authorization if enabled
        if config.EnableAuthorization then
            app.UseAuthorization() |> ignore
        
        app
    
    /// Add JWT authentication middleware
    let useJwtAuthentication (app: IApplicationBuilder) =
        app.UseMiddleware<JwtMiddleware>()
    
    /// Add API key authentication middleware
    let useApiKeyAuthentication (app: IApplicationBuilder) =
        app.UseMiddleware<ApiKeyMiddleware>()
    
    /// Add security headers middleware
    let useSecurityHeaders (app: IApplicationBuilder) =
        app.Use(fun context next ->
            // Standard security headers
            context.Response.Headers.Add("X-Content-Type-Options", "nosniff")
            context.Response.Headers.Add("X-Frame-Options", "DENY")
            context.Response.Headers.Add("X-XSS-Protection", "1; mode=block")
            context.Response.Headers.Add("Referrer-Policy", "strict-origin-when-cross-origin")
            context.Response.Headers.Add("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'")
            
            next.Invoke()
        )
    
    /// Add HTTPS redirection if required
    let useHttpsRedirection (app: IApplicationBuilder) (config: TarsSecurityConfig) =
        if config.RequireHttps then
            app.UseHttpsRedirection() |> ignore
        app
    
    /// Add rate limiting middleware (placeholder)
    let useRateLimiting (app: IApplicationBuilder) (config: TarsSecurityConfig) =
        if config.EnableRateLimiting then
            // TODO: Implement rate limiting middleware
            app.Use(fun context next ->
                // Placeholder for rate limiting logic
                next.Invoke()
            ) |> ignore
        app

/// <summary>
/// Helper functions for security middleware configuration
/// </summary>
module SecurityHelpers =
    
    /// Check if request is for a public endpoint
    let isPublicEndpoint (path: string) =
        let lowerPath = path.ToLowerInvariant()
        lowerPath.Contains("/health") ||
        lowerPath.Contains("/swagger") ||
        lowerPath.Contains("/docs") ||
        lowerPath.Contains("/favicon.ico") ||
        lowerPath.Contains("/api/auth/login")
    
    /// Check if request requires admin privileges
    let requiresAdminAccess (path: string) =
        let lowerPath = path.ToLowerInvariant()
        lowerPath.Contains("/api/admin/") ||
        lowerPath.Contains("/api/service/config") ||
        lowerPath.Contains("/api/service/restart") ||
        lowerPath.Contains("/api/agents/manage") ||
        lowerPath.Contains("/api/system/")
    
    /// Get required permission for endpoint
    let getRequiredPermission (path: string) (method: string) =
        let lowerPath = path.ToLowerInvariant()
        let lowerMethod = method.ToUpperInvariant()
        
        match lowerPath, lowerMethod with
        | p, "GET" when p.Contains("/api/service") -> Some ReadService
        | p, "POST" when p.Contains("/api/service") -> Some WriteService
        | p, "PUT" when p.Contains("/api/service") -> Some WriteService
        | p, "DELETE" when p.Contains("/api/service") -> Some ManageService
        
        | p, "GET" when p.Contains("/api/agents") -> Some ReadAgents
        | p, "POST" when p.Contains("/api/agents") -> Some WriteAgents
        | p, "PUT" when p.Contains("/api/agents") -> Some WriteAgents
        | p, "DELETE" when p.Contains("/api/agents") -> Some ManageAgents
        
        | p, "GET" when p.Contains("/api/tasks") -> Some ReadTasks
        | p, "POST" when p.Contains("/api/tasks") -> Some WriteTasks
        | p, "PUT" when p.Contains("/api/tasks") -> Some WriteTasks
        | p, "DELETE" when p.Contains("/api/tasks") -> Some ManageTasks
        
        | p, "GET" when p.Contains("/api/roadmaps") -> Some ReadRoadmaps
        | p, "POST" when p.Contains("/api/roadmaps") -> Some WriteRoadmaps
        | p, "PUT" when p.Contains("/api/roadmaps") -> Some WriteRoadmaps
        | p, "DELETE" when p.Contains("/api/roadmaps") -> Some ManageRoadmaps
        
        | p, _ when requiresAdminAccess p -> Some SystemAdmin
        
        | _ -> None
