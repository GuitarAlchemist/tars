namespace TarsEngine.FSharp.WindowsService.Security

open System
open System.Text
open System.Security.Claims
open System.IdentityModel.Tokens.Jwt
open Microsoft.IdentityModel.Tokens
open Microsoft.Extensions.Logging
open Microsoft.AspNetCore.Http
open System.Threading.Tasks

/// <summary>
/// Simple JWT authentication for TARS Windows Service
/// Lightweight implementation without complex dependencies
/// </summary>
type SimpleJwtAuth(jwtSecret: string, logger: ILogger<SimpleJwtAuth>) =
    
    let jwtHandler = JwtSecurityTokenHandler()
    let issuer = "TARS"
    let audience = "TARS-API"
    
    /// Get signing key
    let getSigningKey() =
        let keyBytes = Encoding.UTF8.GetBytes(jwtSecret)
        SymmetricSecurityKey(keyBytes)
    
    /// Generate JWT token
    member this.GenerateToken(userId: string, username: string, role: string) =
        try
            let claims = [
                Claim(JwtRegisteredClaimNames.Sub, userId)
                Claim(JwtRegisteredClaimNames.UniqueName, username)
                Claim(JwtRegisteredClaimNames.Jti, Guid.NewGuid().ToString())
                Claim("role", role)
                Claim("auth_type", "JWT")
            ]
            
            let now = DateTime.UtcNow
            let expires = now.AddHours(1.0) // 1 hour expiration
            
            let tokenDescriptor = SecurityTokenDescriptor(
                Subject = ClaimsIdentity(claims),
                Expires = expires,
                Issuer = issuer,
                Audience = audience,
                SigningCredentials = SigningCredentials(getSigningKey(), SecurityAlgorithms.HmacSha256)
            )
            
            let token = jwtHandler.CreateToken(tokenDescriptor)
            let tokenString = jwtHandler.WriteToken(token)
            
            logger.LogInformation("JWT token generated for user: {UserId}", userId)
            Ok (tokenString, expires)
            
        with
        | ex ->
            logger.LogError(ex, "Failed to generate JWT token for user: {UserId}", userId)
            Error $"Token generation failed: {ex.Message}"
    
    /// Validate JWT token
    member this.ValidateToken(token: string) =
        try
            let validationParameters = TokenValidationParameters(
                ValidateIssuerSigningKey = true,
                IssuerSigningKey = getSigningKey(),
                ValidateIssuer = true,
                ValidIssuer = issuer,
                ValidateAudience = true,
                ValidAudience = audience,
                ValidateLifetime = true,
                ClockSkew = TimeSpan.FromMinutes(5.0)
            )
            
            let mutable validatedToken: SecurityToken = null
            let principal = jwtHandler.ValidateToken(token, validationParameters, &validatedToken)
            
            // Extract user information
            let userId = 
                let claim = principal.FindFirst(JwtRegisteredClaimNames.Sub)
                if claim <> null then claim.Value else "unknown"
            
            let username = 
                let claim = principal.FindFirst(JwtRegisteredClaimNames.UniqueName)
                if claim <> null then claim.Value else "Unknown"
            
            let role = 
                let claim = principal.FindFirst("role")
                if claim <> null then claim.Value else "User"
            
            logger.LogDebug("JWT token validated for user: {UserId}", userId)
            Ok (userId, username, role, validatedToken.ValidTo)
            
        with
        | :? SecurityTokenExpiredException ->
            logger.LogWarning("JWT token expired")
            Error "Token has expired"
        | :? SecurityTokenInvalidSignatureException ->
            logger.LogWarning("JWT token has invalid signature")
            Error "Invalid token signature"
        | :? SecurityTokenValidationException as ex ->
            logger.LogWarning("JWT token validation failed: {Message}", ex.Message)
            Error $"Token validation failed: {ex.Message}"
        | ex ->
            logger.LogError(ex, "Unexpected error validating JWT token")
            Error $"Token validation error: {ex.Message}"

/// <summary>
/// Simple JWT middleware for TARS endpoints
/// </summary>
type SimpleJwtMiddleware(next: RequestDelegate, jwtAuth: SimpleJwtAuth, logger: ILogger<SimpleJwtMiddleware>) =
    
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
    
    /// Check if endpoint allows anonymous access
    let allowsAnonymous (context: HttpContext) =
        let path = context.Request.Path.Value.ToLowerInvariant()
        path.Contains("/health") || 
        path.Contains("/swagger") || 
        path.Contains("/docs") ||
        path.Contains("/favicon.ico") ||
        path.Contains("/api/auth/login")
    
    /// Process the request
    member this.InvokeAsync(context: HttpContext) = task {
        try
            match extractToken context with
            | Some token ->
                // Validate JWT token
                match jwtAuth.ValidateToken(token) with
                | Ok (userId, username, role, expiresAt) ->
                    logger.LogDebug("JWT authentication successful for user: {UserId}", userId)
                    
                    // Set user context
                    let claims = [
                        Claim(ClaimTypes.NameIdentifier, userId)
                        Claim(ClaimTypes.Name, username)
                        Claim(ClaimTypes.Role, role)
                    ]
                    let identity = ClaimsIdentity(claims, "JWT")
                    let principal = ClaimsPrincipal(identity)
                    context.User <- principal
                    
                    // Store additional info in context
                    context.Items.["UserId"] <- userId
                    context.Items.["Username"] <- username
                    context.Items.["Role"] <- role
                    context.Items.["TokenExpiresAt"] <- expiresAt
                    
                    do! next.Invoke(context)
                    
                | Error reason ->
                    logger.LogWarning("JWT authentication failed: {Reason}", reason)
                    context.Response.StatusCode <- 401
                    context.Response.Headers.Add("WWW-Authenticate", "Bearer")
                    do! context.Response.WriteAsync($"Authentication failed: {reason}")
            
            | None ->
                // No token provided
                if allowsAnonymous context then
                    logger.LogDebug("Anonymous access allowed for path: {Path}", context.Request.Path)
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
/// Helper functions for JWT authentication
/// </summary>
module JwtHelpers =
    
    /// Get current user ID from HTTP context
    let getCurrentUserId (context: HttpContext) =
        match context.Items.TryGetValue("UserId") with
        | true, userId -> Some (userId :?> string)
        | _ -> None
    
    /// Get current username from HTTP context
    let getCurrentUsername (context: HttpContext) =
        match context.Items.TryGetValue("Username") with
        | true, username -> Some (username :?> string)
        | _ -> None
    
    /// Get current user role from HTTP context
    let getCurrentRole (context: HttpContext) =
        match context.Items.TryGetValue("Role") with
        | true, role -> Some (role :?> string)
        | _ -> None
    
    /// Check if user has required role
    let hasRole (requiredRole: string) (context: HttpContext) =
        match getCurrentRole context with
        | Some role -> role = requiredRole || role = "Administrator" || role = "System"
        | None -> false
    
    /// Check if user is authenticated
    let isAuthenticated (context: HttpContext) =
        context.User.Identity.IsAuthenticated
    
    /// Generate secure JWT secret
    let generateSecureSecret() =
        let random = System.Security.Cryptography.RandomNumberGenerator.Create()
        let bytes = Array.zeroCreate 64 // 512 bits
        random.GetBytes(bytes)
        Convert.ToBase64String(bytes)
