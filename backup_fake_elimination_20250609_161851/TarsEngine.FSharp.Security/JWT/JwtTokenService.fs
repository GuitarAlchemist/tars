namespace TarsEngine.FSharp.Security.JWT

open System
open System.Text
open System.Security.Claims
open System.IdentityModel.Tokens.Jwt
open Microsoft.IdentityModel.Tokens
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Security.Core

/// <summary>
/// JWT token service for TARS authentication
/// Handles token generation, validation, and refresh
/// </summary>
type JwtTokenService(config: TarsSecurityConfig, logger: ILogger<JwtTokenService>) =
    
    let jwtHandler = JwtSecurityTokenHandler()
    
    /// Get signing key from configuration
    let getSigningKey() =
        let keyBytes = Encoding.UTF8.GetBytes(config.JwtSecret)
        SymmetricSecurityKey(keyBytes)
    
    /// Get token validation parameters
    let getValidationParameters() =
        TokenValidationParameters(
            ValidateIssuerSigningKey = true,
            IssuerSigningKey = getSigningKey(),
            ValidateIssuer = true,
            ValidIssuer = config.JwtIssuer,
            ValidateAudience = true,
            ValidAudience = config.JwtAudience,
            ValidateLifetime = true,
            ClockSkew = TimeSpan.FromMinutes(5.0)
        )
    
    /// Create claims for a user
    let createClaims (userId: string) (username: string) (role: TarsRole) (permissions: TarsPermission list) =
        let claims = ResizeArray<Claim>()
        
        // Standard claims
        claims.Add(Claim(JwtRegisteredClaimNames.Sub, userId))
        claims.Add(Claim(JwtRegisteredClaimNames.UniqueName, username))
        claims.Add(Claim(JwtRegisteredClaimNames.Jti, Guid.NewGuid().ToString()))
        claims.Add(Claim(JwtRegisteredClaimNames.Iat, DateTimeOffset.UtcNow.ToUnixTimeSeconds().ToString(), ClaimValueTypes.Integer64))
        
        // TARS-specific claims
        claims.Add(Claim("role", TarsRole.toString role))
        claims.Add(Claim("auth_type", "JWT"))
        
        // Permission claims
        for permission in permissions do
            claims.Add(Claim("permission", TarsPermission.toString permission))
        
        claims.ToArray()
    
    /// Generate JWT token for user
    member this.GenerateToken(userId: string, username: string, role: TarsRole, ?permissions: TarsPermission list) =
        try
            let userPermissions = permissions |> Option.defaultValue (TarsRole.getDefaultPermissions role)
            let claims = createClaims userId username role userPermissions
            
            let now = DateTime.UtcNow
            let expires = now.AddMinutes(float config.JwtExpirationMinutes)
            
            let tokenDescriptor = SecurityTokenDescriptor(
                Subject = ClaimsIdentity(claims),
                Expires = expires,
                Issuer = config.JwtIssuer,
                Audience = config.JwtAudience,
                SigningCredentials = SigningCredentials(getSigningKey(), SecurityAlgorithms.HmacSha256)
            )
            
            let token = jwtHandler.CreateToken(tokenDescriptor)
            let tokenString = jwtHandler.WriteToken(token)
            
            let tokenInfo = {
                Token = tokenString
                RefreshToken = None // TODO: Implement refresh tokens
                ExpiresAt = expires
                IssuedAt = now
                Subject = userId
                Issuer = config.JwtIssuer
                Audience = config.JwtAudience
            }
            
            logger.LogInformation("JWT token generated for user: {UserId}", userId)
            Ok tokenInfo
            
        with
        | ex ->
            logger.LogError(ex, "Failed to generate JWT token for user: {UserId}", userId)
            Error $"Token generation failed: {ex.Message}"
    
    /// Validate JWT token
    member this.ValidateToken(token: string) =
        try
            let validationParameters = getValidationParameters()
            let mutable validatedToken: SecurityToken = null
            
            let principal = jwtHandler.ValidateToken(token, validationParameters, &validatedToken)
            
            // Extract claims
            let userId =
                let claim = principal.FindFirst(JwtRegisteredClaimNames.Sub)
                if claim <> null then Some claim.Value else None
            let username =
                let claim = principal.FindFirst(JwtRegisteredClaimNames.UniqueName)
                if claim <> null then Some claim.Value else None
            let roleString =
                let claim = principal.FindFirst("role")
                if claim <> null then Some claim.Value else None
            let permissionClaims = principal.FindAll("permission") |> Seq.map (fun c -> c.Value) |> Seq.toList
            
            match userId, username, roleString with
            | Some uid, Some uname, Some roleStr ->
                let role = TarsRole.fromString roleStr
                let permissions = 
                    permissionClaims 
                    |> List.choose TarsPermission.fromString
                
                let securityContext = {
                    UserId = uid
                    Username = uname
                    Role = role
                    Permissions = permissions
                    AuthType = JWT
                    IsAuthenticated = true
                    ExpiresAt = Some validatedToken.ValidTo
                    Claims = principal.Claims |> Seq.toList
                }
                
                logger.LogDebug("JWT token validated for user: {UserId}", uid)
                Valid securityContext
                
            | _ ->
                logger.LogWarning("JWT token missing required claims")
                Invalid "Token missing required claims"
                
        with
        | :? SecurityTokenExpiredException ->
            logger.LogWarning("JWT token expired")
            Expired "Token has expired"
        | :? SecurityTokenInvalidSignatureException ->
            logger.LogWarning("JWT token has invalid signature")
            Invalid "Invalid token signature"
        | :? SecurityTokenValidationException as ex ->
            logger.LogWarning("JWT token validation failed: {Message}", ex.Message)
            Invalid $"Token validation failed: {ex.Message}"
        | ex ->
            logger.LogError(ex, "Unexpected error validating JWT token")
            Invalid $"Token validation error: {ex.Message}"
    
    /// Extract user ID from token without full validation (for logging/debugging)
    member this.ExtractUserId(token: string) =
        try
            let jwtToken = jwtHandler.ReadJwtToken(token)
            jwtToken.Claims 
            |> Seq.tryFind (fun c -> c.Type = JwtRegisteredClaimNames.Sub)
            |> Option.map (fun c -> c.Value)
        with
        | _ -> None
    
    /// Check if token is expired
    member this.IsTokenExpired(token: string) =
        try
            let jwtToken = jwtHandler.ReadJwtToken(token)
            jwtToken.ValidTo < DateTime.UtcNow
        with
        | _ -> true
    
    /// Generate API token for service-to-service communication
    member this.GenerateServiceToken(serviceName: string, ?expirationHours: int) =
        let expHours = expirationHours |> Option.defaultValue 24
        let permissions = [SystemAdmin] // Service tokens get full permissions
        
        this.GenerateToken($"service:{serviceName}", serviceName, System, permissions)
    
    /// Generate agent token for agent authentication
    member this.GenerateAgentToken(agentId: string, agentType: string, ?expirationHours: int) =
        let expHours = expirationHours |> Option.defaultValue 8
        let permissions = TarsRole.getDefaultPermissions Agent

        this.GenerateToken($"agent:{agentId}", $"{agentType}:{agentId}", Agent, permissions)

    /// Refresh an existing token (placeholder for future implementation)
    member this.RefreshToken(refreshToken: string) =
        // TODO: Implement refresh token logic
        Error "Refresh tokens not yet implemented"
