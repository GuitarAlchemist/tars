namespace TarsEngine.FSharp.WindowsService.API

open System
open System.Threading.Tasks
open Microsoft.AspNetCore.Mvc
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.WindowsService.Security

/// <summary>
/// Authentication request model
/// </summary>
[<CLIMutable>]
type LoginRequest = {
    Username: string
    Password: string
    Role: string option
}

/// <summary>
/// Authentication response model
/// </summary>
[<CLIMutable>]
type LoginResponse = {
    Token: string
    ExpiresAt: DateTime
    User: UserInfo
}

/// <summary>
/// User information model
/// </summary>
and [<CLIMutable>] UserInfo = {
    UserId: string
    Username: string
    Role: string
    Permissions: string list
}

/// <summary>
/// Token validation request
/// </summary>
[<CLIMutable>]
type ValidateTokenRequest = {
    Token: string
}

/// <summary>
/// Token validation response
/// </summary>
[<CLIMutable>]
type ValidateTokenResponse = {
    IsValid: bool
    User: UserInfo option
    ExpiresAt: DateTime option
    Message: string
}

/// <summary>
/// Authentication controller for TARS API
/// Handles login, token validation, and user management
/// </summary>
[<ApiController>]
[<Route("api/[controller]")>]
type AuthController(jwtAuth: SimpleJwtAuth, logger: ILogger<AuthController>) =
    inherit ControllerBase()
    
    /// Simple user validation (in production, use proper user store)
    let validateUser (username: string) (password: string) =
        // TODO: Replace with proper user authentication
        // This is a simple implementation for demonstration
        match username.ToLowerInvariant(), password with
        | "admin", "admin123" -> Some ("admin", "Administrator", "Administrator")
        | "user", "user123" -> Some ("user", "User", "User")
        | "agent", "agent123" -> Some ("agent", "Agent", "Agent")
        | "system", "system123" -> Some ("system", "System", "System")
        | _ -> None
    
    /// Login endpoint
    [<HttpPost("login")>]
    member this.Login([<FromBody>] request: LoginRequest) = task {
        try
            if String.IsNullOrWhiteSpace(request.Username) || String.IsNullOrWhiteSpace(request.Password) then
                logger.LogWarning("Login attempt with empty username or password")
                return this.BadRequest("Username and password are required") :> IActionResult
            else
                match validateUser request.Username request.Password with
                | Some (userId, username, role) ->
                    // Override role if specified in request
                    let finalRole =
                        match request.Role with
                        | Some roleStr -> roleStr
                        | None -> role

                    match jwtAuth.GenerateToken(userId, username, finalRole) with
                    | Ok (token, expiresAt) ->
                        let response = {
                            Token = token
                            ExpiresAt = expiresAt
                            User = {
                                UserId = userId
                                Username = username
                                Role = finalRole
                                Permissions = ["read"; "write"] // Simplified permissions
                            }
                        }

                        logger.LogInformation("User {Username} logged in successfully with role {Role}", username, finalRole)
                        return this.Ok(response) :> IActionResult

                    | Error error ->
                        logger.LogError("Failed to generate token for user {Username}: {Error}", request.Username, error)
                        return this.StatusCode(500, "Failed to generate authentication token") :> IActionResult
                
                | None ->
                    logger.LogWarning("Invalid login attempt for username: {Username}", request.Username)
                    return this.Unauthorized("Invalid username or password") :> IActionResult
                    
        with
        | ex ->
            logger.LogError(ex, "Error during login for username: {Username}", request.Username)
            return this.StatusCode(500, "Internal server error") :> IActionResult
    }
    
    /// Validate token endpoint
    [<HttpPost("validate")>]
    member this.ValidateToken([<FromBody>] request: ValidateTokenRequest) = task {
        try
            if String.IsNullOrWhiteSpace(request.Token) then
                let response = {
                    IsValid = false
                    User = None
                    ExpiresAt = None
                    Message = "Token is required"
                }
                return this.BadRequest(response) :> IActionResult
            else
                match jwtAuth.ValidateToken(request.Token) with
                | Ok (userId, username, role, expiresAt) ->
                    let response = {
                        IsValid = true
                        User = Some {
                            UserId = userId
                            Username = username
                            Role = role
                            Permissions = ["read"; "write"] // Simplified permissions
                        }
                        ExpiresAt = Some expiresAt
                        Message = "Token is valid"
                    }
                    return this.Ok(response) :> IActionResult

                | Error reason ->
                    let response = {
                        IsValid = false
                        User = None
                        ExpiresAt = None
                        Message = reason
                    }
                    return this.Ok(response) :> IActionResult
                    
        with
        | ex ->
            logger.LogError(ex, "Error validating token")
            let response = {
                IsValid = false
                User = None
                ExpiresAt = None
                Message = "Token validation error"
            }
            return this.StatusCode(500, response) :> IActionResult
    }
    

