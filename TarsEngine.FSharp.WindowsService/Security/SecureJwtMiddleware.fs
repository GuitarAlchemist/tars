namespace TarsEngine.FSharp.WindowsService.Security

open System
open System.Threading.Tasks
open Microsoft.AspNetCore.Http
open Microsoft.Extensions.Logging
open Microsoft.Extensions.DependencyInjection

/// <summary>
/// Enhanced JWT middleware with security escalation integration
/// Monitors authentication attempts and escalates security incidents to DevSecOps agent
/// </summary>
type SecureJwtMiddleware(next: RequestDelegate, jwtAuth: SimpleJwtAuth, logger: ILogger<SecureJwtMiddleware>) =
    
    let mutable failedAttempts = Map.empty<string, (int * DateTime)> // IP -> (count, last attempt)
    let maxFailedAttempts = 5
    let lockoutDuration = TimeSpan.FromMinutes(15.0)
    
    /// Extract client information from request
    let extractClientInfo (context: HttpContext) =
        let ipAddress = 
            match context.Request.Headers.TryGetValue("X-Forwarded-For") with
            | true, values when values.Count > 0 -> Some values.[0]
            | _ -> 
                match context.Connection.RemoteIpAddress with
                | null -> None
                | ip -> Some (ip.ToString())
        
        let userAgent = 
            match context.Request.Headers.TryGetValue("User-Agent") with
            | true, values when values.Count > 0 -> Some values.[0]
            | _ -> None
        
        (ipAddress, userAgent)
    
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
    
    /// Check if IP is locked out due to failed attempts
    let isIpLockedOut (ipAddress: string) =
        match failedAttempts.TryFind(ipAddress) with
        | Some (count, lastAttempt) when count >= maxFailedAttempts ->
            DateTime.UtcNow - lastAttempt < lockoutDuration
        | _ -> false
    
    /// Record failed authentication attempt
    let recordFailedAttempt (ipAddress: string option) (escalationManager: SecurityEscalationManager option) =
        match ipAddress with
        | Some ip ->
            let (currentCount, _) = failedAttempts.TryFind(ip) |> Option.defaultValue (0, DateTime.UtcNow)
            let newCount = currentCount + 1
            failedAttempts <- failedAttempts.Add(ip, (newCount, DateTime.UtcNow))
            
            // Report to security escalation manager
            match escalationManager with
            | Some manager ->
                let severity = if newCount >= maxFailedAttempts then High else Medium
                let incidentType = if newCount >= maxFailedAttempts then BruteForceAttack else AuthenticationFailure
                
                manager.ReportSecurityIncident(
                    incidentType,
                    severity,
                    $"Authentication failure from IP {ip}",
                    $"Failed authentication attempt #{newCount} from IP address {ip}",
                    source = "JwtMiddleware",
                    ipAddress = Some ip,
                    evidence = Map [
                        ("FailedAttemptCount", newCount.ToString())
                        ("IsLockedOut", (newCount >= maxFailedAttempts).ToString())
                        ("LockoutDuration", lockoutDuration.ToString())
                    ]
                ) |> ignore
                
                if newCount >= maxFailedAttempts then
                    logger.LogWarning("🔒 IP address {IpAddress} locked out due to {FailedAttempts} failed authentication attempts", ip, newCount)
            | None ->
                logger.LogWarning("⚠️ Security escalation manager not available for failed attempt reporting")
        | None ->
            logger.LogWarning("⚠️ Cannot record failed attempt - IP address unknown")
    
    /// Record successful authentication
    let recordSuccessfulAuth (ipAddress: string option) (userId: string) (escalationManager: SecurityEscalationManager option) =
        // Clear failed attempts for this IP
        match ipAddress with
        | Some ip ->
            failedAttempts <- failedAttempts.Remove(ip)
            
            // Report successful authentication to escalation manager
            match escalationManager with
            | Some manager ->
                manager.ReportSecurityIncident(
                    AuthenticationFailure, // Using as success event
                    Low,
                    $"Successful authentication for user {userId}",
                    $"User {userId} successfully authenticated from IP {ip}",
                    source = "JwtMiddleware",
                    userId = Some userId,
                    ipAddress = Some ip,
                    evidence = Map [
                        ("AuthenticationResult", "Success")
                        ("ClearedFailedAttempts", "true")
                    ]
                ) |> ignore
            | None -> ()
        | None -> ()
    
    /// Check for suspicious activity patterns
    let checkSuspiciousActivity (context: HttpContext) (escalationManager: SecurityEscalationManager option) =
        let (ipAddress, userAgent) = extractClientInfo context
        
        // Check for suspicious user agents
        match userAgent with
        | Some ua when ua.Contains("bot") || ua.Contains("crawler") || ua.Contains("scanner") ->
            match escalationManager with
            | Some manager ->
                manager.ReportSecurityIncident(
                    SuspiciousActivity,
                    Medium,
                    "Suspicious User Agent Detected",
                    $"Potentially automated request with user agent: {ua}",
                    source = "JwtMiddleware",
                    ipAddress = ipAddress,
                    userAgent = Some ua,
                    evidence = Map [
                        ("UserAgent", ua)
                        ("RequestPath", context.Request.Path.Value)
                        ("RequestMethod", context.Request.Method)
                    ]
                ) |> ignore
                
                logger.LogWarning("🤖 Suspicious user agent detected from {IpAddress}: {UserAgent}", ipAddress |> Option.defaultValue "Unknown", ua)
            | None -> ()
        | _ -> ()
    
    /// Check if endpoint allows anonymous access
    let allowsAnonymous (context: HttpContext) =
        let path = context.Request.Path.Value.ToLowerInvariant()
        path.Contains("/health") || 
        path.Contains("/swagger") || 
        path.Contains("/docs") ||
        path.Contains("/favicon.ico") ||
        path.Contains("/api/auth/login")
    
    /// Process the request with security monitoring
    member this.InvokeAsync(context: HttpContext) = task {
        try
            let (ipAddress, userAgent) = extractClientInfo context
            let escalationManager = context.RequestServices.GetService<SecurityEscalationManager>()
            
            // Check for suspicious activity
            checkSuspiciousActivity context (Some escalationManager)
            
            // Check if IP is locked out
            let isLockedOut =
                match ipAddress with
                | Some ip when isIpLockedOut ip ->
                    logger.LogWarning("🔒 Blocked request from locked out IP: {IpAddress}", ip)

                    // Report lockout violation
                    match escalationManager with
                    | Some manager ->
                        manager.ReportSecurityIncident(
                            UnauthorizedAccess,
                            High,
                            "Access attempt from locked out IP",
                            $"Request blocked from locked out IP address: {ip}",
                            source = "JwtMiddleware",
                            ipAddress = Some ip,
                            userAgent = userAgent,
                            evidence = Map [
                                ("LockoutStatus", "Active")
                                ("RequestPath", context.Request.Path.Value)
                                ("RequestMethod", context.Request.Method)
                            ]
                        ) |> ignore
                    | None -> ()

                    context.Response.StatusCode <- 429 // Too Many Requests
                    do! context.Response.WriteAsync("Too many failed authentication attempts. Please try again later.")
                    true
                | _ -> false

            if not isLockedOut then
                match extractToken context with
                | Some token ->
                    // Validate JWT token
                    match jwtAuth.ValidateToken(token) with
                    | Ok (userId, username, role, expiresAt) ->
                        logger.LogDebug("🔐 JWT authentication successful for user: {UserId}", userId)

                        // Record successful authentication
                        recordSuccessfulAuth ipAddress userId (Some escalationManager)

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
                    logger.LogWarning("🔐 JWT authentication failed: {Reason} from IP: {IpAddress}", reason, ipAddress |> Option.defaultValue "Unknown")
                    
                    // Record failed attempt and check for escalation
                    recordFailedAttempt ipAddress (Some escalationManager)
                    
                    // Report token tampering if signature is invalid
                    if reason.Contains("signature") then
                        match escalationManager with
                        | Some manager ->
                            manager.ReportSecurityIncident(
                                TokenTampering,
                                High,
                                "Invalid JWT token signature detected",
                                $"Potential token tampering attempt: {reason}",
                                source = "JwtMiddleware",
                                ipAddress = ipAddress,
                                userAgent = userAgent,
                                evidence = Map [
                                    ("TokenValidationError", reason)
                                    ("PotentialTampering", "true")
                                ]
                            ) |> ignore
                        | None -> ()
                    
                    context.Response.StatusCode <- 401
                    context.Response.Headers.Add("WWW-Authenticate", "Bearer")
                    do! context.Response.WriteAsync($"Authentication failed: {reason}")
            
            | None ->
                // No token provided
                if allowsAnonymous context then
                    logger.LogDebug("🔓 Anonymous access allowed for path: {Path}", context.Request.Path)
                    do! next.Invoke(context)
                else
                    logger.LogWarning("🔐 No JWT token provided for protected endpoint: {Path} from IP: {IpAddress}", context.Request.Path, ipAddress |> Option.defaultValue "Unknown")
                    
                    // Report unauthorized access attempt
                    match escalationManager with
                    | Some manager ->
                        manager.ReportSecurityIncident(
                            UnauthorizedAccess,
                            Medium,
                            "Unauthorized access attempt",
                            $"Access attempt to protected endpoint without authentication: {context.Request.Path}",
                            source = "JwtMiddleware",
                            ipAddress = ipAddress,
                            userAgent = userAgent,
                            evidence = Map [
                                ("RequestPath", context.Request.Path.Value)
                                ("RequestMethod", context.Request.Method)
                                ("AuthenticationProvided", "false")
                            ]
                        ) |> ignore
                    | None -> ()
                    
                    context.Response.StatusCode <- 401
                    context.Response.Headers.Add("WWW-Authenticate", "Bearer")
                    do! context.Response.WriteAsync("Authentication required")
                    
        with
        | ex ->
            logger.LogError(ex, "❌ Error in secure JWT middleware")
            
            // Report middleware error as security incident
            let escalationManager = context.RequestServices.GetService<SecurityEscalationManager>()
            match escalationManager with
            | Some manager ->
                manager.ReportSecurityIncident(
                    SystemCompromise,
                    Critical,
                    "JWT middleware error",
                    $"Critical error in JWT authentication middleware: {ex.Message}",
                    source = "JwtMiddleware",
                    evidence = Map [
                        ("ExceptionType", ex.GetType().Name)
                        ("ExceptionMessage", ex.Message)
                        ("StackTrace", ex.StackTrace)
                    ]
                ) |> ignore
            | None -> ()
            
            context.Response.StatusCode <- 500
            do! context.Response.WriteAsync("Internal server error")
    }
