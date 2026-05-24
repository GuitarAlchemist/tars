namespace TarsEngine.FSharp.Core.Context

open System
open System.Text.RegularExpressions
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Context.Types

/// Configuration for security guards
type SecurityConfig = {
    AllowedTools: Set<string>
    EnableSanitization: bool
    EnableInjectionDetection: bool
    ContentFilters: string list
    MaxContextLength: int
    SuspiciousPatterns: string list
}

/// Security guard for context and tool calls
type ContextSecurityGuard(config: SecurityConfig, logger: ILogger<ContextSecurityGuard>) =
    
    /// Patterns that indicate potential prompt injection
    let injectionPatterns = [
        @"(?i)ignore\s+previous\s+instructions?"
        @"(?i)you\s+are\s+now\s+a?"
        @"(?i)change\s+your\s+rules?"
        @"(?i)forget\s+everything"
        @"(?i)system\s*:\s*"
        @"(?i)assistant\s*:\s*"
        @"(?i)user\s*:\s*"
        @"(?i)#\s*instruction\s*:"
        @"(?i)override\s+system"
        @"(?i)jailbreak"
        @"(?i)pretend\s+to\s+be"
        @"(?i)act\s+as\s+if"
        @"(?i)roleplay\s+as"
    ]
    
    /// Patterns for sensitive content detection
    let sensitivePatterns = [
        @"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b" // Email addresses
        @"\b(?:\d{4}[-\s]?){3}\d{4}\b" // Credit card numbers
        @"\b\d{3}-\d{2}-\d{4}\b" // SSN format
        @"(?i)password\s*[:=]\s*\S+" // Password assignments
        @"(?i)api[_-]?key\s*[:=]\s*\S+" // API keys
        @"(?i)secret\s*[:=]\s*\S+" // Secrets
        @"(?i)token\s*[:=]\s*[A-Za-z0-9+/=]{20,}" // Tokens
        @"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----" // Private keys
    ]
    
    /// TARS-specific security patterns
    let tarsSecurityPatterns = [
        @"(?i)system\.shutdown"
        @"(?i)process\.kill"
        @"(?i)file\.delete.*\*"
        @"(?i)rm\s+-rf\s+/"
        @"(?i)format\s+c:"
        @"(?i)del\s+/s\s+/q"
    ]
    
    /// Detect prompt injection attempts
    let detectPromptInjection (text: string) =
        injectionPatterns
        |> List.exists (fun pattern -> 
            Regex.IsMatch(text, pattern, RegexOptions.IgnoreCase ||| RegexOptions.Multiline))
    
    /// Detect sensitive content
    let detectSensitiveContent (text: string) =
        sensitivePatterns
        |> List.choose (fun pattern ->
            let matches = Regex.Matches(text, pattern, RegexOptions.IgnoreCase)
            if matches.Count > 0 then
                Some (pattern, matches.Count)
            else
                None)
    
    /// Detect dangerous TARS operations
    let detectDangerousOperations (text: string) =
        tarsSecurityPatterns
        |> List.filter (fun pattern -> 
            Regex.IsMatch(text, pattern, RegexOptions.IgnoreCase))
    
    /// Sanitize text by removing/replacing dangerous content
    let sanitizeText (text: string) =
        let mutable sanitized = text
        
        // Remove prompt injection patterns
        for pattern in injectionPatterns do
            sanitized <- Regex.Replace(sanitized, pattern, "[SANITIZED]", RegexOptions.IgnoreCase ||| RegexOptions.Multiline)
        
        // Redact sensitive content
        for pattern in sensitivePatterns do
            sanitized <- Regex.Replace(sanitized, pattern, "[REDACTED]", RegexOptions.IgnoreCase)
        
        // Remove dangerous operations
        for pattern in tarsSecurityPatterns do
            sanitized <- Regex.Replace(sanitized, pattern, "[BLOCKED]", RegexOptions.IgnoreCase)
        
        // Remove excessive whitespace
        sanitized <- Regex.Replace(sanitized, @"\n\s*\n\s*\n", "\n\n")
        
        // Limit length
        if sanitized.Length > config.MaxContextLength then
            sanitized <- sanitized.Substring(0, config.MaxContextLength) + "\n[TRUNCATED]"
        
        sanitized.Trim()
    
    /// Validate tool call against allowlist
    let validateToolCall (tool: string) (args: string) =
        // Check if tool is in allowlist
        if not (config.AllowedTools.Contains(tool)) then
            logger.LogWarning("Tool call blocked - not in allowlist: {Tool}", tool)
            false
        else
            // Additional validation for specific tools
            match tool.ToLower() with
            | "fs.read" ->
                // Ensure no path traversal
                not (args.Contains("..") || args.Contains("~"))
            | "git.diff" ->
                // Ensure reasonable git operations
                not (args.Contains("--force") || args.Contains("--hard"))
            | "run.tests" ->
                // Ensure safe test execution
                not (args.Contains("rm") || args.Contains("del"))
            | "cuda.benchmark" ->
                // Ensure safe CUDA operations
                not (args.Contains("--unsafe") || args.Contains("--override"))
            | "metascript.execute" ->
                // Ensure safe metascript execution
                not (detectDangerousOperations args)
            | _ ->
                // Default validation - check for dangerous patterns
                not (detectDangerousOperations args)
    
    /// Log security event
    let logSecurityEvent (eventType: string) (details: string) (severity: string) =
        match severity.ToLower() with
        | "high" -> logger.LogError("Security event [{EventType}]: {Details}", eventType, details)
        | "medium" -> logger.LogWarning("Security event [{EventType}]: {Details}", eventType, details)
        | "low" -> logger.LogInformation("Security event [{EventType}]: {Details}", eventType, details)
        | _ -> logger.LogDebug("Security event [{EventType}]: {Details}", eventType, details)
    
    interface IContextGuard with
        
        member _.SanitizeContext(context) =
            if not config.EnableSanitization then
                context
            else
                logger.LogDebug("Sanitizing context of length {Length}", context.Length)
                
                let sanitized = sanitizeText context
                
                if sanitized <> context then
                    let changeRatio = float (context.Length - sanitized.Length) / float context.Length
                    logSecurityEvent "context_sanitized" 
                        $"Sanitized {changeRatio:P1} of context content" 
                        (if changeRatio > 0.1 then "medium" else "low")
                
                sanitized
        
        member _.ApproveToolCall(tool, args) =
            logger.LogDebug("Validating tool call: {Tool} with args: {Args}", tool, args)
            
            let approved = validateToolCall tool args
            
            if approved then
                logSecurityEvent "tool_call_approved" $"Tool: {tool}" "low"
            else
                logSecurityEvent "tool_call_blocked" $"Tool: {tool}, Args: {args}" "high"
            
            approved
        
        member _.DetectInjection(text) =
            if not config.EnableInjectionDetection then
                false
            else
                let hasInjection = detectPromptInjection text
                let sensitiveContent = detectSensitiveContent text
                let dangerousOps = detectDangerousOperations text
                
                if hasInjection then
                    logSecurityEvent "prompt_injection_detected" "Potential prompt injection attempt" "high"
                
                if not sensitiveContent.IsEmpty then
                    logSecurityEvent "sensitive_content_detected" 
                        $"Found {sensitiveContent.Length} types of sensitive content" "medium"
                
                if not dangerousOps.IsEmpty then
                    logSecurityEvent "dangerous_operation_detected" 
                        $"Found {dangerousOps.Length} dangerous operations" "high"
                
                hasInjection || not sensitiveContent.IsEmpty || not dangerousOps.IsEmpty
