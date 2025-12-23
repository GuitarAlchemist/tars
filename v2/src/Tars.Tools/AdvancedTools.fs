namespace Tars.Tools.Standard

open System
open System.Collections.Generic
open System.Threading.Tasks
open Tars.Tools

module ResilienceTools =

    /// Retry counter for tracking
    let private retryCounts = Dictionary<string, int>()

    /// Circuit breaker states
    let private circuitStates = Dictionary<string, DateTime * int>() // (lastFailure, failureCount)

    [<TarsToolAttribute("retry_with_backoff",
                        "Retries an operation with exponential backoff. Input JSON: { \"operation\": \"name\", \"max_retries\": 3, \"base_delay_ms\": 1000 }")>]
    let retryWithBackoff (args: string) =
        task {
            try
                let doc = System.Text.Json.JsonDocument.Parse(args)
                let root = doc.RootElement

                let operation = root.GetProperty("operation").GetString()
                let mutable maxRetriesProp = Unchecked.defaultof<System.Text.Json.JsonElement>

                let maxRetries =
                    if root.TryGetProperty("max_retries", &maxRetriesProp) then
                        maxRetriesProp.GetInt32()
                    else
                        3

                let mutable baseDelayProp = Unchecked.defaultof<System.Text.Json.JsonElement>

                let baseDelayMs =
                    if root.TryGetProperty("base_delay_ms", &baseDelayProp) then
                        baseDelayProp.GetInt32()
                    else
                        1000

                // Track retry count
                if not (retryCounts.ContainsKey(operation)) then
                    retryCounts.[operation] <- 0

                let currentRetry = retryCounts.[operation]

                if currentRetry >= maxRetries then
                    retryCounts.[operation] <- 0 // Reset for next time
                    return $"❌ Max retries (%d{maxRetries}) reached for '%s{operation}'. Giving up."
                else
                    // Calculate delay with exponential backoff + jitter
                    let delay = baseDelayMs * (pown 2 currentRetry)
                    let jitter = Random().Next(0, delay / 4)
                    let actualDelay = Math.Min(delay + jitter, 30000) // Cap at 30s

                    retryCounts.[operation] <- currentRetry + 1

                    printfn
                        $"🔄 RETRY %d{currentRetry + 1}/%d{maxRetries} for '%s{operation}' - waiting %d{actualDelay}ms"
                    do! Task.Delay(actualDelay)

                    return
                        $"✅ Retry %d{currentRetry + 1}/%d{maxRetries} for '%s{operation}' ready. Waited %d{actualDelay}ms."
            with ex ->
                return "retry_with_backoff error: " + ex.Message
        }

    [<TarsToolAttribute("reset_retry", "Resets retry counter for an operation. Input: operation name")>]
    let resetRetry (operation: string) =
        task {
            if retryCounts.ContainsKey(operation) then
                retryCounts.[operation] <- 0

            printfn $"🔄 RESET: Retry counter for '%s{operation}'"
            return $"Retry counter reset for '%s{operation}'"
        }

    [<TarsToolAttribute("circuit_breaker",
                        "Checks if a service circuit is open (too many failures). Input JSON: { \"service\": \"name\", \"action\": \"check|trip|reset\", \"threshold\": 5, \"cooldown_sec\": 60 }")>]
    let circuitBreaker (args: string) =
        task {
            try
                let doc = System.Text.Json.JsonDocument.Parse(args)
                let root = doc.RootElement

                let service = root.GetProperty("service").GetString()
                let action = root.GetProperty("action").GetString().ToLower()
                let mutable thresholdProp = Unchecked.defaultof<System.Text.Json.JsonElement>

                let threshold =
                    if root.TryGetProperty("threshold", &thresholdProp) then
                        thresholdProp.GetInt32()
                    else
                        5

                let mutable cooldownProp = Unchecked.defaultof<System.Text.Json.JsonElement>

                let cooldownSec =
                    if root.TryGetProperty("cooldown_sec", &cooldownProp) then
                        cooldownProp.GetInt32()
                    else
                        60

                match action with
                | "check" ->
                    if circuitStates.ContainsKey(service) then
                        let (lastFailure, failCount) = circuitStates.[service]
                        let elapsed = (DateTime.Now - lastFailure).TotalSeconds

                        if failCount >= threshold && elapsed < float cooldownSec then
                            return
                                $"🔴 CIRCUIT OPEN: '%s{service}' has %d{failCount} failures. Cooldown: %.0f{float cooldownSec - elapsed}s remaining."
                        elif failCount >= threshold then
                            // Cooldown passed, allow half-open
                            return $"🟡 CIRCUIT HALF-OPEN: '%s{service}' cooldown passed. Try one request."
                        else
                            return $"🟢 CIRCUIT CLOSED: '%s{service}' has %d{failCount}/%d{threshold} failures."
                    else
                        return $"🟢 CIRCUIT CLOSED: '%s{service}' is healthy."

                | "trip" ->
                    if circuitStates.ContainsKey(service) then
                        let (_, failCount) = circuitStates.[service]
                        circuitStates.[service] <- (DateTime.Now, failCount + 1)
                    else
                        circuitStates.[service] <- (DateTime.Now, 1)

                    let (_, newCount) = circuitStates.[service]
                    printfn $"⚡ CIRCUIT TRIP: '%s{service}' failure count: %d{newCount}"
                    return $"Circuit '%s{service}' tripped. Failure count: %d{newCount}/%d{threshold}"

                | "reset" ->
                    if circuitStates.ContainsKey(service) then
                        circuitStates.Remove(service) |> ignore

                    printfn $"✅ CIRCUIT RESET: '%s{service}'"
                    return $"Circuit '%s{service}' reset to healthy state."

                | _ -> return $"Unknown action: %s{action}. Use check, trip, or reset."
            with ex ->
                return "circuit_breaker error: " + ex.Message
        }

module CacheTools =

    /// Simple in-memory cache
    let private cache = Dictionary<string, string * DateTime>()
    let private defaultTtlMinutes = 30

    [<TarsToolAttribute("cache_set",
                        "Stores a value in cache. Input JSON: { \"key\": \"name\", \"value\": \"data\", \"ttl_minutes\": 30 }")>]
    let cacheSet (args: string) =
        task {
            try
                let doc = System.Text.Json.JsonDocument.Parse(args)
                let root = doc.RootElement

                let key = root.GetProperty("key").GetString()
                let value = root.GetProperty("value").GetString()
                let mutable ttlProp = Unchecked.defaultof<System.Text.Json.JsonElement>

                let ttlMinutes =
                    if root.TryGetProperty("ttl_minutes", &ttlProp) then
                        ttlProp.GetInt32()
                    else
                        defaultTtlMinutes

                let expiry = DateTime.Now.AddMinutes(float ttlMinutes)
                cache.[key] <- (value, expiry)

                printfn $"📦 CACHE SET: '%s{key}' (expires in %d{ttlMinutes} min)"
                return $"Cached '%s{key}' with TTL %d{ttlMinutes} minutes."
            with ex ->
                return "cache_set error: " + ex.Message
        }

    [<TarsToolAttribute("cache_get", "Retrieves a value from cache. Input: cache key")>]
    let cacheGet (key: string) =
        task {
            let k = key.Trim()

            if cache.ContainsKey(k) then
                let (value, expiry) = cache.[k]

                if DateTime.Now < expiry then
                    printfn $"📦 CACHE HIT: '%s{k}'"
                    return $"CACHE HIT: %s{value}"
                else
                    cache.Remove(k) |> ignore
                    return $"CACHE MISS: '%s{k}' expired."
            else
                return $"CACHE MISS: '%s{k}' not found."
        }

    [<TarsToolAttribute("cache_clear", "Clears cache entries. Input: optional key pattern (empty = clear all)")>]
    let cacheClear (pattern: string) =
        task {
            if String.IsNullOrWhiteSpace(pattern) then
                let count = cache.Count
                cache.Clear()
                printfn $"🗑️ CACHE CLEARED: All %d{count} entries"
                return $"Cleared all %d{count} cache entries."
            else
                let keysToRemove =
                    cache.Keys |> Seq.filter (fun k -> k.Contains(pattern)) |> Seq.toList

                for key in keysToRemove do
                    cache.Remove(key) |> ignore

                printfn $"🗑️ CACHE CLEARED: %d{keysToRemove.Length} entries matching '%s{pattern}'"
                return $"Cleared %d{keysToRemove.Length} entries matching '%s{pattern}'."
        }

    [<TarsToolAttribute("cache_stats", "Returns cache statistics. Input: ignored")>]
    let cacheStats (_: string) =
        task {
            let total = cache.Count

            let expired =
                cache.Values |> Seq.filter (fun (_, exp) -> DateTime.Now >= exp) |> Seq.length

            let valid = total - expired

            return $"Cache Stats: %d{total} total entries, %d{valid} valid, %d{expired} expired."
        }

module MonitoringTools =

    /// Metrics storage
    let private metrics = Dictionary<string, float list>()
    let private componentHealth = Dictionary<string, bool * DateTime>()

    [<TarsToolAttribute("record_metric",
                        "Records a metric value. Input JSON: { \"name\": \"metric_name\", \"value\": 123.45 }")>]
    let recordMetric (args: string) =
        task {
            try
                let doc = System.Text.Json.JsonDocument.Parse(args)
                let root = doc.RootElement

                let name = root.GetProperty("name").GetString()
                let value = root.GetProperty("value").GetDouble()

                if not (metrics.ContainsKey(name)) then
                    metrics.[name] <- []

                // Keep last 100 values
                metrics.[name] <- (value :: metrics.[name]) |> List.truncate 100

                printfn $"📊 METRIC: %s{name} = %.2f{value}"
                return $"Recorded metric '%s{name}' = %.2f{value}"
            with ex ->
                return "record_metric error: " + ex.Message
        }

    [<TarsToolAttribute("get_metrics", "Gets metric statistics. Input: metric name (or 'all')")>]
    let getMetrics (name: string) =
        task {
            let n = name.Trim()

            if n = "all" || String.IsNullOrWhiteSpace(n) then
                if metrics.Count = 0 then
                    return "No metrics recorded yet."
                else
                    let summary =
                        metrics
                        |> Seq.map (fun kvp ->
                            let values = kvp.Value
                            let avg = if values.IsEmpty then 0.0 else List.average values
                            $"  %s{kvp.Key}: avg=%.2f{avg}, count=%d{values.Length}")
                        |> String.concat "\n"

                    return $"All Metrics:\n%s{summary}"
            else if metrics.ContainsKey(n) then
                let values = metrics.[n]
                let avg = List.average values
                let minV = List.min values
                let maxV = List.max values

                return $"Metric '%s{n}': avg=%.2f{avg}, min=%.2f{minV}, max=%.2f{maxV}, count=%d{values.Length}"
            else
                return $"Metric '%s{n}' not found."
        }

    [<TarsToolAttribute("health_check",
                        "Reports or queries component health. Input JSON: { \"component\": \"name\", \"action\": \"set|get\", \"healthy\": true }")>]
    let healthCheck (args: string) =
        task {
            try
                let doc = System.Text.Json.JsonDocument.Parse(args)
                let root = doc.RootElement

                let compName = root.GetProperty("component").GetString()
                let action = root.GetProperty("action").GetString().ToLower()

                match action with
                | "set" ->
                    let healthy = root.GetProperty("healthy").GetBoolean()
                    componentHealth.[compName] <- (healthy, DateTime.Now)

                    let status = if healthy then "🟢 HEALTHY" else "🔴 UNHEALTHY"
                    printfn $"%s{status}: %s{compName}"
                    return $"%s{compName}: %s{status}"

                | "get" ->
                    if componentHealth.ContainsKey(compName) then
                        let (healthy, lastUpdate) = componentHealth.[compName]
                        let status = if healthy then "🟢 HEALTHY" else "🔴 UNHEALTHY"
                        let age = (DateTime.Now - lastUpdate).TotalSeconds
                        return $"%s{compName}: %s{status} (updated %.0f{age}s ago)"
                    else
                        return $"%s{compName}: ⚪ UNKNOWN"

                | "all" ->
                    if componentHealth.Count = 0 then
                        return "No health data recorded."
                    else
                        let summary =
                            componentHealth
                            |> Seq.map (fun kvp ->
                                let (healthy, _) = kvp.Value
                                let icon = if healthy then "🟢" else "🔴"
                                $"  %s{icon} %s{kvp.Key}")
                            |> String.concat "\n"

                        return $"Health Status:\n%s{summary}"

                | _ -> return $"Unknown action: %s{action}. Use set, get, or all."
            with ex ->
                return "health_check error: " + ex.Message
        }

    [<TarsToolAttribute("report_status", "Generates a full system status report. Input: ignored")>]
    let reportStatus (_: string) =
        task {
            let healthyCount =
                componentHealth.Values |> Seq.filter (fun (h, _) -> h) |> Seq.length

            let totalComponents = componentHealth.Count
            let metricCount = metrics.Count

            let report =
                "System Status Report\n"
                + "═════════════════════════════\n"
                + $"Components: %d{healthyCount}/%d{totalComponents} healthy\n"
                + $"Metrics tracked: %d{metricCount}\n"
                + sprintf "Generated: %s" (DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"))

            return report
        }
