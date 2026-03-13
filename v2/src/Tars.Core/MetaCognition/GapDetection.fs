namespace Tars.Core.MetaCognition

open System

/// Pure-logic capability gap detection.
/// Analyzes failure clusters and success patterns to identify where TARS is weak.
module GapDetection =

    /// Extract domain tags from a goal string (simple keyword extraction).
    let extractDomainTags (goal: string) : string list =
        let lower = goal.ToLowerInvariant()
        let keywords =
            [ "code", "code-generation"
              "refactor", "refactoring"
              "test", "testing"
              "debug", "debugging"
              "explain", "explanation"
              "search", "search"
              "analyze", "analysis"
              "plan", "planning"
              "design", "design"
              "optimize", "optimization"
              "document", "documentation"
              "deploy", "deployment"
              "security", "security"
              "performance", "performance"
              "data", "data-processing" ]
        keywords
        |> List.choose (fun (keyword, domain) ->
            if lower.Contains(keyword) then Some domain else None)
        |> fun domains -> if domains.IsEmpty then [ "general" ] else domains

    /// Compute failure rate by domain from success/failure records.
    /// Returns Map<domain, (failures, total, rate)>.
    let failureRateByDomain
        (successes: (string * string list) list)  // (goal, tags)
        (failures: FailureRecord list)
        : Map<string, float * int * int> =

        // Count successes by domain
        let successCounts =
            successes
            |> List.collect (fun (goal, tags) ->
                let domains = if tags.IsEmpty then extractDomainTags goal else tags
                domains |> List.map (fun d -> d, 1))
            |> List.groupBy fst
            |> List.map (fun (domain, items) -> domain, items.Length)
            |> Map.ofList

        // Count failures by domain
        let failureCounts =
            failures
            |> List.collect (fun f ->
                let domains = if f.Tags.IsEmpty then extractDomainTags f.Goal else f.Tags
                domains |> List.map (fun d -> d, 1))
            |> List.groupBy fst
            |> List.map (fun (domain, items) -> domain, items.Length)
            |> Map.ofList

        // Combine
        let allDomains =
            Set.union
                (successCounts |> Map.keys |> Set.ofSeq)
                (failureCounts |> Map.keys |> Set.ofSeq)

        allDomains
        |> Set.toList
        |> List.map (fun domain ->
            let s = successCounts |> Map.tryFind domain |> Option.defaultValue 0
            let f = failureCounts |> Map.tryFind domain |> Option.defaultValue 0
            let total = s + f
            let rate = if total = 0 then 0.0 else float f / float total
            domain, (rate, f, total))
        |> Map.ofList

    /// Suggest a remedy based on the root cause of a cluster.
    let suggestRemedy (rootCause: FailureRootCause) : GapRemedy =
        match rootCause with
        | FailureRootCause.MissingTool toolName ->
            GapRemedy.AcquireTool(toolName, sprintf "Tool '%s' needed but not available" toolName)
        | FailureRootCause.WrongPattern(used, _) ->
            GapRemedy.LearnPattern(sprintf "Alternative to '%s' for this domain" used)
        | FailureRootCause.KnowledgeGap domain ->
            GapRemedy.IngestKnowledge(domain, [ "documentation"; "examples" ])
        | FailureRootCause.InsufficientContext _ ->
            GapRemedy.ImprovePrompt("current", "Add more context gathering steps")
        | FailureRootCause.ModelLimitation _ ->
            GapRemedy.ComposePatterns([ "chunking"; "summarization" ])
        | FailureRootCause.BadPrompt _ ->
            GapRemedy.ImprovePrompt("current", "Fix prompt formatting/instructions")
        | FailureRootCause.ExternalFailure _ ->
            GapRemedy.LearnPattern "Add retry/fallback handling"
        | FailureRootCause.Unknown _ ->
            GapRemedy.LearnPattern "Investigate and develop new approach"

    /// Detect capability gaps from failure clusters and success data.
    let detectGaps
        (threshold: float)
        (clusters: FailureCluster list)
        (successes: (string * string list) list)
        (failures: FailureRecord list)
        : CapabilityGap list =

        let rates = failureRateByDomain successes failures

        rates
        |> Map.toList
        |> List.choose (fun (domain, (rate, failCount, total)) ->
            if rate >= threshold && total >= 2 then
                // Find clusters related to this domain
                let relatedClusters =
                    clusters
                    |> List.filter (fun c ->
                        c.Members |> List.exists (fun f ->
                            let tags = if f.Tags.IsEmpty then extractDomainTags f.Goal else f.Tags
                            tags |> List.contains domain))
                    |> List.map (fun c -> c.ClusterId)

                let primaryCause =
                    clusters
                    |> List.filter (fun c -> relatedClusters |> List.contains c.ClusterId)
                    |> List.tryHead
                    |> Option.map (fun c -> c.RootCause)
                    |> Option.defaultValue (FailureRootCause.Unknown domain)

                Some
                    { GapId = sprintf "gap-%s" domain
                      Domain = domain
                      Description = sprintf "%.0f%% failure rate in %s (%d/%d)" (rate * 100.0) domain failCount total
                      FailureRate = rate
                      SampleSize = total
                      RelatedClusters = relatedClusters
                      SuggestedRemedy = suggestRemedy primaryCause
                      DetectedAt = DateTime.UtcNow
                      Confidence = min 1.0 (float total / 10.0) * rate }
            else
                None)

    /// Rank gaps by severity: failure rate * sample size * recency.
    let rankGaps (gaps: CapabilityGap list) : CapabilityGap list =
        gaps
        |> List.sortByDescending (fun g ->
            g.FailureRate * float g.SampleSize * g.Confidence)
