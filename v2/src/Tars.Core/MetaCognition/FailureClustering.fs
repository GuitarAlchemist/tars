namespace Tars.Core.MetaCognition

open System

/// Pure-logic failure clustering: groups failures by similarity and classifies root causes.
/// No LLM dependency — uses string similarity and keyword heuristics.
module FailureClustering =

    // =====================================================================
    // Similarity metrics
    // =====================================================================

    /// Jaccard similarity between two string lists (tags, goal tokens, etc.)
    let jaccardSimilarity (a: string list) (b: string list) : float =
        if a.IsEmpty && b.IsEmpty then 1.0
        else
            let setA = Set.ofList a
            let setB = Set.ofList b
            let intersection = Set.intersect setA setB |> Set.count |> float
            let union = Set.union setA setB |> Set.count |> float
            if union = 0.0 then 0.0 else intersection / union

    /// Simple token-based similarity for error messages.
    /// Splits on whitespace, lowercases, and computes Jaccard.
    let errorSimilarity (e1: string) (e2: string) : float =
        let tokenize (s: string) =
            s.ToLowerInvariant().Split([| ' '; '\t'; '\n'; '\r'; ':'; '('; ')'; '['; ']' |], StringSplitOptions.RemoveEmptyEntries)
            |> Array.toList
        jaccardSimilarity (tokenize e1) (tokenize e2)

    /// Composite similarity between two failure records.
    /// Weights: error message 50%, goal 30%, pattern 20%.
    let failureSimilarity (f1: FailureRecord) (f2: FailureRecord) : float =
        let errSim = errorSimilarity f1.ErrorMessage f2.ErrorMessage
        let goalSim = errorSimilarity f1.Goal f2.Goal
        let patternSim = if f1.PatternUsed = f2.PatternUsed then 1.0 else 0.0
        0.5 * errSim + 0.3 * goalSim + 0.2 * patternSim

    // =====================================================================
    // Clustering (single-linkage agglomerative)
    // =====================================================================

    /// Single-linkage agglomerative clustering.
    /// Merges clusters when any pair of members has similarity >= threshold.
    let cluster (threshold: float) (failures: FailureRecord list) : FailureRecord list list =
        if failures.IsEmpty then []
        else
            // Start with each failure in its own cluster
            let mutable clusters = failures |> List.map (fun f -> [ f ])

            let rec mergePass () =
                let mutable merged = false
                let mutable newClusters = []
                let mutable used = Set.empty

                for i in 0 .. clusters.Length - 1 do
                    if not (Set.contains i used) then
                        let mutable current = clusters.[i]
                        for j in (i + 1) .. clusters.Length - 1 do
                            if not (Set.contains j used) then
                                // Single-linkage: merge if ANY pair exceeds threshold
                                let shouldMerge =
                                    current |> List.exists (fun a ->
                                        clusters.[j] |> List.exists (fun b ->
                                            failureSimilarity a b >= threshold))
                                if shouldMerge then
                                    current <- current @ clusters.[j]
                                    used <- Set.add j used
                                    merged <- true
                        newClusters <- current :: newClusters
                        used <- Set.add i used

                clusters <- List.rev newClusters
                if merged then mergePass ()

            mergePass ()
            clusters

    // =====================================================================
    // Root cause classification (heuristic)
    // =====================================================================

    /// Classify root cause from error message patterns.
    /// Uses keyword matching — no LLM needed.
    let classifyRootCause (members: FailureRecord list) : FailureRootCause =
        let allErrors =
            members
            |> List.map (fun f -> f.ErrorMessage.ToLowerInvariant())
            |> String.concat " "

        let contains (s: string) = allErrors.Contains(s)

        if contains "tool not found" || contains "tool_call failed" || contains "no tool" || contains "missing tool" then
            let toolName =
                members
                |> List.tryPick (fun f ->
                    let lower = f.ErrorMessage.ToLowerInvariant()
                    if lower.Contains("tool") then f.FailedAtStep else None)
                |> Option.defaultValue "unknown"
            FailureRootCause.MissingTool toolName
        elif contains "timeout" || contains "rate limit" || contains "connection refused" || contains "503" || contains "429" then
            FailureRootCause.ExternalFailure "service unavailable or rate-limited"
        elif contains "validation failed" || contains "no match" || contains "wrong pattern" || contains "pattern mismatch" then
            let pattern = members |> List.tryHead |> Option.map (fun f -> f.PatternUsed) |> Option.defaultValue "unknown"
            FailureRootCause.WrongPattern(pattern, "try alternative")
        elif contains "not enough context" || contains "missing" || contains "incomplete" || contains "insufficient" then
            FailureRootCause.InsufficientContext "required information not available"
        elif contains "token limit" || contains "context length" || contains "too long" || contains "truncat" then
            FailureRootCause.ModelLimitation "context or token limit exceeded"
        elif contains "prompt" || contains "instruction" || contains "parse error" || contains "malformed" then
            FailureRootCause.BadPrompt "prompt format or instruction issue"
        elif contains "unknown" || contains "domain" || contains "unfamiliar" then
            FailureRootCause.KnowledgeGap "unrecognized domain"
        else
            FailureRootCause.Unknown (members |> List.head |> fun f -> f.ErrorMessage |> fun s -> s.Substring(0, min 100 s.Length))

    // =====================================================================
    // Build clusters
    // =====================================================================

    /// Build FailureCluster records from raw failure records.
    let buildClusters (threshold: float) (failures: FailureRecord list) : FailureCluster list =
        let rawClusters = cluster threshold failures

        rawClusters
        |> List.mapi (fun i members ->
            let sorted = members |> List.sortBy (fun f -> f.Timestamp)
            let representative =
                members
                |> List.sortByDescending (fun f -> f.Score)
                |> List.head

            { ClusterId = sprintf "cluster-%03d" (i + 1)
              Representative = representative
              Members = members
              RootCause = classifyRootCause members
              Frequency = members.Length
              FirstSeen = sorted |> List.head |> fun f -> f.Timestamp
              LastSeen = sorted |> List.last |> fun f -> f.Timestamp
              AffectedGoalPatterns =
                  members
                  |> List.map (fun f -> f.PatternUsed)
                  |> List.distinct })
