namespace Tars.Evolution

open System
open System.IO
open Tars.Cortex

/// <summary>
/// Pattern-library coherence check, run by `tars evolve --self-improvement`.
/// The execute-and-learn cycle that used to live here was never called and was deleted (#246).
/// </summary>
module RetroactionLoop =

    // =========================================================================
    // Configuration
    // =========================================================================

    type RetroactionConfig =
        { /// Maximum patterns in library before pruning
          MaxLibrarySize: int
          /// Minimum diversity score (0-1) before coherence check intervenes
          MinDiversity: float }

    let defaultConfig =
        { MaxLibrarySize = 50
          MinDiversity = 0.3 }

    let private getPatternsDir () =
        let dir = Path.Combine(Environment.CurrentDirectory, ".tars", "patterns")
        if not (Directory.Exists dir) then
            Directory.CreateDirectory dir |> ignore
        dir

    // =========================================================================
    // Coherence Check
    // =========================================================================

    /// Ensures the pattern library maintains diversity and doesn't converge
    /// to a single strategy. Returns warnings and optionally prunes duplicates.
    let coherenceCheck
        (config: RetroactionConfig)
        : string list =
        let patterns = PatternLibrary.loadAll ()
        let mutable warnings = []

        // 1. Check library size
        if patterns.Length > config.MaxLibrarySize then
            warnings <- sprintf "Library size %d exceeds max %d. Consider pruning low-scoring patterns." patterns.Length config.MaxLibrarySize :: warnings

        if patterns.IsEmpty then
            warnings <- "Pattern library is empty. No learning has occurred yet." :: warnings
        else
            // 2. Check diversity: count distinct goals
            let distinctGoals = patterns |> List.map (fun p -> p.Goal) |> List.distinct |> List.length
            let diversity = float distinctGoals / float (max 1 patterns.Length)

            if diversity < config.MinDiversity then
                warnings <- sprintf "Library diversity %.2f is below minimum %.2f. Patterns are converging." diversity config.MinDiversity :: warnings

            // 3. Check for near-duplicate names (possible redundancy)
            let nameClusters =
                patterns
                |> List.groupBy (fun p ->
                    // Strip version suffixes like _v1, _v2
                    let name = p.Name
                    let underscoreV = name.LastIndexOf("_v")
                    if underscoreV > 0 then name.Substring(0, underscoreV) else name)
                |> List.filter (fun (_, group) -> group.Length > 3)

            for (baseName, group) in nameClusters do
                warnings <- sprintf "Pattern '%s' has %d variants. Consider keeping only the top scorer." baseName group.Length :: warnings

            // 4. Check average fitness
            let avgScore = patterns |> List.averageBy (fun p -> p.Score)
            if avgScore < 0.3 then
                warnings <- sprintf "Average pattern fitness %.2f is low. Learning may not be effective." avgScore :: warnings

            // 5. Prune if over limit: remove lowest-scoring patterns
            if patterns.Length > config.MaxLibrarySize then
                let toRemove =
                    patterns
                    |> List.sortBy (fun p -> p.Score)
                    |> List.take (patterns.Length - config.MaxLibrarySize)

                for p in toRemove do
                    let dir = getPatternsDir ()
                    let safeName = p.Name.Replace(" ", "_").Replace("/", "_")
                    let path = Path.Combine(dir, sprintf "%s.json" safeName)
                    if File.Exists path then
                        File.Delete path
                        warnings <- sprintf "Pruned low-scoring pattern '%s' (score=%.2f)" p.Name p.Score :: warnings

        warnings
