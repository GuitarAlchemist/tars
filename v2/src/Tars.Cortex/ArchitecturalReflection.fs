/// <summary>
/// Architectural Self-Reflection System
/// =====================================
/// Enables TARS to reason about its own architecture using:
/// - CodebaseRAG: Search and understand its own code
/// - LLM Analysis: Identify patterns and anti-patterns
/// - Self-Improvement: Generate actionable improvement tasks
/// 
/// This is the "consciousness" layer of TARS - the ability to
/// understand, question, and improve its own implementation.
/// </summary>
namespace Tars.Cortex

open System
open System.IO
open System.Threading.Tasks
open Tars.Core
open Tars.Llm
open Tars.Llm.LlmService

/// Architectural Self-Reflection for TARS
module ArchitecturalReflection =

    // =========================================================================
    // Types
    // =========================================================================

    /// An architectural insight discovered through reflection
    type ArchitecturalInsight = {
        Id: string
        Category: InsightCategory
        Title: string
        Description: string
        AffectedFiles: string list
        Severity: InsightSeverity
        Confidence: float
        SuggestedAction: string option
        Evidence: string list
        DiscoveredAt: DateTime
    }
    
    and InsightCategory =
        | DuplicatedLogic      // Same logic in multiple places
        | MissingAbstraction   // Pattern that should be extracted
        | CouplingIssue        // Tight coupling between modules
        | InconsistentPattern  // Different approaches to same problem
        | TechnicalDebt        // Known shortcuts or TODOs
        | ArchitecturalDrift   // Deviation from design principles
        | OptimizationTarget   // Performance improvement opportunity
        | MissingCapability    // Gap in the system
    
    and InsightSeverity =
        | Critical  // Blocks evolution, must fix
        | High      // Significant impact on maintainability
        | Medium    // Should address in next iteration
        | Low       // Nice to have improvement
        | Info      // Informational only

    /// A reflection query about architecture
    type ReflectionQuery = {
        Question: string
        Focus: string list
        Context: string option
    }

    /// Result of architectural reflection
    type ReflectionResult = {
        Query: ReflectionQuery
        Insights: ArchitecturalInsight list
        Summary: string
        RecommendedActions: string list
        Duration: TimeSpan
    }

    /// Configuration for reflection
    type ReflectionConfig = {
        MaxInsights: int
        MinConfidence: float
        FocusAreas: string list
        Quiet: bool
    }

    let defaultConfig = {
        MaxInsights = 10
        MinConfidence = 0.5
        FocusAreas = []
        Quiet = false
    }

    // =========================================================================
    // Standard Reflection Queries
    // =========================================================================

    /// Pre-defined queries for common architectural concerns
    let standardQueries = [
        { Question = "What patterns are duplicated across multiple modules that could be consolidated?"
          Focus = ["*.fs"]
          Context = Some "Look for similar function signatures, repeated code blocks, or common patterns" }
        
        { Question = "Which modules have the highest coupling and could benefit from better abstraction?"
          Focus = ["Tars.Core"; "Tars.Cortex"]
          Context = Some "Analyze imports, type dependencies, and function call patterns" }
        
        { Question = "What interfaces or abstractions are missing that would improve extensibility?"
          Focus = ["*.fs"]
          Context = Some "Look for concrete types used where interfaces would be better" }
        
        { Question = "What tools are missing that would help TARS improve itself?"
          Focus = ["Tars.Tools"]
          Context = Some "Consider refactoring, testing, documentation, and code analysis tools" }
        
        { Question = "How can the evolution loop be made more effective?"
          Focus = ["Evolve.fs"; "Engine.fs"; "ResearchEnhancedCurriculum.fs"]
          Context = Some "Look for bottlenecks, missing feedback loops, or underutilized capabilities" }
        
        { Question = "How well is knowledge being captured and reused across runs?"
          Focus = ["Knowledge"; "Memory"]
          Context = Some "Analyze knowledge graph, ledger, and vector store usage" }
        
        { Question = "Where are error handling and resilience patterns inconsistent?"
          Focus = ["*.fs"]
          Context = Some "Look for try/catch without proper handling, missing retries, or silent failures" }
        
        { Question = "What operations are potentially slow and could be optimized?"
          Focus = ["*.fs"]
          Context = Some "Look for synchronous operations, repeated computations, or slow patterns" }

        { Question = "Where are the primary nondeterministic hotspots causing low-confidence tool calls?"
          Focus = ["Tars.Graph"; "Tars.Tools"]
          Context = Some "Analyze complex tool interactions and graph transition logic for stability improvement" }
    ]

    // =========================================================================
    // JSON Parsing Helpers
    // =========================================================================

    let private extractJson (response: string) : string option =
        // Strategy 1: Look for JSON code blocks
        let codeBlockPattern = @"```(?:json)?\s*([\s\S]*?)\s*```"
        let m = System.Text.RegularExpressions.Regex.Match(response, codeBlockPattern)
        if m.Success then Some(m.Groups.[1].Value.Trim())
        else
            // Strategy 2: Look for the balanced boundaries [ ... ]
            let startIdx = response.IndexOf('[')
            let endIdx = response.LastIndexOf(']')
            if startIdx >= 0 && endIdx > startIdx then
                Some(response.Substring(startIdx, endIdx - startIdx + 1))
            else
                // Strategy 3: Look for { ... }
                let startIdx = response.IndexOf('{')
                let endIdx = response.LastIndexOf('}')
                if startIdx >= 0 && endIdx > startIdx then
                    Some(response.Substring(startIdx, endIdx - startIdx + 1))
                else None

    /// Parse insights from LLM response
    let private parseInsights (quiet: bool) (response: string) : ArchitecturalInsight list =
        try
            match extractJson response with
            | None -> []
            | Some json ->
                use doc = System.Text.Json.JsonDocument.Parse(json)
                let root = doc.RootElement
                
                let targetArray =
                    if root.ValueKind = System.Text.Json.JsonValueKind.Array then Some root
                    else
                        let mutable p = Unchecked.defaultof<System.Text.Json.JsonElement>
                        if root.ValueKind = System.Text.Json.JsonValueKind.Object then
                            if root.TryGetProperty("insights", &p) && p.ValueKind = System.Text.Json.JsonValueKind.Array then Some p
                            elif root.TryGetProperty("suggestions", &p) && p.ValueKind = System.Text.Json.JsonValueKind.Array then Some p
                            else
                                root.EnumerateObject()
                                |> Seq.tryFind (fun prop -> prop.Value.ValueKind = System.Text.Json.JsonValueKind.Array)
                                |> Option.map (fun prop -> prop.Value)
                        else None

                match targetArray with
                | Some arr ->
                    [ for elem in arr.EnumerateArray() do
                        let getString (prop: string) = 
                            let mutable p = Unchecked.defaultof<System.Text.Json.JsonElement>
                            if elem.TryGetProperty(prop, &p) then 
                                match p.ValueKind with
                                | System.Text.Json.JsonValueKind.String -> p.GetString()
                                | System.Text.Json.JsonValueKind.Number -> p.GetRawText()
                                | _ -> ""
                            else ""
                        
                        let getFloat (prop: string) =
                            let mutable p = Unchecked.defaultof<System.Text.Json.JsonElement>
                            if elem.TryGetProperty(prop, &p) then 
                                try 
                                    match p.ValueKind with
                                    | System.Text.Json.JsonValueKind.Number -> p.GetDouble()
                                    | System.Text.Json.JsonValueKind.String -> 
                                        match Double.TryParse(p.GetString()) with
                                        | true, v -> v
                                        | _ -> 0.5
                                    | _ -> 0.5
                                with _ -> 0.5
                            else 0.5
                        
                        let getArray (prop: string) =
                            let mutable p = Unchecked.defaultof<System.Text.Json.JsonElement>
                            if elem.TryGetProperty(prop, &p) && p.ValueKind = System.Text.Json.JsonValueKind.Array then
                                [ for item in p.EnumerateArray() do
                                    match item.ValueKind with
                                    | System.Text.Json.JsonValueKind.String -> yield item.GetString()
                                    | _ -> () ]
                            else []
                        
                        let category = 
                            match (getString "category").ToLowerInvariant() with
                            | s when s.Contains("duplicat") -> DuplicatedLogic
                            | s when s.Contains("abstract") -> MissingAbstraction
                            | s when s.Contains("coupl") -> CouplingIssue
                            | s when s.Contains("inconsist") -> InconsistentPattern
                            | s when s.Contains("debt") -> TechnicalDebt
                            | s when s.Contains("drift") -> ArchitecturalDrift
                            | s when s.Contains("optim") -> OptimizationTarget
                            | s when s.Contains("missing") -> MissingCapability
                            | _ -> TechnicalDebt
                        
                        let severity =
                            match (getString "severity").ToLowerInvariant() with
                            | "critical" -> Critical
                            | "high" -> High
                            | "medium" -> Medium
                            | "low" -> Low
                            | _ -> Info
                        
                        yield {
                            Id = Guid.NewGuid().ToString("N").[..8]
                            Category = category
                            Title = getString "title"
                            Description = getString "description"
                            AffectedFiles = getArray "affectedFiles"
                            Severity = severity
                            Confidence = getFloat "confidence"
                            SuggestedAction = 
                                let action = getString "suggestedAction"
                                if String.IsNullOrWhiteSpace(action) then None else Some action
                            Evidence = getArray "evidence"
                            DiscoveredAt = DateTime.UtcNow
                        }
                    ]
                | None -> []
        with ex ->
            if quiet |> not then
                printfn "   [LOG] Warning: Failed to parse insights: %s" ex.Message
            []

    // =========================================================================
    // Core Reflection Engine
    // =========================================================================

    /// Reflects on architecture using CodebaseRAG and LLM
    let reflectOnArchitecture 
        (llm: ILlmService) 
        (codebaseIndex: CodebaseRAG.CodebaseIndex) 
        (query: ReflectionQuery)
        (config: ReflectionConfig) =
        task {
            let startTime = DateTime.UtcNow
            
            // 1. Search codebase for relevant code
            let! searchResults = codebaseIndex.SearchAsync(query.Question, 10)
            
            let codeContext = 
                searchResults
                |> List.map (fun r -> 
                    sprintf "=== %s (lines %d-%d) ===\n%s" 
                        (Path.GetFileName r.Chunk.FilePath)
                        r.Chunk.StartLine
                        r.Chunk.EndLine
                        (if r.Chunk.Content.Length > 800 
                         then r.Chunk.Content.Substring(0, 800) + "\n..."
                         else r.Chunk.Content))
                |> String.concat "\n\n"
            
            // 2. Build reflection prompt
            let systemPrompt = """You are TARS's self-reflection system - a metacognitive layer that analyzes TARS's own codebase.

Your role is to:
1. Identify architectural patterns and anti-patterns
2. Find opportunities for improvement
3. Detect inconsistencies and technical debt
4. Suggest concrete, actionable improvements

Be specific and cite actual code when possible. Format insights as JSON."""

            let userPrompt = 
                sprintf """ARCHITECTURAL REFLECTION QUERY:
%s

ADDITIONAL CONTEXT:
%s

RELEVANT CODE FROM TARS CODEBASE:
%s

---

Analyze the code and provide architectural insights as a JSON array. Each insight should have: category, title, description, affectedFiles, severity, confidence (0-1), suggestedAction, evidence."""
                    query.Question
                    (query.Context |> Option.defaultValue "None")
                    codeContext
            
            // 3. Query LLM for insights
            let request = {
                LlmRequest.Default with
                    SystemPrompt = Some systemPrompt
                    Messages = [{ Role = Role.User; Content = userPrompt }]
                    Temperature = Some 0.3
            }
            
            let! response = llm.CompleteAsync(request)
            
            // 4. Parse insights from response
            let insights = parseInsights config.Quiet response.Text
            
            if insights.IsEmpty && config.Quiet |> not then
                printfn "   [LOG] No insights parsed from response. Length: %d" response.Text.Length
                if response.Text.Length < 1000 then
                    printfn "   [LOG] Response: %s" response.Text
            let filteredInsights = 
                insights 
                |> List.filter (fun i -> i.Confidence >= config.MinConfidence)
                |> List.truncate config.MaxInsights
            
            // 6. Generate summary
            let summary = 
                if filteredInsights.IsEmpty then
                    "No significant architectural issues found in the analyzed code."
                else
                    sprintf "Found %d architectural insights across %d categories. %d require immediate attention."
                        filteredInsights.Length
                        (filteredInsights |> List.map (fun i -> i.Category) |> List.distinct |> List.length)
                        (filteredInsights |> List.filter (fun i -> i.Severity = Critical || i.Severity = High) |> List.length)
            
            // 7. Extract recommended actions
            let actions = 
                filteredInsights
                |> List.choose (fun i -> i.SuggestedAction)
                |> List.distinct
            
            return {
                Query = query
                Insights = filteredInsights
                Summary = summary
                RecommendedActions = actions
                Duration = DateTime.UtcNow - startTime
            }
        }

    /// An improvement task generated from architectural insight
    type ImprovementTask = {
        Id: Guid
        Title: string
        Description: string
        Priority: int  // 1 = Critical, 5 = Info
        AffectedFiles: string list
        SourceInsight: string
    }

    /// Generates self-improvement tasks from insights
    let generateImprovementTasks (insights: ArchitecturalInsight list) : ImprovementTask list =
        insights
        |> List.filter (fun i -> i.SuggestedAction.IsSome)
        |> List.map (fun insight ->
            let priority = 
                match insight.Severity with
                | Critical -> 1
                | High -> 2
                | Medium -> 3
                | Low -> 4
                | Info -> 5
            
            let description = 
                sprintf "[Self-Improvement] %s\n\nCONTEXT: %s\n\nACTION: %s" 
                    insight.Title
                    insight.Description
                    (insight.SuggestedAction |> Option.defaultValue "No specific action")
            
            {
                Id = Guid.NewGuid()
                Title = insight.Title
                Description = description
                Priority = priority
                AffectedFiles = insight.AffectedFiles
                SourceInsight = insight.Id
            }
        )

    // =========================================================================
    // Full Reflection Workflow
    // =========================================================================

    /// Runs a complete architectural reflection session
    let runReflectionSession
        (llm: ILlmService)
        (codebaseIndex: CodebaseRAG.CodebaseIndex)
        (customQueries: ReflectionQuery list option)
        (config: ReflectionConfig) =
        task {
            let queries = customQueries |> Option.defaultValue standardQueries
            let mutable allInsights: ArchitecturalInsight list = []
            let mutable results: ReflectionResult list = []
            
            printfn "🔍 Starting Architectural Self-Reflection..."
            printfn "   Analyzing %d reflection queries..." queries.Length
            
            let! resultsRaw = 
                queries
                |> List.map (fun query -> 
                    task {
                        let! res = reflectOnArchitecture llm codebaseIndex query config
                        if not config.Quiet then
                             printfn "   ✓ Completed: %s" (if query.Question.Length > 40 then query.Question.[..40] + "..." else query.Question)
                        return res
                    })
                |> Task.WhenAll
            
            let results = Array.toList resultsRaw
            let allInsights = results |> List.collect (fun r -> r.Insights)
            
            // Deduplicate insights
            let uniqueInsights = 
                allInsights
                |> List.distinctBy (fun i -> i.Title, i.Category)
                |> List.sortBy (fun i -> 
                    match i.Severity with
                    | Critical -> 0 | High -> 1 | Medium -> 2 | Low -> 3 | Info -> 4)
            
            printfn "   ✓ Found %d unique insights" uniqueInsights.Length
            
            // Generate improvement tasks
            let improvementTasks = generateImprovementTasks uniqueInsights
            
            printfn "   ✓ Generated %d self-improvement tasks" improvementTasks.Length
            
            return {|
                Insights = uniqueInsights
                Results = results
                ImprovementTasks = improvementTasks
                Summary = sprintf "Architectural reflection complete. Found %d insights across %d queries, generated %d improvement tasks."
                            uniqueInsights.Length
                            queries.Length
                            improvementTasks.Length
            |}
        }
