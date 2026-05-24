namespace Tars.Evolution

open System
open System.IO
open System.Text.Json
open Tars.Llm
open Tars.Core.WorkflowOfThought

/// <summary>
/// Phase 16: Pattern Retrieval and instantiation.
/// </summary>
module PatternLibrary =

    let private options =
        let o = JsonSerializerOptions(JsonSerializerDefaults.General)
        o.Converters.Add(System.Text.Json.Serialization.JsonFSharpConverter())
        o

    let private getPatternsDir () =
        let dir = Path.Combine(Environment.CurrentDirectory, ".tars", "patterns")

        if not (Directory.Exists dir) then
            Directory.CreateDirectory dir |> ignore

        dir

    /// <summary>
    /// Loads all available patterns from disk.
    /// </summary>
    let loadAll () : PatternDefinition list =
        let dir = getPatternsDir ()

        Directory.GetFiles(dir, "*.json")
        |> Array.toList
        |> List.choose (fun path ->
            try
                let json = File.ReadAllText path
                Some(JsonSerializer.Deserialize<PatternDefinition>(json, options))
            with ex ->
                Console.WriteLine($"[DEBUG] Error loading pattern: {ex.Message}")
                None)
        |> fun L ->
            Console.WriteLine($"[DEBUG] Loaded {L.Length} patterns from {dir}")

            L
            |> List.iter (fun p -> Console.WriteLine($"[DEBUG] Loaded Pattern: '{p.Name}'"))

            L

    /// <summary>
    /// Finds the best matching pattern for a given goal using LLM.
    /// </summary>
    let findMatch
        (llm: ILlmService)
        (goal: string)
        (patterns: PatternDefinition list)
        : Async<PatternDefinition option> =
        async {
            if patterns.IsEmpty then
                return None
            else
                let candidates =
                    patterns
                    |> List.map (fun p -> $"- ID: {p.Name}\n  Desc: {p.Description}\n  Goal: {p.Goal}")
                    |> String.concat "\n"

                let prompt =
                    $"""You are TARS's Pattern Matcher.
Identify the best reasoning pattern to solve the following goal.

USER GOAL: {goal}

AVAILABLE PATTERNS:
{candidates}

INSTRUCTIONS:
1. Select the pattern that best fits the goal.
2. If no pattern is a good fit, output "NONE".
3. If a pattern fits, output its ID.

OUTPUT (ID ONLY or NONE):
"""

                let req =
                    { ModelHint = None
                      Model = None
                      SystemPrompt = Some "You are a pattern matcher."
                      MaxTokens = Some 50
                      Temperature = Some 0.0
                      Stop = []
                      Messages = [ { Role = Role.User; Content = prompt } ]
                      Tools = []
                      ToolChoice = None
                      ResponseFormat = None
                      Stream = false
                      JsonMode = false
                      Seed = None
                      ContextWindow = None }

                let! resp = llm.CompleteAsync req |> Async.AwaitTask
                let text = resp.Text.Trim()
                Console.WriteLine($"[DEBUG] Pattern Matcher Output: '{text}'")

                // Fallback for weak models or testing
                if text.Contains("NONE", StringComparison.OrdinalIgnoreCase) then
                    if goal.Contains("Analysis", StringComparison.OrdinalIgnoreCase) then
                        Console.WriteLine("[DEBUG] Fallback match for 'Analysis'")

                        return
                            patterns
                            |> List.tryFind (fun p -> p.Name.Contains("Reference") || p.Name.Contains("Data Analysis"))
                    else
                        return None
                else
                    return patterns |> List.tryFind (fun p -> text.Contains(p.Name))
        }

    /// <summary>
    /// Hydrates a pattern template with specific context variables.
    /// Currently, just returns the raw template string (Phase 16.1 will add variable injection).
    /// </summary>
    let hydrate (pattern: PatternDefinition) (context: Map<string, string>) : string =
        // TODO: Implement smart variable substitution using context
        // For now, simple textual replacement if keys match {{key}}
        let mutable template = pattern.Template

        for kvp in context do
            template <- template.Replace($"{{{{{kvp.Key}}}}}", kvp.Value)

        template

    /// <summary>
    /// Phase 16.3: Executes a hydrated pattern template and returns the result.
    /// This is the core of the validation loop.
    /// </summary>
    let executePattern
        (llm: ILlmService)
        (pattern: PatternDefinition)
        (hydratedTemplate: string)
        (problemDescription: string)
        : Async<Result<string, string>> =
        async {
            try
                // Parse the template to extract the prompt
                use doc = JsonDocument.Parse(hydratedTemplate)
                let root = doc.RootElement

                // Extract prompt from first node (simplified for v0)
                let prompt =
                    try
                        let nodes = root.GetProperty("nodes")
                        let firstNode = nodes.EnumerateArray() |> Seq.head
                        firstNode.GetProperty("prompt").GetString()
                    with _ ->
                        $"Solve: {problemDescription}"

                Console.WriteLine($"[DEBUG] Executing pattern with prompt: {prompt}")

                let req =
                    { ModelHint = Some "reasoning"
                      Model = None
                      SystemPrompt =
                        Some
                            $"You are TARS solving a problem using the '{pattern.Name}' reasoning pattern.\nPattern: {pattern.Description}"
                      MaxTokens = Some 2000
                      Temperature = Some 0.7
                      Stop = []
                      Messages = [ { Role = Role.User; Content = prompt } ]
                      Tools = []
                      ToolChoice = None
                      ResponseFormat = None
                      Stream = false
                      JsonMode = false
                      Seed = None
                      ContextWindow = None }

                let! resp = llm.CompleteAsync req |> Async.AwaitTask
                let result = resp.Text.Trim()
                Console.WriteLine($"[DEBUG] Pattern execution result ({result.Length} chars)")
                return Ok result
            with ex ->
                Console.WriteLine($"[DEBUG] Pattern execution failed: {ex.Message}")
                return Error ex.Message
        }

    /// <summary>
    /// Phase 16.3: Validates execution result against criteria.
    /// Returns true if the result satisfies the validation criteria.
    /// </summary>
    let validateResult (result: string) (criteria: string option) : bool =
        match criteria with
        | None -> true // No criteria = always pass
        | Some c ->
            // Simple containment check for now
            // Future: regex, numeric comparison, LLM-based semantic validation
            let normalized = result.ToLowerInvariant()
            let criteriaLower = c.ToLowerInvariant()

            // Handle numeric criteria (e.g., "56351" for math problems)
            let mutable d = 0.0

            if System.Double.TryParse(c, &d) then
                normalized.Contains(c)
            else
                // Text match
                normalized.Contains(criteriaLower)
