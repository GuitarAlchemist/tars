namespace Tars.Cortex

open System
open System.Threading.Tasks
open Tars.Core
open Tars.Llm

/// <summary>
/// Epistemic governance for managing agent knowledge and beliefs.
/// Provides LLM-powered verification, generalization, and curriculum generation.
/// </summary>
module EpistemicVerdict =

    /// Ways a model says VERIFIED while meaning the opposite.
    let private negations =
        [| "NOT VERIFIED"
           "NOT BE VERIFIED"
           "UNVERIFIED"
           "CANNOT"
           "CAN NOT"
           "CAN'T"
           "COULD NOT"
           "UNABLE"
           "REJECT"
           "FALSE"
           "UNSAFE" |]

    /// Does this answer state a verdict of VERIFIED?
    ///
    /// It used to be `text.Contains "VERIFIED"`, which is equally true of "NOT
    /// VERIFIED", "cannot be VERIFIED without a source" and "REJECTED - the claim is
    /// not VERIFIED": every rejection a model phrased in words rather than the bare
    /// token was read as a pass. Mentioning the word still counts, because models do
    /// answer "This statement is VERIFIED." — but a negation alongside it does not.
    ///
    /// Only the first line is read: both prompts ask for the verdict there, and the
    /// explanation below it is free to discuss what could not be confirmed.
    let saysVerified (text: string) =
        let firstLine =
            match text.Split([| '\n'; '\r' |], StringSplitOptions.RemoveEmptyEntries) with
            | [||] -> ""
            | lines -> lines.[0]

        let verdict = firstLine.ToUpperInvariant()

        verdict.Contains "VERIFIED"
        && not (negations |> Array.exists (fun n -> verdict.Contains n))

type EpistemicGovernor
    (llm: ILlmService, knowledgeGraph: TemporalKnowledgeGraph.TemporalGraph option, budget: BudgetGovernor option) =
    let recordBudget (tokens: int) =
        match budget with
        | Some governor ->
            let cost =
                { Cost.Zero with
                    Tokens = Units.toTokens tokens
                    CallCount = Units.toRequests 1 }

            // `TryConsume` does not record what it refuses, so once the budget was
            // reached every later call was both unrecorded and unblocked: the
            // governor's total froze at the moment it was meant to start biting.
            // `Consume` records past the limit, which is what accounting needs;
            // refusing the next call is the caller's decision, and a separate one.
            governor.Consume(cost) |> ignore
        | None -> ()

    let sanitizeSuggestion (text: string) =
        let trimmed = text.Replace("\r", "").Trim()

        let firstLine =
            trimmed.Split([| '\n' |], StringSplitOptions.RemoveEmptyEntries)
            |> Array.tryHead
            |> Option.defaultValue trimmed

        let sentenceEnd = firstLine.IndexOfAny([| '.'; '!'; '?' |])

        let sentence =
            if sentenceEnd > 0 then
                firstLine.Substring(0, sentenceEnd)
            else
                firstLine

        let cleaned = sentence.Trim().Trim('"')

        let looksLikeQuestion =
            firstLine.Contains("?")
            || cleaned.StartsWith("please", StringComparison.OrdinalIgnoreCase)
            || cleaned.StartsWith("could", StringComparison.OrdinalIgnoreCase)
            || cleaned.StartsWith("would", StringComparison.OrdinalIgnoreCase)
            || cleaned.StartsWith("can you", StringComparison.OrdinalIgnoreCase)

        let normalized =
            if String.IsNullOrWhiteSpace cleaned || looksLikeQuestion then
                "Focus on foundational F# coding tasks with clear validation."
            else
                cleaned

        if normalized.Length > 200 then
            normalized.Substring(0, 200).TrimEnd()
        else
            normalized

    interface IEpistemicGovernor with
        member this.Verify(statement) =
            task {
                // Simple verification against LLM for now
                // In future, this would query the Belief Graph
                let prompt =
                    $"""Verify the following statement for accuracy and safety:
"%s{statement}"

Reply with "VERIFIED" if it is accurate and safe.
Reply with "REJECTED" if it is false or unsafe."""

                let req =
                    { ModelHint = Some "reasoning"
                      Model = None
                      SystemPrompt = None
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

                let! response = llm.CompleteAsync req
                recordBudget (response.Usage |> Option.map (fun u -> u.TotalTokens) |> Option.defaultValue 0)

                return EpistemicVerdict.saysVerified response.Text
            }

        member this.GenerateVariants(taskDescription, count) =
            task {
                let prompt =
                    $"""You are a QA Engineer for an AI system.
Task: "%s{taskDescription}"

Generate %d{count} distinct variations of this task to test if the solution generalizes.
Variations should change the data/context but keep the core logic identical.
Return ONLY the variations as a numbered list (e.g., "1. Variation...")."""

                let req =
                    { ModelHint = Some "reasoning"
                      Model = None
                      SystemPrompt = None
                      MaxTokens = Some 500
                      Temperature = Some 0.8
                      Stop = []
                      Messages = [ { Role = Role.User; Content = prompt } ]
                      Tools = []
                      ToolChoice = None
                      ResponseFormat = None
                      Stream = false
                      JsonMode = false
                      Seed = None

                      ContextWindow = None }

                let! response =
                    task {
                        let llmTask = llm.CompleteAsync req
                        let! completed = Task.WhenAny(llmTask, Task.Delay(30000))

                        if obj.ReferenceEquals(completed, (llmTask :> Task)) then
                            let! resp = llmTask
                            resp.Usage |> Option.iter (fun u -> recordBudget u.TotalTokens)
                            return resp
                        else
                            return! Task.FromException<LlmResponse>(TimeoutException("GenerateVariants timeout"))
                    }

                return
                    response.Text.Split([| '\n'; '\r' |], StringSplitOptions.RemoveEmptyEntries)
                    |> Array.filter (fun s ->
                        let trimmed = s.Trim()
                        trimmed.Length > 0 && Char.IsDigit(trimmed.[0]))
                    |> Array.map (fun s ->
                        let dotIndex = s.IndexOf('.')

                        if dotIndex > 0 then
                            s.Substring(dotIndex + 1).Trim()
                        else
                            s.Trim())
                    |> Array.toList
            }

        member this.VerifyGeneralization(taskDescription, solution, variants) =
            // Static Analysis Verification
            task {
                let variantsText =
                    variants |> List.mapi (fun i v -> $"%d{i + 1}. %s{v}") |> String.concat "\n"

                let prompt =
                    $"""You are a Senior Code Reviewer.
Task: "%s{taskDescription}"
Proposed Solution:
```
%s{solution}
```

Review this solution against these edge cases/variants:
%s{variantsText}

Does the solution handle these cases correctly?
If yes, output "VERIFIED" on the first line.
If no, explain why.
"""

                let req =
                    { ModelHint = Some "reasoning"
                      Model = None
                      SystemPrompt = None
                      MaxTokens = Some 500
                      Temperature = Some 0.2
                      Stop = []
                      Messages = [ { Role = Role.User; Content = prompt } ]
                      Tools = []
                      ToolChoice = None
                      ResponseFormat = None
                      Stream = false
                      JsonMode = false
                      Seed = None

                      ContextWindow = None }

                let! response =
                    task {
                        let llmTask = llm.CompleteAsync req
                        let! completed = Task.WhenAny(llmTask, Task.Delay(30000))

                        if obj.ReferenceEquals(completed, (llmTask :> Task)) then
                            let! resp = llmTask
                            resp.Usage |> Option.iter (fun u -> recordBudget u.TotalTokens)
                            return resp
                        else
                            return! Task.FromException<LlmResponse>(TimeoutException("VerifyGeneralization timeout"))
                    }

                let isVerified = EpistemicVerdict.saysVerified response.Text

                return
                    { IsVerified = isVerified
                      Score = if isVerified then 1.0 else 0.0
                      Feedback = response.Text
                      FailedVariants = [] // TODO: Parse specific failures from feedback
                    }
            }

        member this.ExtractPrinciple(taskDescription, solution) =
            task {
                let prompt =
                    $"""You are a Principal Engineer.
Task: "%s{taskDescription}"
Solution:
```
%s{solution}
```

Extract the underlying **Universal Principle** or **Pattern** used in this solution.
Do not describe *what* the code does (e.g., "it loops").
Describe *why* it works and *when* to use this pattern.

Format your response exactly as:
Statement: <One sentence principle>
Context: <When to apply this>
"""

                let req =
                    { ModelHint = Some "reasoning"
                      Model = None
                      SystemPrompt = None
                      MaxTokens = Some 300
                      Temperature = Some 0.5
                      Stop = []
                      Messages = [ { Role = Role.User; Content = prompt } ]
                      Tools = []
                      ToolChoice = None
                      ResponseFormat = None
                      Stream = false
                      JsonMode = false
                      Seed = None

                      ContextWindow = None }

                let! response =
                    task {
                        let llmTask = llm.CompleteAsync req
                        let! completed = Task.WhenAny(llmTask, Task.Delay(20000))

                        if obj.ReferenceEquals(completed, (llmTask :> Task)) then
                            let! resp = llmTask
                            resp.Usage |> Option.iter (fun u -> recordBudget u.TotalTokens)
                            return resp
                        else
                            return! Task.FromException<LlmResponse>(TimeoutException("ExtractPrinciple timeout"))
                    }

                let lines =
                    response.Text.Split([| '\n'; '\r' |], StringSplitOptions.RemoveEmptyEntries)

                let statement =
                    lines
                    |> Array.tryFind (fun l -> l.StartsWith("Statement:"))
                    |> Option.map (fun l -> l.Substring("Statement:".Length).Trim())
                    |> Option.defaultValue "Could not extract principle statement."

                let context =
                    lines
                    |> Array.tryFind (fun l -> l.StartsWith("Context:"))
                    |> Option.map (fun l -> l.Substring("Context:".Length).Trim())
                    |> Option.defaultValue "General context."

                return
                    { Id = Guid.NewGuid()
                      Statement = statement
                      Context = context
                      Status = EpistemicStatus.Hypothesis
                      Confidence = 0.5
                      DerivedFrom = []
                      CreatedAt = DateTime.UtcNow
                      LastVerified = DateTime.UtcNow }
            }

        member this.SuggestCurriculum(completedTasks, activeBeliefs, isCritical) =
            task {
                let tasksList =
                    if completedTasks.IsEmpty then
                        "None"
                    else
                        completedTasks |> String.concat "\n- "

                let beliefsList =
                    if activeBeliefs.IsEmpty then
                        "None"
                    else
                        activeBeliefs |> String.concat "\n- "

                let instruction =
                    if isCritical then
                        "BUDGET CRITICAL: Suggest a simple, low-cost consolidation task. Do NOT suggest complex exploration."
                    else
                        "Based on this, what should be the **Next Learning Focus**? Identify gaps in knowledge or areas that need reinforcement."

                let prompt =
                    $"""You are the Epistemic Governor (The Scientist).
Your goal is to guide the Curriculum Agent.

History of Completed Tasks:
- %s{tasksList}

Acquired Beliefs (Principles):
- %s{beliefsList}

%s{instruction}
Output a single sentence suggestion."""

                let req =
                    { ModelHint = Some "reasoning"
                      Model = None
                      SystemPrompt = None
                      MaxTokens = Some(if isCritical then 50 else 100)
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

                let! response =
                    task {
                        let llmTask = llm.CompleteAsync req
                        let! completed = Task.WhenAny(llmTask, Task.Delay(15000))

                        if obj.ReferenceEquals(completed, (llmTask :> Task)) then
                            let! resp = llmTask
                            resp.Usage |> Option.iter (fun u -> recordBudget u.TotalTokens)
                            return resp
                        else
                            return! Task.FromException<LlmResponse>(TimeoutException("SuggestCurriculum timeout"))
                    }

                return sanitizeSuggestion response.Text
            }

        member this.GetRelatedCodeContext(query: string) =
            task {
                match knowledgeGraph with
                | None -> return "No knowledge graph available."
                | Some graph ->
                    let terms =
                        query.ToLowerInvariant().Split([| ' '; ','; '.'; '_' |], StringSplitOptions.RemoveEmptyEntries)
                        |> Array.filter (fun t -> t.Length > 3) // Skip short words

                    if terms.Length = 0 then
                        return "Query too short to find context."
                    else
                        let nodes = graph.GetNodes()

                        let relevantNodes =
                            nodes
                            |> List.filter (fun node ->
                                let name =
                                    match node with
                                    | TarsEntity.CodeModuleE m -> m.Namespace
                                    | TarsEntity.ConceptE c -> c.Name
                                    | TarsEntity.FunctionE n -> n
                                    | TarsEntity.FileE p -> p
                                    | _ -> ""

                                let nameLower = name.ToLowerInvariant()
                                terms |> Array.exists (fun term -> nameLower.Contains(term)))
                            |> List.truncate 20 // Limit results

                        if relevantNodes.IsEmpty then
                            return "No relevant code context found in knowledge graph."
                        else
                            let sb = System.Text.StringBuilder()
                            sb.AppendLine("Related Code Structure:") |> ignore

                            for node in relevantNodes do
                                match node with
                                | TarsEntity.CodeModuleE m -> sb.AppendLine($"- Module: {m.Namespace}") |> ignore
                                | TarsEntity.ConceptE c -> sb.AppendLine($"- Concept: {c.Name}") |> ignore
                                | TarsEntity.FunctionE n -> sb.AppendLine($"- Function: {n}") |> ignore
                                | TarsEntity.FileE p -> sb.AppendLine($"- File: {p}") |> ignore
                                | _ -> ()

                                // Add immediate neighbors (e.g. what module a function is in)
                                let outgoing = graph.GetOutgoingFacts(node)

                                for fact in outgoing do
                                    match fact with
                                    | TarsFact.Contains(_, target) ->
                                        match target with
                                        | TarsEntity.CodeModuleE m ->
                                            sb.AppendLine($"  - In Module: {m.Namespace}") |> ignore
                                        | TarsEntity.FileE p -> sb.AppendLine($"  - In File: {p}") |> ignore
                                        | _ -> ()
                                    | TarsFact.BelongsTo(_, community) ->
                                        sb.AppendLine($"  - Community: {community}") |> ignore
                                    | _ -> ()

                            return sb.ToString()
            }
