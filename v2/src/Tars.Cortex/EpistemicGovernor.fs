namespace Tars.Cortex

open System
open System.Threading.Tasks
open Tars.Core
open Tars.Llm
open Tars.Llm.LlmService

type EpistemicStatus =
    | Hypothesis // Proposed solution, untested generalization
    | VerifiedFact // Passed generalization tests
    | UniversalPrinciple // Abstracted and reused successfully > N times
    | Heuristic // Useful but known to be brittle (flagged)
    | Fallacy // Proven false

type Belief =
    { Id: Guid
      Statement: string
      Context: string // When does this apply?
      Status: EpistemicStatus
      Confidence: float
      DerivedFrom: Guid list // TaskIds
      CreatedAt: DateTime
      LastVerified: DateTime }

type VerificationResult =
    { IsVerified: bool
      Score: float
      Feedback: string
      FailedVariants: string list }

type IEpistemicGovernor =
    abstract member GenerateVariants: taskDescription: string * count: int -> Task<string list>

    abstract member VerifyGeneralization:
        taskDescription: string * solution: string * variants: string list -> Task<VerificationResult>

    abstract member ExtractPrinciple: taskDescription: string * solution: string -> Task<Belief>

    abstract member SuggestCurriculum: completedTasks: string list * activeBeliefs: string list -> Task<string>

type EpistemicGovernor(llm: ILlmService, knowledgeGraph: KnowledgeGraph option) =

    interface IEpistemicGovernor with
        member this.GenerateVariants(taskDescription, count) =
            task {
                let prompt =
                    sprintf
                        """You are a QA Engineer for an AI system.
Task: "%s"

Generate %d distinct variations of this task to test if the solution generalizes.
Variations should change the data/context but keep the core logic identical.
Return ONLY the variations as a numbered list (e.g., "1. Variation...")."""
                        taskDescription
                        count

                let req =
                    { ModelHint = Some "reasoning"
                      MaxTokens = Some 500
                      Temperature = Some 0.8
                      Messages = [ { Role = Role.User; Content = prompt } ] }

                let! response = llm.CompleteAsync req

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
                    variants
                    |> List.mapi (fun i v -> sprintf "%d. %s" (i + 1) v)
                    |> String.concat "\n"

                let prompt =
                    sprintf
                        """You are a Senior Code Reviewer.
Task: "%s"
Proposed Solution:
```
%s
```

Review this solution against these edge cases/variants:
%s

Does the solution handle these cases correctly?
If yes, output "VERIFIED" on the first line.
If no, explain why.
"""
                        taskDescription
                        solution
                        variantsText

                let req =
                    { ModelHint = Some "reasoning"
                      MaxTokens = Some 500
                      Temperature = Some 0.2
                      Messages = [ { Role = Role.User; Content = prompt } ] }

                let! response = llm.CompleteAsync req
                let isVerified = response.Text.Contains("VERIFIED")

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
                    sprintf
                        """You are a Principal Engineer.
Task: "%s"
Solution:
```
%s
```

Extract the underlying **Universal Principle** or **Pattern** used in this solution.
Do not describe *what* the code does (e.g., "it loops").
Describe *why* it works and *when* to use this pattern.

Format your response exactly as:
Statement: <One sentence principle>
Context: <When to apply this>
"""
                        taskDescription
                        solution

                let req =
                    { ModelHint = Some "reasoning"
                      MaxTokens = Some 300
                      Temperature = Some 0.5
                      Messages = [ { Role = Role.User; Content = prompt } ] }

                let! response = llm.CompleteAsync req

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

        member this.SuggestCurriculum(completedTasks, activeBeliefs) =
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

                let prompt =
                    sprintf
                        """You are the Epistemic Governor (The Scientist).
Your goal is to guide the Curriculum Agent to explore new knowledge areas.

History of Completed Tasks:
- %s

Acquired Beliefs (Principles):
- %s

Based on this, what should be the **Next Learning Focus**?
Identify gaps in knowledge or areas that need reinforcement.
Output a single sentence suggestion."""
                        tasksList
                        beliefsList

                let req =
                    { ModelHint = Some "reasoning"
                      MaxTokens = Some 100
                      Temperature = Some 0.7
                      Messages = [ { Role = Role.User; Content = prompt } ] }

                let! response = llm.CompleteAsync req
                return response.Text.Trim()
            }
