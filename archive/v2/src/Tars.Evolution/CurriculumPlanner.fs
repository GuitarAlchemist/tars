namespace Tars.Evolution

open System
open System.Text.Json
open System.Threading.Tasks
open Tars.Llm
open Tars.Core.MetaCognition

/// Generates targeted curriculum tasks from detected capability gaps.
/// Uses LLM to create meaningful exercises, or falls back to template-based generation.
module CurriculumPlanner =

    /// Generate targeted tasks from capability gaps without LLM (template-based).
    let generateTasksFromTemplates (gaps: CapabilityGap list) (maxTasks: int) : TargetedTask list =
        gaps
        |> List.truncate maxTasks
        |> List.mapi (fun i gap ->
            let description, difficulty =
                match gap.SuggestedRemedy with
                | GapRemedy.LearnPattern desc ->
                    sprintf "Practice %s: solve a %s task using a new pattern" gap.Domain desc, 3
                | GapRemedy.AcquireTool(toolName, purpose) ->
                    sprintf "Use tool '%s' for %s in a %s task" toolName purpose gap.Domain, 2
                | GapRemedy.IngestKnowledge(domain, _) ->
                    sprintf "Research %s fundamentals and solve a basic %s problem" domain domain, 2
                | GapRemedy.ImprovePrompt(_, suggestion) ->
                    sprintf "Solve a %s task with improved prompting: %s" gap.Domain suggestion, 3
                | GapRemedy.ComposePatterns patterns ->
                    sprintf "Solve a %s task by composing: %s" gap.Domain (String.Join(", ", patterns)), 4

            { TaskId = sprintf "targeted-%s-%03d" gap.Domain (i + 1)
              GapId = gap.GapId
              Description = description
              Difficulty = difficulty
              ExpectedOutcome = sprintf "Successfully complete a %s task with improved approach" gap.Domain
              ValidationCriteria = Some (sprintf "Task in domain '%s' completes without errors" gap.Domain)
              Priority = gap.FailureRate * gap.Confidence })

    /// Generate targeted tasks using LLM for richer, more specific exercises.
    let generateTasksWithLlm
        (llm: ILlmService)
        (gaps: CapabilityGap list)
        (maxTasks: int)
        : Task<TargetedTask list> =
        task {
            let gapDescriptions =
                gaps
                |> List.truncate maxTasks
                |> List.mapi (fun i g ->
                    sprintf "%d. Domain: %s (%.0f%% failure rate, %d samples)\n   Root cause: %s\n   Suggested remedy: %A"
                        (i + 1) g.Domain (g.FailureRate * 100.0) g.SampleSize g.Description g.SuggestedRemedy)
                |> String.concat "\n"

            let jsonExample = """[{"gap_domain": "...", "description": "...", "difficulty": 1-5, "expected_outcome": "...", "validation": "..."}]"""
            let prompt =
                sprintf """You are a curriculum designer for a self-improving AI reasoning system.
The system has identified these capability gaps:

%s

For each gap, generate a targeted training task. Each task should:
1. Directly exercise the weak capability
2. Start simple and build confidence
3. Have clear success criteria

Output a JSON array matching this schema: %s""" gapDescriptions jsonExample

            let req =
                { ModelHint = Some "reasoning"
                  Model = None
                  SystemPrompt = Some "You are a curriculum designer. Output JSON ONLY."
                  MaxTokens = Some 2000
                  Temperature = Some 0.3
                  Stop = []
                  Messages = [ { Role = Role.User; Content = prompt } ]
                  Tools = []
                  ToolChoice = None
                  ResponseFormat = None
                  Stream = false
                  JsonMode = false
                  Seed = None
                  ContextWindow = None }

            try
                let! resp = llm.CompleteAsync req
                let text = resp.Text.Trim()

                // Extract JSON array
                let firstBracket = text.IndexOf('[')
                let lastBracket = text.LastIndexOf(']')

                if firstBracket >= 0 && lastBracket > firstBracket then
                    let json = text.Substring(firstBracket, lastBracket - firstBracket + 1)
                    let doc = JsonDocument.Parse(json)
                    let tasks =
                        doc.RootElement.EnumerateArray()
                        |> Seq.mapi (fun i elem ->
                            let domain =
                                let mutable p = JsonElement()
                                if elem.TryGetProperty("gap_domain", &p) then p.GetString()
                                else "general"
                            let gap = gaps |> List.tryFind (fun g -> g.Domain = domain)
                            let gapId = gap |> Option.map (fun g -> g.GapId) |> Option.defaultValue "unknown"

                            { TaskId = sprintf "llm-targeted-%s-%03d" domain (i + 1)
                              GapId = gapId
                              Description =
                                  let mutable p = JsonElement()
                                  if elem.TryGetProperty("description", &p) then p.GetString()
                                  else sprintf "Practice %s skills" domain
                              Difficulty =
                                  let mutable p = JsonElement()
                                  if elem.TryGetProperty("difficulty", &p) then p.GetInt32()
                                  else 3
                              ExpectedOutcome =
                                  let mutable p = JsonElement()
                                  if elem.TryGetProperty("expected_outcome", &p) then p.GetString()
                                  else "Task completes successfully"
                              ValidationCriteria =
                                  let mutable p = JsonElement()
                                  if elem.TryGetProperty("validation", &p) then Some(p.GetString())
                                  else None
                              Priority =
                                  gap
                                  |> Option.map (fun g -> g.FailureRate * g.Confidence)
                                  |> Option.defaultValue 0.5 })
                        |> Seq.toList

                    return tasks
                else
                    return generateTasksFromTemplates gaps maxTasks
            with _ ->
                return generateTasksFromTemplates gaps maxTasks
        }

    /// Convert TargetedTask to the existing Problem type for retroaction loop compatibility.
    let toProblem (task: TargetedTask) : Problem =
        { Id = ProblemId task.TaskId
          Source = Custom "meta-cognitive"
          Title = sprintf "[Gap: %s] %s" task.GapId (task.Description.Substring(0, min 60 task.Description.Length))
          Description = task.Description
          Difficulty =
              match task.Difficulty with
              | d when d <= 1 -> Beginner
              | d when d <= 3 -> Intermediate
              | _ -> Advanced
          Tags = [ "meta-cognitive"; "gap-filling"; task.GapId ]
          ValidationCriteria = task.ValidationCriteria
          ReferenceSolution = None }

    /// Merge targeted tasks with existing curriculum, prioritizing gap-filling tasks.
    let mergeCurriculum
        (existing: Problem list)
        (targeted: TargetedTask list)
        : Problem list =
        let targetedProblems = targeted |> List.map toProblem
        // Interleave: targeted first (high priority), then existing
        targetedProblems @ existing
