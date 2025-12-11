/// UncertaintyGatedPlanner - Plans with confidence thresholds
/// Part of v2.2 Cognitive Patterns
namespace Tars.Cortex

open System
open System.Threading.Tasks
open Tars.Llm
open Tars.Llm.LlmService

/// Confidence level for a plan step
type ConfidenceLevel =
    | High // > 0.8
    | Medium // 0.5 - 0.8
    | Low // 0.2 - 0.5
    | VeryLow // < 0.2

/// A plan step with uncertainty tracking
type UncertainStep =
    { Id: string
      Action: string
      Confidence: float
      Level: ConfidenceLevel
      Alternatives: string list
      NeedsVerification: bool }

/// Result of planning with uncertainty
type PlanResult =
    { Steps: UncertainStep list
      OverallConfidence: float
      RequiresHumanReview: bool
      HighRiskSteps: UncertainStep list }

/// Configuration for uncertainty-gated planning
type PlannerConfig =
    { MinConfidenceToExecute: float
      RequireReviewBelowConfidence: float
      MaxAlternativesToGenerate: int }

    static member Default =
        { MinConfidenceToExecute = 0.3
          RequireReviewBelowConfidence = 0.5
          MaxAlternativesToGenerate = 3 }

/// Uncertainty-Gated Planner - Pattern 5 from research
/// Plans with confidence tracking and gates execution on uncertainty
type UncertaintyGatedPlanner(llm: ILlmService, config: PlannerConfig) =

    let classifyConfidence (score: float) =
        if score > 0.8 then High
        elif score > 0.5 then Medium
        elif score > 0.2 then Low
        else VeryLow

    /// Generate a plan with uncertainty estimates
    member this.PlanAsync(goal: string) : Task<PlanResult> =
        task {
            let prompt =
                $"""You are a planning assistant. Generate a step-by-step plan for the following goal.
For each step, estimate your confidence (0.0 to 1.0) in that step being correct.

Goal: {goal}

Respond in this exact format for each step:
STEP: [step description]
CONFIDENCE: [0.0 to 1.0]
ALTERNATIVES: [comma-separated alternatives if confidence < 0.7, or "none"]

Generate 3-5 steps."""

            let request =
                { ModelHint = Some "reasoning"
                  Model = None
                  SystemPrompt = None
                  MaxTokens = Some 1000
                  Temperature = Some 0.3
                  Stop = []
                  Messages = [ { Role = Role.User; Content = prompt } ]
                  Tools = []
                  ToolChoice = None
                  ResponseFormat = None
                  Stream = false
                  JsonMode = false
                  Seed = None }

            let! response = llm.CompleteAsync(request)
            let steps = this.ParsePlanResponse(response.Text)

            let overallConfidence =
                if List.isEmpty steps then
                    0.0
                else
                    steps |> List.averageBy (fun s -> s.Confidence)

            let highRiskSteps =
                steps |> List.filter (fun s -> s.Confidence < config.MinConfidenceToExecute)

            return
                { Steps = steps
                  OverallConfidence = overallConfidence
                  RequiresHumanReview = overallConfidence < config.RequireReviewBelowConfidence
                  HighRiskSteps = highRiskSteps }
        }

    /// Parse the LLM response into structured steps
    member private this.ParsePlanResponse(response: string) : UncertainStep list =
        let lines = response.Split([| '\n'; '\r' |], StringSplitOptions.RemoveEmptyEntries)
        let mutable steps = []
        let mutable currentStep = None
        let mutable currentConfidence = 0.5
        let mutable currentAlternatives = []
        let mutable stepCount = 0

        for line in lines do
            let trimmed = line.Trim()

            if trimmed.StartsWith("STEP:", StringComparison.OrdinalIgnoreCase) then
                // Save previous step if any
                match currentStep with
                | Some action ->
                    stepCount <- stepCount + 1

                    let step =
                        { Id = $"step_{stepCount}"
                          Action = action
                          Confidence = currentConfidence
                          Level = classifyConfidence currentConfidence
                          Alternatives = currentAlternatives
                          NeedsVerification = currentConfidence < config.MinConfidenceToExecute }

                    steps <- step :: steps
                | None -> ()

                currentStep <- Some(trimmed.Substring(5).Trim())
                currentConfidence <- 0.5
                currentAlternatives <- []

            elif trimmed.StartsWith("CONFIDENCE:", StringComparison.OrdinalIgnoreCase) then
                let confStr = trimmed.Substring(11).Trim()

                match Double.TryParse(confStr) with
                | true, v -> currentConfidence <- v
                | _ -> ()

            elif trimmed.StartsWith("ALTERNATIVES:", StringComparison.OrdinalIgnoreCase) then
                let altStr = trimmed.Substring(13).Trim()

                if altStr.ToLowerInvariant() <> "none" then
                    currentAlternatives <-
                        altStr.Split(',')
                        |> Array.map (fun s -> s.Trim())
                        |> Array.filter (fun s -> s.Length > 0)
                        |> Array.toList

        // Don't forget the last step
        match currentStep with
        | Some action ->
            stepCount <- stepCount + 1

            let step =
                { Id = $"step_{stepCount}"
                  Action = action
                  Confidence = currentConfidence
                  Level = classifyConfidence currentConfidence
                  Alternatives = currentAlternatives
                  NeedsVerification = currentConfidence < config.MinConfidenceToExecute }

            steps <- step :: steps
        | None -> ()

        steps |> List.rev

    /// Check if a plan is ready for execution
    member this.CanExecute(plan: PlanResult) =
        plan.OverallConfidence >= config.MinConfidenceToExecute
        && List.isEmpty plan.HighRiskSteps

module UncertaintyGatedPlanner =
    let createDefault llm =
        UncertaintyGatedPlanner(llm, PlannerConfig.Default)

    let create llm config = UncertaintyGatedPlanner(llm, config)
