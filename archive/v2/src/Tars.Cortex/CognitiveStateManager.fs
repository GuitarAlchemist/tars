namespace Tars.Cortex

open System
open Tars.Cortex.WoTTypes

/// <summary>
/// Service that manages the cognitive state of the agent.
/// Tracks modes, entropy, and stability across executions.
/// </summary>
module CognitiveStateManager =

    let private calculateEntropy (result: WoTResult) : float =
        // Heuristic: Entropy increases with branching and novel triples
        // Normalize to 0.0 - 1.0
        let branchingImpact = Math.Min(result.Metrics.BranchingFactor / 5.0, 1.0)
        let noveltyImpact = Math.Min(float result.TriplesDelta.Length / 10.0, 1.0)
        (branchingImpact * 0.6) + (noveltyImpact * 0.4)

    let private calculateEigenvalue (state: WoTCognitiveState) (result: WoTResult) : float =
        // Eigenvalue tracks stability/success.
        // 1.0 = stable, 0.0 = unstable
        let successFactor = if result.Success then 1.1 else 0.8
        let current = state.Eigenvalue
        // Decay towards success/failure
        Math.Clamp(current * successFactor, 0.1, 1.0)

    let private determineMode (state: WoTCognitiveState) (entropy: float) (eigenvalue: float) (result: WoTResult) : WoTCognitiveMode =
        // Critical overrides everything
        if not result.Success && int result.Errors.Length > 2 then
            WoTCognitiveMode.Critical
        // High entropy implies exploration
        elif entropy > 0.7 then
            WoTCognitiveMode.Exploratory
        // High stability and constraints imply convergence
        elif eigenvalue > 0.8 && (result.Metrics.ConstraintScore |> Option.defaultValue 0.0) > 0.8 then
            WoTCognitiveMode.Convergent
        // Default to keeping current or gently shifting
        else
            state.Mode

    let initialState : WoTCognitiveState =
        { Mode = WoTCognitiveMode.Exploratory
          Eigenvalue = 1.0
          Entropy = 0.5
          BranchingFactor = 1.0
          ActivePattern = None
          WoTRunId = None
          StepCount = 0
          TokenBudget = None
          LastTransition = DateTime.UtcNow
          ConstraintScore = None
          SuccessRate = 1.0 }

    /// <summary>
    /// Update cognitive state based on the result of a WoT execution.
    /// </summary>
    let update (state: WoTCognitiveState) (result: WoTResult) : WoTCognitiveState =
        let entropy = calculateEntropy result
        let eigenvalue = calculateEigenvalue state result
        let newMode = determineMode state entropy eigenvalue result
        
        let transitionTime = 
            if newMode <> state.Mode then DateTime.UtcNow else state.LastTransition

        // Update success rate logic (simple moving average for now)
        let newSuccessRate = (state.SuccessRate * 0.8) + ((if result.Success then 1.0 else 0.0) * 0.2)
        
        // Accumulate budget usage if tracked
        let newBudget = 
            match state.TokenBudget with
            | Some b -> Some (b - result.Metrics.TotalTokens)
            | None -> None

        { state with
            Mode = newMode
            Entropy = entropy
            Eigenvalue = eigenvalue
            BranchingFactor = result.Metrics.BranchingFactor
            WoTRunId = Some result.Trace.RunId
            StepCount = state.StepCount + result.Metrics.TotalSteps
            TokenBudget = newBudget
            LastTransition = transitionTime
            ConstraintScore = result.Metrics.ConstraintScore
            SuccessRate = newSuccessRate }
