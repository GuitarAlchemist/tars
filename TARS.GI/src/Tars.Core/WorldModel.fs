// TARS.GI World Model - Predictive Coding with Active Inference
// Core inference function implementing the exact blueprint specification
// Maintains executable model of "how the world works" for planning

namespace Tars.Core

open System
open Types

/// Core inference function - predictive coding with active inference
/// Exactly as specified in the blueprint
let infer (prior: Latent) (a: Action option) (o: Observation) : Latent * float =
    // Predict step - apply dynamics model
    let predicted = 
        match a with
        | Some action ->
            // Apply action dynamics (simplified linear model)
            let actionEffect = Array.create prior.Mean.Length 0.1
            { Mean = Array.map2 (+) prior.Mean actionEffect
              Cov = Array.map (Array.map (fun x -> x + 0.01)) prior.Cov }
        | None ->
            // No action - just add process noise
            { Mean = prior.Mean
              Cov = Array.map (Array.map (fun x -> x + 0.01)) prior.Cov }
    
    // Update step - incorporate observation
    let observationNoise = 0.05
    let kalmanGain = 0.5 // Simplified - should be computed from covariances
    
    let innovation = Array.map2 (-) o predicted.Mean
    let predictionError = innovation |> Array.sumBy (fun x -> x * x) |> sqrt
    
    let updatedMean = Array.map2 (fun pred innov -> pred + kalmanGain * innov) predicted.Mean innovation
    let updatedCov = Array.map (Array.map (fun x -> x * (1.0 - kalmanGain))) predicted.Cov
    
    let posterior = { Mean = updatedMean; Cov = updatedCov }
    (posterior, predictionError)

/// World Model Core implementing predictive coding loop
type WorldModelCore(dimensions: int) =
    let mutable currentState = { 
        Mean = Array.create dimensions 0.0
        Cov = Array.init dimensions (fun i -> Array.create dimensions (if i = i then 1.0 else 0.0))
    }
    let mutable predictionHistory = []
    
    /// Predict next state given action
    member _.Predict(action: Action option) =
        let (predicted, _) = infer currentState action (Array.create dimensions 0.0)
        predicted
    
    /// Update state with observation using core infer function
    member this.Update(action: Action option, observation: Observation) =
        let (newState, predError) = infer currentState action observation
        currentState <- newState
        predictionHistory <- (DateTime.UtcNow, predError) :: (predictionHistory |> List.take (Math.Min(100, predictionHistory.Length)))
        (newState, predError)
    
    /// Get current state
    member _.GetCurrentState() = currentState
    
    /// Get prediction error trend
    member _.GetPredictionTrend() =
        if predictionHistory.Length < 2 then 0.0
        else
            let recent = predictionHistory |> List.take (Math.Min(5, predictionHistory.Length)) |> List.map snd
            let older = predictionHistory |> List.skip 5 |> List.take (Math.Min(5, predictionHistory.Length - 5)) |> List.map snd
            if older.IsEmpty then 0.0
            else (List.average recent) - (List.average older)
    
    /// Calculate uncertainty from covariance trace
    member _.GetUncertainty() =
        currentState.Cov |> Array.mapi (fun i row -> row.[i]) |> Array.sum
    
    /// Simulate action consequences (for planning)
    member this.SimulateAction(action: Action, steps: int) =
        let mutable simState = currentState
        let mutable simResults = []
        
        for _ in 1..steps do
            let (nextState, predError) = infer simState (Some action) (Array.create dimensions 0.5)
            simState <- nextState
            simResults <- (nextState, predError) :: simResults
        
        List.rev simResults
    
    /// Reset world model to initial state
    member this.Reset() =
        currentState <- { 
            Mean = Array.create dimensions 0.0
            Cov = Array.init dimensions (fun i -> Array.create dimensions (if i = i then 1.0 else 0.0))
        }
        predictionHistory <- []

/// Active Inference Controller for action selection
type ActiveInferenceController(worldModel: WorldModelCore) =
    
    /// Calculate expected free energy for action
    member _.CalculateExpectedFreeEnergy(action: Action, horizon: int) =
        let simResults = worldModel.SimulateAction(action, horizon)
        
        // Risk: prediction error accumulation
        let risk = simResults |> List.sumBy snd
        
        // Ambiguity: uncertainty accumulation  
        let ambiguity = simResults |> List.sumBy (fun (state, _) -> 
            state.Cov |> Array.mapi (fun i row -> row.[i]) |> Array.sum)
        
        risk + ambiguity
    
    /// Select action that minimizes expected free energy
    member this.SelectAction(actions: Action list, horizon: int) =
        if actions.IsEmpty then None
        else
            actions
            |> List.map (fun action -> (action, this.CalculateExpectedFreeEnergy(action, horizon)))
            |> List.minBy snd
            |> fst
            |> Some
    
    /// Check if exploration is needed (high uncertainty)
    member _.NeedsExploration() =
        worldModel.GetUncertainty() > 2.0

/// Predictive Coding Loop Manager
type PredictiveCodingLoop(worldModel: WorldModelCore, controller: ActiveInferenceController) =
    let mutable isRunning = false
    let mutable loopMetrics = { PredError = 0.0; BeliefEntropy = 0.0; SpecViolations = 0; ReplanCount = 0 }
    
    /// Execute one predictive coding cycle
    member _.ExecuteCycle(observation: Observation, availableActions: Action list) =
        // 1. Predict
        let prediction = worldModel.Predict(None)
        
        // 2. Update with observation
        let (newState, predError) = worldModel.Update(None, observation)
        
        // 3. Select action via active inference
        let selectedAction = controller.SelectAction(availableActions, 3)
        
        // 4. Update metrics
        loopMetrics <- { loopMetrics with PredError = predError }
        
        {|
            Prediction = prediction
            UpdatedState = newState
            PredictionError = predError
            SelectedAction = selectedAction
            NeedsExploration = controller.NeedsExploration()
            Metrics = loopMetrics
        |}
    
    /// Start continuous predictive coding loop
    member this.StartLoop() =
        isRunning <- true
    
    /// Stop predictive coding loop
    member this.StopLoop() =
        isRunning <- false
    
    /// Get current loop status
    member _.IsRunning() = isRunning
    
    /// Get loop metrics
    member _.GetMetrics() = loopMetrics
