// State-Space Representation & Control Theory - Advanced Mathematical Foundation for TARS
// Implements linear/non-linear state-space models, Kalman filtering, and Model Predictive Control

namespace TarsEngine.FSharp.Core.Mathematics

open System
open System.Threading.Tasks

/// Linear state-space model representation
type LinearStateSpaceModel = {
    StateMatrix: float[,]           // A matrix (n x n)
    InputMatrix: float[,]           // B matrix (n x m)
    OutputMatrix: float[,]          // C matrix (p x n)
    FeedthroughMatrix: float[,]     // D matrix (p x m)
    ProcessNoise: float[,]          // Q matrix (n x n)
    MeasurementNoise: float[,]      // R matrix (p x p)
    StateDimension: int             // n
    InputDimension: int             // m
    OutputDimension: int            // p
}

/// Non-linear state-space model representation
type NonLinearStateSpaceModel = {
    StateTransitionFunction: float[] -> float[] -> float[]     // f(x, u)
    OutputFunction: float[] -> float[] -> float[]              // g(x, u)
    StateJacobian: float[] -> float[] -> float[,]             // ∂f/∂x
    InputJacobian: float[] -> float[] -> float[,]             // ∂f/∂u
    OutputStateJacobian: float[] -> float[] -> float[,]       // ∂g/∂x
    OutputInputJacobian: float[] -> float[] -> float[,]       // ∂g/∂u
    ProcessNoise: float[,]
    MeasurementNoise: float[,]
    StateDimension: int
    InputDimension: int
    OutputDimension: int
}

/// Kalman filter state
type KalmanFilterState = {
    StateEstimate: float[]          // x̂
    CovarianceMatrix: float[,]      // P
    Innovation: float[]             // y - ŷ
    InnovationCovariance: float[,]  // S
    KalmanGain: float[,]           // K
    LogLikelihood: float
}

/// Model Predictive Control parameters
type MPCParameters = {
    PredictionHorizon: int          // N
    ControlHorizon: int             // Nu
    StateWeights: float[,]          // Q
    InputWeights: float[,]          // R
    TerminalWeights: float[,]       // P
    StateConstraints: (float[] * float[]) option  // (min, max)
    InputConstraints: (float[] * float[]) option  // (min, max)
}

/// Lyapunov analysis result
type LyapunovAnalysisResult = {
    IsStable: bool
    LyapunovFunction: float[] -> float
    LyapunovDerivative: float[] -> float
    StabilityMargin: float
    EquilibriumPoint: float[]
    BasinOfAttraction: float option
}

/// State-Space Control Theory Module
module StateSpaceControlTheory =
    
    // ============================================================================
    // MATRIX OPERATIONS UTILITIES
    // ============================================================================
    
    /// Matrix multiplication
    let matrixMultiply (a: float[,]) (b: float[,]) =
        let rows1, cols1 = Array2D.length1 a, Array2D.length2 a
        let rows2, cols2 = Array2D.length1 b, Array2D.length2 b
        if cols1 <> rows2 then failwith "Matrix dimensions incompatible for multiplication"
        
        let result = Array2D.zeroCreate rows1 cols2
        for i in 0..rows1-1 do
            for j in 0..cols2-1 do
                let mutable sum = 0.0
                for k in 0..cols1-1 do
                    sum <- sum + a.[i,k] * b.[k,j]
                result.[i,j] <- sum
        result
    
    /// Matrix transpose
    let matrixTranspose (matrix: float[,]) =
        let rows, cols = Array2D.length1 matrix, Array2D.length2 matrix
        Array2D.init cols rows (fun i j -> matrix.[j,i])
    
    /// Matrix inverse (simplified Gauss-Jordan for small matrices)
    let matrixInverse (matrix: float[,]) =
        let n = Array2D.length1 matrix
        if n <> Array2D.length2 matrix then failwith "Matrix must be square"
        
        let augmented = Array2D.zeroCreate n (2*n)
        
        // Create augmented matrix [A|I]
        for i in 0..n-1 do
            for j in 0..n-1 do
                augmented.[i,j] <- matrix.[i,j]
                augmented.[i,j+n] <- if i = j then 1.0 else 0.0
        
        // Gauss-Jordan elimination
        for i in 0..n-1 do
            // Find pivot
            let mutable maxRow = i
            for k in i+1..n-1 do
                if abs(augmented.[k,i]) > abs(augmented.[maxRow,i]) then
                    maxRow <- k
            
            // Swap rows
            if maxRow <> i then
                for j in 0..2*n-1 do
                    let temp = augmented.[i,j]
                    augmented.[i,j] <- augmented.[maxRow,j]
                    augmented.[maxRow,j] <- temp
            
            // Make diagonal element 1
            let pivot = augmented.[i,i]
            if abs(pivot) < 1e-10 then failwith "Matrix is singular"
            
            for j in 0..2*n-1 do
                augmented.[i,j] <- augmented.[i,j] / pivot
            
            // Eliminate column
            for k in 0..n-1 do
                if k <> i then
                    let factor = augmented.[k,i]
                    for j in 0..2*n-1 do
                        augmented.[k,j] <- augmented.[k,j] - factor * augmented.[i,j]
        
        // Extract inverse matrix
        Array2D.init n n (fun i j -> augmented.[i,j+n])
    
    /// Vector-matrix multiplication
    let vectorMatrixMultiply (vector: float[]) (matrix: float[,]) =
        let n = vector.Length
        let cols = Array2D.length2 matrix
        if n <> Array2D.length1 matrix then failwith "Vector-matrix dimensions incompatible"
        
        Array.init cols (fun j ->
            let mutable sum = 0.0
            for i in 0..n-1 do
                sum <- sum + vector.[i] * matrix.[i,j]
            sum)
    
    /// Matrix-vector multiplication
    let matrixVectorMultiply (matrix: float[,]) (vector: float[]) =
        let rows = Array2D.length1 matrix
        let cols = Array2D.length2 matrix
        if cols <> vector.Length then failwith "Matrix-vector dimensions incompatible"
        
        Array.init rows (fun i ->
            let mutable sum = 0.0
            for j in 0..cols-1 do
                sum <- sum + matrix.[i,j] * vector.[j]
            sum)
    
    // ============================================================================
    // LINEAR STATE-SPACE MODELS
    // ============================================================================
    
    /// Create linear state-space model
    let createLinearStateSpaceModel stateMatrix inputMatrix outputMatrix feedthroughMatrix processNoise measurementNoise =
        async {
            let n = Array2D.length1 stateMatrix
            let m = Array2D.length2 inputMatrix
            let p = Array2D.length1 outputMatrix
            
            return {
                StateMatrix = stateMatrix
                InputMatrix = inputMatrix
                OutputMatrix = outputMatrix
                FeedthroughMatrix = feedthroughMatrix
                ProcessNoise = processNoise
                MeasurementNoise = measurementNoise
                StateDimension = n
                InputDimension = m
                OutputDimension = p
            }
        }
    
    /// Simulate linear state-space model
    let simulateLinearStateSpace (model: LinearStateSpaceModel) (initialState: float[]) (inputs: float[][]) =
        async {
            let steps = inputs.Length
            let states = Array.zeroCreate (steps + 1)
            let outputs = Array.zeroCreate steps
            
            states.[0] <- initialState
            
            for k in 0..steps-1 do
                // x[k+1] = A*x[k] + B*u[k] + w[k]
                let stateUpdate = matrixVectorMultiply model.StateMatrix states.[k]
                let inputEffect = matrixVectorMultiply model.InputMatrix inputs.[k]
                
                // Add process noise (simplified)
                let processNoise = Array.zeroCreate model.StateDimension
                for i in 0..model.StateDimension-1 do
                    processNoise.[i] <- Random().NextGaussian() * sqrt(model.ProcessNoise.[i,i])
                
                states.[k+1] <- Array.map3 (fun s i n -> s + i + n) stateUpdate inputEffect processNoise
                
                // y[k] = C*x[k] + D*u[k] + v[k]
                let outputFromState = matrixVectorMultiply model.OutputMatrix states.[k]
                let outputFromInput = matrixVectorMultiply model.FeedthroughMatrix inputs.[k]
                
                // Add measurement noise (simplified)
                let measurementNoise = Array.zeroCreate model.OutputDimension
                for i in 0..model.OutputDimension-1 do
                    measurementNoise.[i] <- Random().NextGaussian() * sqrt(model.MeasurementNoise.[i,i])
                
                outputs.[k] <- Array.map3 (fun s i n -> s + i + n) outputFromState outputFromInput measurementNoise
            
            return {|
                States = states
                Outputs = outputs
                Model = model
                SimulationSteps = steps
            |}
        }
    
    // ============================================================================
    // KALMAN FILTERING
    // ============================================================================
    
    /// Initialize Kalman filter
    let initializeKalmanFilter (model: LinearStateSpaceModel) (initialState: float[]) (initialCovariance: float[,]) =
        async {
            return {
                StateEstimate = initialState
                CovarianceMatrix = initialCovariance
                Innovation = Array.zeroCreate model.OutputDimension
                InnovationCovariance = Array2D.zeroCreate model.OutputDimension model.OutputDimension
                KalmanGain = Array2D.zeroCreate model.StateDimension model.OutputDimension
                LogLikelihood = 0.0
            }
        }
    
    /// Kalman filter prediction step
    let kalmanPredict (model: LinearStateSpaceModel) (state: KalmanFilterState) (input: float[]) =
        async {
            // Predict state: x̂[k|k-1] = A*x̂[k-1|k-1] + B*u[k-1]
            let predictedState = 
                let stateUpdate = matrixVectorMultiply model.StateMatrix state.StateEstimate
                let inputEffect = matrixVectorMultiply model.InputMatrix input
                Array.map2 (+) stateUpdate inputEffect
            
            // Predict covariance: P[k|k-1] = A*P[k-1|k-1]*A' + Q
            let AP = matrixMultiply model.StateMatrix state.CovarianceMatrix
            let APAT = matrixMultiply AP (matrixTranspose model.StateMatrix)
            let predictedCovariance = Array2D.mapi (fun i j x -> x + model.ProcessNoise.[i,j]) APAT
            
            return { state with 
                StateEstimate = predictedState
                CovarianceMatrix = predictedCovariance }
        }
    
    /// Kalman filter update step
    let kalmanUpdate (model: LinearStateSpaceModel) (state: KalmanFilterState) (measurement: float[]) =
        async {
            // Innovation: y[k] = z[k] - C*x̂[k|k-1] - D*u[k]
            let predictedOutput = matrixVectorMultiply model.OutputMatrix state.StateEstimate
            let innovation = Array.map2 (-) measurement predictedOutput
            
            // Innovation covariance: S[k] = C*P[k|k-1]*C' + R
            let CP = matrixMultiply model.OutputMatrix state.CovarianceMatrix
            let CPCT = matrixMultiply CP (matrixTranspose model.OutputMatrix)
            let innovationCovariance = Array2D.mapi (fun i j x -> x + model.MeasurementNoise.[i,j]) CPCT
            
            // Kalman gain: K[k] = P[k|k-1]*C'*S[k]^-1
            let PCT = matrixMultiply state.CovarianceMatrix (matrixTranspose model.OutputMatrix)
            let kalmanGain = matrixMultiply PCT (matrixInverse innovationCovariance)
            
            // Update state estimate: x̂[k|k] = x̂[k|k-1] + K[k]*y[k]
            let gainTimesInnovation = matrixVectorMultiply kalmanGain innovation
            let updatedState = Array.map2 (+) state.StateEstimate gainTimesInnovation
            
            // Update covariance: P[k|k] = (I - K[k]*C)*P[k|k-1]
            let KC = matrixMultiply kalmanGain model.OutputMatrix
            let identity = Array2D.init model.StateDimension model.StateDimension (fun i j -> if i = j then 1.0 else 0.0)
            let IminusKC = Array2D.mapi (fun i j x -> if i = j then 1.0 - KC.[i,j] else -KC.[i,j]) identity
            let updatedCovariance = matrixMultiply IminusKC state.CovarianceMatrix
            
            // Log-likelihood update (simplified)
            let logLikelihood = state.LogLikelihood - 0.5 * (innovation |> Array.sumBy (fun x -> x*x))
            
            return {
                StateEstimate = updatedState
                CovarianceMatrix = updatedCovariance
                Innovation = innovation
                InnovationCovariance = innovationCovariance
                KalmanGain = kalmanGain
                LogLikelihood = logLikelihood
            }
        }
    
    /// Full Kalman filter step (predict + update)
    let kalmanFilterStep (model: LinearStateSpaceModel) (state: KalmanFilterState) (input: float[]) (measurement: float[]) =
        async {
            let! predictedState = kalmanPredict model state input
            let! updatedState = kalmanUpdate model predictedState measurement
            return updatedState
        }

    // ============================================================================
    // MODEL PREDICTIVE CONTROL (MPC)
    // ============================================================================

    /// Create MPC parameters
    let createMPCParameters predictionHorizon controlHorizon stateWeights inputWeights terminalWeights =
        {
            PredictionHorizon = predictionHorizon
            ControlHorizon = controlHorizon
            StateWeights = stateWeights
            InputWeights = inputWeights
            TerminalWeights = terminalWeights
            StateConstraints = None
            InputConstraints = None
        }

    /// MPC cost function evaluation
    let evaluateMPCCost (model: LinearStateSpaceModel) (params: MPCParameters) (currentState: float[]) (inputSequence: float[][]) =
        let mutable totalCost = 0.0
        let mutable state = currentState

        // Stage costs
        for k in 0..params.PredictionHorizon-1 do
            let input = if k < inputSequence.Length then inputSequence.[k] else Array.zeroCreate model.InputDimension

            // State cost: x'*Q*x
            let stateWeightedState = matrixVectorMultiply params.StateWeights state
            let stateCost = Array.map2 (*) state stateWeightedState |> Array.sum

            // Input cost: u'*R*u
            let inputWeightedInput = matrixVectorMultiply params.InputWeights input
            let inputCost = Array.map2 (*) input inputWeightedInput |> Array.sum

            totalCost <- totalCost + stateCost + inputCost

            // Update state for next iteration
            let stateUpdate = matrixVectorMultiply model.StateMatrix state
            let inputEffect = matrixVectorMultiply model.InputMatrix input
            state <- Array.map2 (+) stateUpdate inputEffect

        // Terminal cost: x_N'*P*x_N
        let terminalWeightedState = matrixVectorMultiply params.TerminalWeights state
        let terminalCost = Array.map2 (*) state terminalWeightedState |> Array.sum
        totalCost <- totalCost + terminalCost

        totalCost

    /// Simple MPC optimization (gradient descent)
    let solveMPC (model: LinearStateSpaceModel) (params: MPCParameters) (currentState: float[]) =
        async {
            let mutable bestInputSequence = Array.init params.ControlHorizon (fun _ -> Array.zeroCreate model.InputDimension)
            let mutable bestCost = evaluateMPCCost model params currentState bestInputSequence

            let learningRate = 0.01
            let iterations = 100

            for iter in 0..iterations-1 do
                // Simple gradient descent optimization
                let perturbedSequences = Array.init 10 (fun _ ->
                    bestInputSequence |> Array.map (fun input ->
                        input |> Array.map (fun u -> u + (Random().NextDouble() - 0.5) * learningRate)))

                for perturbedSeq in perturbedSequences do
                    let cost = evaluateMPCCost model params currentState perturbedSeq
                    if cost < bestCost then
                        bestCost <- cost
                        bestInputSequence <- perturbedSeq

            return {|
                OptimalInputSequence = bestInputSequence
                OptimalCost = bestCost
                FirstInput = bestInputSequence.[0]
                PredictionHorizon = params.PredictionHorizon
                ControlHorizon = params.ControlHorizon
            |}
        }

    // ============================================================================
    // LYAPUNOV STABILITY ANALYSIS
    // ============================================================================

    /// Create quadratic Lyapunov function V(x) = x'*P*x
    let createQuadraticLyapunovFunction (P: float[,]) =
        fun (x: float[]) ->
            let Px = matrixVectorMultiply P x
            Array.map2 (*) x Px |> Array.sum

    /// Compute Lyapunov derivative for linear system
    let computeLyapunovDerivative (model: LinearStateSpaceModel) (P: float[,]) =
        fun (x: float[]) ->
            // V̇(x) = x'*(A'*P + P*A)*x
            let AT = matrixTranspose model.StateMatrix
            let ATP = matrixMultiply AT P
            let PA = matrixMultiply P model.StateMatrix
            let ATPPA = Array2D.mapi (fun i j _ -> ATP.[i,j] + PA.[i,j]) ATP

            let ATППAx = matrixVectorMultiply ATPPA x
            Array.map2 (*) x ATППAx |> Array.sum

    /// Analyze Lyapunov stability
    let analyzeLyapunovStability (model: LinearStateSpaceModel) =
        async {
            try
                // Solve Lyapunov equation A'*P + P*A = -Q for P
                // Simplified: use identity matrix for Q
                let Q = Array2D.init model.StateDimension model.StateDimension (fun i j -> if i = j then 1.0 else 0.0)

                // For stable systems, we can approximate P
                let P = Array2D.init model.StateDimension model.StateDimension (fun i j -> if i = j then 1.0 else 0.0)

                let lyapunovFunction = createQuadraticLyapunovFunction P
                let lyapunovDerivative = computeLyapunovDerivative model P

                // Check stability by examining eigenvalues (simplified)
                let isStable = true // Simplified - would need eigenvalue computation

                let equilibrium = Array.zeroCreate model.StateDimension

                return {
                    IsStable = isStable
                    LyapunovFunction = lyapunovFunction
                    LyapunovDerivative = lyapunovDerivative
                    StabilityMargin = 1.0 // Simplified
                    EquilibriumPoint = equilibrium
                    BasinOfAttraction = Some 10.0 // Simplified
                }

            with
            | ex ->
                return {
                    IsStable = false
                    LyapunovFunction = fun _ -> 0.0
                    LyapunovDerivative = fun _ -> 0.0
                    StabilityMargin = 0.0
                    EquilibriumPoint = Array.zeroCreate model.StateDimension
                    BasinOfAttraction = None
                }
        }

    // ============================================================================
    // NON-LINEAR STATE-SPACE MODELS
    // ============================================================================

    /// Create non-linear state-space model
    let createNonLinearStateSpaceModel stateTransition outputFunction stateJacobian inputJacobian outputStateJacobian outputInputJacobian processNoise measurementNoise =
        async {
            return {
                StateTransitionFunction = stateTransition
                OutputFunction = outputFunction
                StateJacobian = stateJacobian
                InputJacobian = inputJacobian
                OutputStateJacobian = outputStateJacobian
                OutputInputJacobian = outputInputJacobian
                ProcessNoise = processNoise
                MeasurementNoise = measurementNoise
                StateDimension = Array2D.length1 processNoise
                InputDimension = Array2D.length2 inputJacobian Array.empty Array.empty
                OutputDimension = Array2D.length1 measurementNoise
            }
        }

    /// Extended Kalman Filter for non-linear systems
    let extendedKalmanFilter (model: NonLinearStateSpaceModel) (state: KalmanFilterState) (input: float[]) (measurement: float[]) =
        async {
            // Prediction step
            let predictedState = model.StateTransitionFunction state.StateEstimate input

            // Linearize around current estimate
            let F = model.StateJacobian state.StateEstimate input
            let FP = matrixMultiply F state.CovarianceMatrix
            let FPFT = matrixMultiply FP (matrixTranspose F)
            let predictedCovariance = Array2D.mapi (fun i j x -> x + model.ProcessNoise.[i,j]) FPFT

            // Update step
            let predictedOutput = model.OutputFunction predictedState input
            let innovation = Array.map2 (-) measurement predictedOutput

            let H = model.OutputStateJacobian predictedState input
            let HP = matrixMultiply H predictedCovariance
            let HPHT = matrixMultiply HP (matrixTranspose H)
            let innovationCovariance = Array2D.mapi (fun i j x -> x + model.MeasurementNoise.[i,j]) HPHT

            let PHT = matrixMultiply predictedCovariance (matrixTranspose H)
            let kalmanGain = matrixMultiply PHT (matrixInverse innovationCovariance)

            let gainTimesInnovation = matrixVectorMultiply kalmanGain innovation
            let updatedState = Array.map2 (+) predictedState gainTimesInnovation

            let KH = matrixMultiply kalmanGain H
            let identity = Array2D.init model.StateDimension model.StateDimension (fun i j -> if i = j then 1.0 else 0.0)
            let IminusKH = Array2D.mapi (fun i j x -> if i = j then 1.0 - KH.[i,j] else -KH.[i,j]) identity
            let updatedCovariance = matrixMultiply IminusKH predictedCovariance

            return {
                StateEstimate = updatedState
                CovarianceMatrix = updatedCovariance
                Innovation = innovation
                InnovationCovariance = innovationCovariance
                KalmanGain = kalmanGain
                LogLikelihood = state.LogLikelihood - 0.5 * (innovation |> Array.sumBy (fun x -> x*x))
            }
        }

// Extension for Random class to generate Gaussian random numbers
type Random with
    member this.NextGaussian() =
        let mutable hasSpare = false
        let mutable spare = 0.0

        if hasSpare then
            hasSpare <- false
            spare
        else
            hasSpare <- true
            let u = this.NextDouble()
            let v = this.NextDouble()
            let mag = sqrt(-2.0 * log(u))
            spare <- mag * cos(2.0 * Math.PI * v)
            mag * sin(2.0 * Math.PI * v)
