// TARS Enhanced F# Closure Factory
// Advanced ML, Control Theory, and Functional Programming Implementations
// Source: .tars/enhanced_closure_factory

module TARS.EnhancedClosureFactory

open System
open System.Threading.Tasks

// ============================================================================
// GRADIENT DESCENT OPTIMIZERS WITH F# COMPUTATIONAL EXPRESSIONS
// ============================================================================

type GradientDescentBuilder() =
    member _.Bind(x, f) = async { let! result = x in return! f result }
    member _.Return(x) = async { return x }
    member _.ReturnFrom(x) = x
    member _.Zero() = async { return () }

let gradientDescent = GradientDescentBuilder()

// Gradient Descent Closure Factory
let createGradientDescentOptimizer learningRate momentum weightDecay =
    fun gradientFunction weights ->
        gradientDescent {
            let! gradient = gradientFunction weights
            let velocity = Array.map2 (fun v g -> momentum * v - learningRate * g) 
                                     (Array.zeroCreate weights.Length) gradient
            let! updatedWeights = async {
                return Array.map3 (fun w v wd -> w + v - weightDecay * w) 
                                  weights velocity (Array.create weights.Length weightDecay)
            }
            return updatedWeights, velocity
        }

// Adam Optimizer with F# Computational Expression
type AdamOptimizerBuilder() =
    member _.Bind(x, f) = async { let! result = x in return! f result }
    member _.Return(x) = async { return x }
    member _.ReturnFrom(x) = x

let adamOptimizer = AdamOptimizerBuilder()

let createAdamOptimizer beta1 beta2 epsilon learningRateSchedule =
    fun gradientFunction weights ->
        let mutable m = Array.zeroCreate weights.Length
        let mutable v = Array.zeroCreate weights.Length
        let mutable t = 0
        
        adamOptimizer {
            t <- t + 1
            let! gradient = gradientFunction weights
            let learningRate = learningRateSchedule t
            
            // Update biased first moment estimate
            m <- Array.map2 (fun mi gi -> beta1 * mi + (1.0 - beta1) * gi) m gradient
            
            // Update biased second raw moment estimate
            v <- Array.map2 (fun vi gi -> beta2 * vi + (1.0 - beta2) * gi * gi) v gradient
            
            // Compute bias-corrected first moment estimate
            let mHat = Array.map (fun mi -> mi / (1.0 - Math.Pow(beta1, float t))) m
            
            // Compute bias-corrected second raw moment estimate
            let vHat = Array.map (fun vi -> vi / (1.0 - Math.Pow(beta2, float t))) v
            
            // Update parameters
            let! updatedWeights = async {
                return Array.map3 (fun wi mhi vhi -> 
                    wi - learningRate * mhi / (Math.Sqrt(vhi) + epsilon)) weights mHat vHat
            }
            
            return updatedWeights, m, v
        }

// ============================================================================
// STATE SPACE REPRESENTATION WITH F# COMPUTATIONAL EXPRESSIONS
// ============================================================================

type StateSpaceBuilder() =
    member _.Bind(x, f) = x |> f
    member _.Return(x) = x
    member _.ReturnFrom(x) = x

let stateSpace = StateSpaceBuilder()

// State Space Representation Closure
let createStateSpaceRepresentation aMatrix bMatrix cMatrix dMatrix =
    fun initialState inputSequence ->
        stateSpace {
            let! states = 
                inputSequence
                |> List.scan (fun state input ->
                    // x[k+1] = A*x[k] + B*u[k]
                    let stateUpdate = Array.map2 (+) 
                        (Array.map (Array.fold (+) 0.0 << Array.map2 (*) state) aMatrix)
                        (Array.map (Array.fold (+) 0.0 << Array.map2 (*) input) bMatrix)
                    stateUpdate
                ) initialState
                
            let! outputs =
                List.map2 (fun state input ->
                    // y[k] = C*x[k] + D*u[k]
                    Array.map2 (+)
                        (Array.map (Array.fold (+) 0.0 << Array.map2 (*) state) cMatrix)
                        (Array.map (Array.fold (+) 0.0 << Array.map2 (*) input) dMatrix)
                ) states inputSequence
                
            return states, outputs
        }

// Observability Analysis Closure
let createObservabilityAnalysis systemMatrices =
    fun order ->
        stateSpace {
            let aMatrix, cMatrix = systemMatrices
            let! observabilityMatrix =
                [0..order-1]
                |> List.fold (fun acc i ->
                    let cAi = Array.map (fun row ->
                        Array.fold (fun innerAcc j ->
                            Array.map2 (+) innerAcc 
                                (Array.map ((*) row.[j]) aMatrix.[j])
                        ) (Array.zeroCreate aMatrix.Length) [0..aMatrix.Length-1]
                    ) cMatrix
                    Array.append acc cAi
                ) cMatrix
                
            let! rank = 
                // Simplified rank calculation (would use proper linear algebra)
                observabilityMatrix
                |> Array.map (Array.filter ((<>) 0.0) >> Array.length)
                |> Array.max
                
            return observabilityMatrix, rank, rank = order
        }

// ============================================================================
// BODE/NYQUIST ANALYSIS WITH F# COMPUTATIONAL EXPRESSIONS
// ============================================================================

type FrequencyAnalysisBuilder() =
    member _.Bind(x, f) = async { let! result = x in return! f result }
    member _.Return(x) = async { return x }
    member _.ReturnFrom(x) = x

let frequencyAnalysis = FrequencyAnalysisBuilder()

// Bode Plot Generator Closure
let createBodePlotGenerator transferFunction frequencyRange plotOptions =
    fun () ->
        frequencyAnalysis {
            let! frequencyPoints = async {
                return [frequencyRange.Min .. frequencyRange.Step .. frequencyRange.Max]
            }
            
            let! magnitudeResponse = async {
                return frequencyPoints
                |> List.map (fun freq ->
                    let s = Complex(0.0, 2.0 * Math.PI * freq)
                    let response = transferFunction s
                    20.0 * Math.Log10(response.Magnitude)
                )
            }
            
            let! phaseResponse = async {
                return frequencyPoints
                |> List.map (fun freq ->
                    let s = Complex(0.0, 2.0 * Math.PI * freq)
                    let response = transferFunction s
                    response.Phase * 180.0 / Math.PI
                )
            }
            
            return {| 
                Frequencies = frequencyPoints
                Magnitude = magnitudeResponse
                Phase = phaseResponse
                PlotOptions = plotOptions
            |}
        }

// Nyquist Stability Analysis Closure
let createNyquistStabilityAnalysis openLoopTF nyquistContour =
    fun () ->
        frequencyAnalysis {
            let! contourPoints = async {
                return nyquistContour
                |> List.map (fun s -> openLoopTF s)
            }
            
            let! encirclements = async {
                // Simplified encirclement calculation
                let crossings = contourPoints
                               |> List.pairwise
                               |> List.filter (fun (p1, p2) -> 
                                   (p1.Real < -1.0 && p2.Real > -1.0) || 
                                   (p1.Real > -1.0 && p2.Real < -1.0))
                               |> List.length
                return crossings / 2
            }
            
            let! stabilityMargins = async {
                let gainMargin = contourPoints
                               |> List.filter (fun p -> abs(p.Imaginary) < 0.001)
                               |> List.map (fun p -> -20.0 * Math.Log10(p.Magnitude))
                               |> List.tryHead
                               |> Option.defaultValue Double.PositiveInfinity
                               
                let phaseMargin = contourPoints
                                |> List.filter (fun p -> abs(p.Magnitude - 1.0) < 0.001)
                                |> List.map (fun p -> 180.0 + p.Phase * 180.0 / Math.PI)
                                |> List.tryHead
                                |> Option.defaultValue 0.0
                                
                return gainMargin, phaseMargin
            }
            
            return {|
                ContourPoints = contourPoints
                Encirclements = encirclements
                StabilityMargins = stabilityMargins
                IsStable = encirclements = 0
            |}
        }

// ============================================================================
// STATE MACHINES WITH F# COMPUTATIONAL EXPRESSIONS
// ============================================================================

type StateMachineBuilder() =
    member _.Bind(x, f) = 
        match x with
        | Some value -> f value
        | None -> None
    member _.Return(x) = Some x
    member _.ReturnFrom(x) = x
    member _.Zero() = None

let stateMachine = StateMachineBuilder()

// Finite State Machine Builder Closure
let createFiniteStateMachineBuilder states transitions initialState finalStates =
    fun input ->
        let mutable currentState = initialState
        let mutable inputIndex = 0
        
        stateMachine {
            let! processedStates = 
                input
                |> List.fold (fun acc symbol ->
                    match acc with
                    | Some stateSequence ->
                        let nextState = transitions
                                      |> List.tryFind (fun (from, sym, _) -> 
                                          from = currentState && sym = symbol)
                                      |> Option.map (fun (_, _, to_) -> to_)
                        
                        match nextState with
                        | Some state ->
                            currentState <- state
                            Some (state :: stateSequence)
                        | None -> None
                    | None -> None
                ) (Some [initialState])
                |> Option.map List.rev
                
            let! isAccepted = 
                if List.contains currentState finalStates then Some true
                else Some false
                
            return {|
                StateSequence = processedStates
                FinalState = currentState
                IsAccepted = isAccepted
                Transitions = transitions |> List.length
            |}
        }

// Hierarchical State Machine Closure
let createHierarchicalStateMachine parentStates childStates transitionGuards =
    fun event ->
        stateMachine {
            let! activeParent = parentStates |> List.tryHead
            let! activeChild = 
                childStates 
                |> List.filter (fun child -> child.Parent = activeParent.Value.Name)
                |> List.tryHead
                
            let! guardResult = 
                transitionGuards
                |> List.tryFind (fun guard -> guard.Event = event)
                |> Option.map (fun guard -> guard.Condition())
                
            let! newState =
                match guardResult with
                | Some true -> 
                    // Transition allowed
                    Some {| 
                        Parent = activeParent.Value
                        Child = activeChild
                        Event = event
                        TransitionAllowed = true
                    |}
                | _ -> 
                    // Transition blocked
                    Some {|
                        Parent = activeParent.Value
                        Child = activeChild
                        Event = event
                        TransitionAllowed = false
                    |}
                    
            return newState
        }

// ============================================================================
// NEURAL NETWORKS WITH F# COMPUTATIONAL EXPRESSIONS
// ============================================================================

type NeuralNetworkBuilder() =
    member _.Bind(x, f) = async { let! result = x in return! f result }
    member _.Return(x) = async { return x }
    member _.ReturnFrom(x) = x
    member _.For(sequence, body) =
        async {
            let results = ResizeArray()
            for item in sequence do
                let! result = body item
                results.Add(result)
            return results |> Seq.toList
        }

let neuralNetwork = NeuralNetworkBuilder()

// Neural Network Builder Closure
let createNeuralNetworkBuilder layerDefinitions activationFunctions weightInitializers =
    fun trainingData ->
        neuralNetwork {
            let! initializedLayers = async {
                return layerDefinitions
                |> List.mapi (fun i layer ->
                    let weights = weightInitializers.[i] layer.InputSize layer.OutputSize
                    let biases = Array.zeroCreate layer.OutputSize
                    {|
                        Weights = weights
                        Biases = biases
                        Activation = activationFunctions.[i]
                        LayerIndex = i
                    |}
                )
            }

            let! forwardPass =
                fun input ->
                    async {
                        return initializedLayers
                        |> List.fold (fun acc layer ->
                            async {
                                let! prevOutput = acc
                                let linearOutput = Array.mapi (fun i _ ->
                                    Array.fold2 (fun sum w x -> sum + w * x) layer.Biases.[i]
                                                layer.Weights.[i] prevOutput
                                ) layer.Weights
                                return Array.map layer.Activation linearOutput
                            }
                        ) (async { return input })
                    }

            let! backwardPass =
                fun target output ->
                    async {
                        // Simplified backpropagation
                        let error = Array.map2 (-) target output
                        let gradients = initializedLayers
                                      |> List.rev
                                      |> List.scan (fun acc layer ->
                                          Array.map (fun e -> e * (1.0 - tanh(e) * tanh(e))) acc
                                      ) error
                        return gradients |> List.rev
                    }

            return {|
                Layers = initializedLayers
                ForwardPass = forwardPass
                BackwardPass = backwardPass
                TrainingData = trainingData |> List.length
            |}
        }

// ============================================================================
// SIGNAL PROCESSING WITH F# COMPUTATIONAL EXPRESSIONS
// ============================================================================

type SignalProcessingBuilder() =
    member _.Bind(x, f) = x |> f
    member _.Return(x) = x
    member _.ReturnFrom(x) = x
    member _.For(sequence, body) = sequence |> List.map body

let signalProcessing = SignalProcessingBuilder()

// Fast Fourier Transform Closure
let createFastFourierTransform signalData windowFunction zeroPadding =
    fun () ->
        signalProcessing {
            let! windowedSignal =
                signalData
                |> Array.mapi (fun i sample -> sample * windowFunction i)

            let! paddedSignal =
                if zeroPadding > 0 then
                    Array.append windowedSignal (Array.zeroCreate zeroPadding)
                else windowedSignal

            let! fftResult =
                // Simplified FFT implementation (would use proper FFT algorithm)
                let n = paddedSignal.Length
                [0..n-1]
                |> List.map (fun k ->
                    paddedSignal
                    |> Array.mapi (fun n sample ->
                        let angle = -2.0 * Math.PI * float k * float n / float paddedSignal.Length
                        Complex(sample * cos(angle), sample * sin(angle))
                    )
                    |> Array.fold (+) Complex.Zero
                )
                |> List.toArray

            let! magnitudeSpectrum =
                fftResult |> Array.map (fun c -> c.Magnitude)

            let! phaseSpectrum =
                fftResult |> Array.map (fun c -> c.Phase)

            return {|
                FFTResult = fftResult
                MagnitudeSpectrum = magnitudeSpectrum
                PhaseSpectrum = phaseSpectrum
                FrequencyResolution = 1.0 / float paddedSignal.Length
            |}
        }

// Digital Filter Designer Closure
let createDigitalFilterDesigner filterType cutoffFrequencies filterOrder windowMethod =
    fun samplingRate ->
        signalProcessing {
            let! normalizedCutoffs =
                cutoffFrequencies
                |> Array.map (fun fc -> 2.0 * fc / samplingRate)

            let! filterCoefficients =
                match filterType with
                | "lowpass" ->
                    // Simplified FIR filter design
                    [0..filterOrder]
                    |> List.map (fun n ->
                        let nf = float n
                        if n = filterOrder / 2 then normalizedCutoffs.[0]
                        else
                            let arg = Math.PI * (nf - float filterOrder / 2.0) * normalizedCutoffs.[0]
                            sin(arg) / (Math.PI * (nf - float filterOrder / 2.0))
                    )
                    |> List.toArray
                | "highpass" ->
                    // High-pass filter implementation
                    [0..filterOrder]
                    |> List.map (fun n ->
                        let nf = float n
                        if n = filterOrder / 2 then 1.0 - normalizedCutoffs.[0]
                        else
                            let arg = Math.PI * (nf - float filterOrder / 2.0)
                            let lpf = sin(arg * normalizedCutoffs.[0]) / (Math.PI * (nf - float filterOrder / 2.0))
                            if n = filterOrder / 2 then 1.0 - lpf else -lpf
                    )
                    |> List.toArray
                | _ -> Array.zeroCreate (filterOrder + 1)

            let! windowedCoefficients =
                filterCoefficients
                |> Array.mapi (fun i coeff ->
                    match windowMethod with
                    | "hamming" ->
                        coeff * (0.54 - 0.46 * cos(2.0 * Math.PI * float i / float filterOrder))
                    | "hanning" ->
                        coeff * (0.5 - 0.5 * cos(2.0 * Math.PI * float i / float filterOrder))
                    | _ -> coeff
                )

            return {|
                FilterCoefficients = windowedCoefficients
                FilterType = filterType
                CutoffFrequencies = cutoffFrequencies
                FilterOrder = filterOrder
                SamplingRate = samplingRate
            |}
        }

// ============================================================================
// GENETIC ALGORITHM WITH F# COMPUTATIONAL EXPRESSIONS
// ============================================================================

type GeneticAlgorithmBuilder() =
    member _.Bind(x, f) = async { let! result = x in return! f result }
    member _.Return(x) = async { return x }
    member _.ReturnFrom(x) = x
    member _.For(sequence, body) =
        async {
            let results = ResizeArray()
            for item in sequence do
                let! result = body item
                results.Add(result)
            return results |> Seq.toList
        }

let geneticAlgorithm = GeneticAlgorithmBuilder()

// Genetic Algorithm Optimizer Closure
let createGeneticAlgorithmOptimizer populationSize mutationRate crossoverRate selectionStrategy =
    fun fitnessFunction geneLength ->
        geneticAlgorithm {
            let random = Random()

            let! initialPopulation = async {
                return [1..populationSize]
                |> List.map (fun _ ->
                    Array.init geneLength (fun _ -> random.NextDouble())
                )
            }

            let! evolveGeneration =
                fun population ->
                    async {
                        // Selection
                        let! selectedParents = async {
                            return population
                            |> List.map (fun individual -> individual, fitnessFunction individual)
                            |> List.sortByDescending snd
                            |> List.take (populationSize / 2)
                            |> List.map fst
                        }

                        // Crossover
                        let! offspring = async {
                            return selectedParents
                            |> List.pairwise
                            |> List.map (fun (parent1, parent2) ->
                                if random.NextDouble() < crossoverRate then
                                    let crossoverPoint = random.Next(geneLength)
                                    Array.concat [
                                        parent1.[0..crossoverPoint-1]
                                        parent2.[crossoverPoint..]
                                    ]
                                else parent1
                            )
                        }

                        // Mutation
                        let! mutatedOffspring = async {
                            return offspring
                            |> List.map (fun individual ->
                                individual
                                |> Array.map (fun gene ->
                                    if random.NextDouble() < mutationRate then
                                        gene + (random.NextDouble() - 0.5) * 0.1
                                    else gene
                                )
                            )
                        }

                        return List.append selectedParents mutatedOffspring
                    }

            return {|
                InitialPopulation = initialPopulation
                EvolveGeneration = evolveGeneration
                PopulationSize = populationSize
                GeneLength = geneLength
            |}
        }

// ============================================================================
// BAYESIAN NETWORKS WITH F# COMPUTATIONAL EXPRESSIONS
// ============================================================================

type BayesianNetworkBuilder() =
    member _.Bind(x, f) =
        match x with
        | Some value -> f value
        | None -> None
    member _.Return(x) = Some x
    member _.ReturnFrom(x) = x

let bayesianNetwork = BayesianNetworkBuilder()

// Bayesian Network Inference Closure
let createBayesianNetworkInference networkStructure conditionalProbabilities =
    fun evidenceVariables queryVariables ->
        bayesianNetwork {
            let! marginalProbabilities =
                queryVariables
                |> List.map (fun queryVar ->
                    // Simplified inference (would use proper algorithms like Variable Elimination)
                    let relevantCPTs = conditionalProbabilities
                                     |> List.filter (fun cpt -> cpt.Variable = queryVar)

                    let probability = relevantCPTs
                                    |> List.fold (fun acc cpt ->
                                        let evidenceMatch = evidenceVariables
                                                          |> List.forall (fun (var, value) ->
                                                              cpt.Parents |> List.contains (var, value))
                                        if evidenceMatch then cpt.Probability else acc
                                    ) 0.5 // Default probability

                    queryVar, probability
                )

            let! jointProbability =
                marginalProbabilities
                |> List.fold (fun acc (_, prob) -> acc * prob) 1.0

            return {|
                MarginalProbabilities = marginalProbabilities
                JointProbability = jointProbability
                EvidenceVariables = evidenceVariables
                QueryVariables = queryVariables
            |}
        }

// ============================================================================
// REINFORCEMENT LEARNING WITH F# COMPUTATIONAL EXPRESSIONS
// ============================================================================

type ReinforcementLearningBuilder() =
    member _.Bind(x, f) = async { let! result = x in return! f result }
    member _.Return(x) = async { return x }
    member _.ReturnFrom(x) = x
    member _.While(guard, body) =
        async {
            let mutable continue' = true
            let results = ResizeArray()
            while continue' && guard() do
                let! result = body()
                results.Add(result)
                continue' <- guard()
            return results |> Seq.toList
        }

let reinforcementLearning = ReinforcementLearningBuilder()

// Q-Learning Agent Closure
let createQLearningAgent stateSpace actionSpace learningRate discountFactor explorationRate =
    fun environment ->
        reinforcementLearning {
            let qTable = Array2D.zeroCreate stateSpace.Length actionSpace.Length
            let random = Random()
            let mutable currentState = 0
            let mutable totalReward = 0.0

            let! selectAction = async {
                return fun state ->
                    if random.NextDouble() < explorationRate then
                        // Explore: random action
                        random.Next(actionSpace.Length)
                    else
                        // Exploit: best known action
                        [0..actionSpace.Length-1]
                        |> List.maxBy (fun action -> qTable.[state, action])
            }

            let! updateQValue = async {
                return fun state action reward nextState ->
                    let maxNextQ = [0..actionSpace.Length-1]
                                 |> List.map (fun a -> qTable.[nextState, a])
                                 |> List.max

                    let currentQ = qTable.[state, action]
                    let newQ = currentQ + learningRate * (reward + discountFactor * maxNextQ - currentQ)
                    qTable.[state, action] <- newQ
                    newQ
            }

            let! trainEpisode = async {
                return fun maxSteps ->
                    async {
                        let mutable step = 0
                        let mutable episodeReward = 0.0
                        currentState <- 0 // Reset to initial state

                        while step < maxSteps do
                            let! action = selectAction
                            let selectedAction = action currentState

                            // Get reward and next state from environment
                            let reward, nextState = environment currentState selectedAction

                            let! newQ = updateQValue
                            let updatedQ = newQ currentState selectedAction reward nextState

                            episodeReward <- episodeReward + reward
                            currentState <- nextState
                            step <- step + 1

                        return episodeReward, step
                    }
            }

            return {|
                QTable = qTable
                SelectAction = selectAction
                UpdateQValue = updateQValue
                TrainEpisode = trainEpisode
                StateSpace = stateSpace.Length
                ActionSpace = actionSpace.Length
            |}
        }

// ============================================================================
// MONTE CARLO METHODS WITH F# COMPUTATIONAL EXPRESSIONS
// ============================================================================

type MonteCarloBuilder() =
    member _.Bind(x, f) = async { let! result = x in return! f result }
    member _.Return(x) = async { return x }
    member _.ReturnFrom(x) = x
    member _.For(sequence, body) =
        async {
            let results = ResizeArray()
            for item in sequence do
                let! result = body item
                results.Add(result)
            return results |> Seq.toList
        }

let monteCarlo = MonteCarloBuilder()

// Monte Carlo Integration Closure
let createMonteCarloIntegration targetFunction bounds numSamples =
    fun () ->
        monteCarlo {
            let random = Random()
            let (lowerBounds, upperBounds) = bounds
            let dimensions = lowerBounds.Length

            let! samples = async {
                return [1..numSamples]
                |> List.map (fun _ ->
                    Array.init dimensions (fun i ->
                        lowerBounds.[i] + random.NextDouble() * (upperBounds.[i] - lowerBounds.[i])
                    )
                )
            }

            let! functionValues = async {
                return samples |> List.map targetFunction
            }

            let! integral = async {
                let average = functionValues |> List.average
                let volume = Array.fold2 (fun acc lower upper -> acc * (upper - lower)) 1.0 lowerBounds upperBounds
                return average * volume
            }

            let! standardError = async {
                let variance = functionValues
                             |> List.map (fun x -> (x - (functionValues |> List.average)) ** 2.0)
                             |> List.average
                return sqrt(variance / float numSamples)
            }

            return {|
                Integral = integral
                StandardError = standardError
                NumSamples = numSamples
                Samples = samples |> List.length
                Dimensions = dimensions
            |}
        }

// ============================================================================
// CLOSURE FACTORY REGISTRY AND MANAGEMENT
// ============================================================================

// Closure Factory Registry
type ClosureFactoryRegistry = {
    GradientDescentOptimizers: Map<string, obj>
    StateSpaceRepresentations: Map<string, obj>
    FrequencyAnalysis: Map<string, obj>
    StateMachines: Map<string, obj>
    NeuralNetworks: Map<string, obj>
    SignalProcessing: Map<string, obj>
    GeneticAlgorithms: Map<string, obj>
    BayesianNetworks: Map<string, obj>
    ReinforcementLearning: Map<string, obj>
    MonteCarloMethods: Map<string, obj>
}

// Initialize the Enhanced Closure Factory
let initializeEnhancedClosureFactory () =
    {
        GradientDescentOptimizers = Map.ofList [
            ("adam", box createAdamOptimizer)
            ("sgd", box createGradientDescentOptimizer)
        ]
        StateSpaceRepresentations = Map.ofList [
            ("state_space", box createStateSpaceRepresentation)
            ("observability", box createObservabilityAnalysis)
        ]
        FrequencyAnalysis = Map.ofList [
            ("bode_plot", box createBodePlotGenerator)
            ("nyquist", box createNyquistStabilityAnalysis)
        ]
        StateMachines = Map.ofList [
            ("finite_state_machine", box createFiniteStateMachineBuilder)
            ("hierarchical_state_machine", box createHierarchicalStateMachine)
        ]
        NeuralNetworks = Map.ofList [
            ("neural_network", box createNeuralNetworkBuilder)
        ]
        SignalProcessing = Map.ofList [
            ("fft", box createFastFourierTransform)
            ("digital_filter", box createDigitalFilterDesigner)
        ]
        GeneticAlgorithms = Map.ofList [
            ("genetic_algorithm", box createGeneticAlgorithmOptimizer)
        ]
        BayesianNetworks = Map.ofList [
            ("bayesian_inference", box createBayesianNetworkInference)
        ]
        ReinforcementLearning = Map.ofList [
            ("q_learning", box createQLearningAgent)
        ]
        MonteCarloMethods = Map.ofList [
            ("monte_carlo_integration", box createMonteCarloIntegration)
        ]
    }

// Get closure from factory
let getClosureFromFactory (registry: ClosureFactoryRegistry) category name =
    match category with
    | "gradient_descent" -> registry.GradientDescentOptimizers.TryFind name
    | "state_space" -> registry.StateSpaceRepresentations.TryFind name
    | "frequency_analysis" -> registry.FrequencyAnalysis.TryFind name
    | "state_machines" -> registry.StateMachines.TryFind name
    | "neural_networks" -> registry.NeuralNetworks.TryFind name
    | "signal_processing" -> registry.SignalProcessing.TryFind name
    | "genetic_algorithms" -> registry.GeneticAlgorithms.TryFind name
    | "bayesian_networks" -> registry.BayesianNetworks.TryFind name
    | "reinforcement_learning" -> registry.ReinforcementLearning.TryFind name
    | "monte_carlo" -> registry.MonteCarloMethods.TryFind name
    | _ -> None
