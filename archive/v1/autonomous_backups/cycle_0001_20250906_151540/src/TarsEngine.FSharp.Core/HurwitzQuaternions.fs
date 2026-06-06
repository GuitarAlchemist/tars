namespace TarsEngine.FSharp.Core

open System
open System.Numerics

/// Hurwitz Quaternions for Advanced Mathematical Reasoning in TARS
/// Implements 4D prime lattice system for sophisticated cognitive capabilities
module HurwitzQuaternions =

    /// Hurwitz quaternion with integer coefficients
    type HurwitzQuaternion = {
        A: int  // Real part
        B: int  // i coefficient  
        C: int  // j coefficient
        D: int  // k coefficient
    }

    /// Quaternion operations
    module Operations =
        
        /// Create a Hurwitz quaternion
        let create a b c d = { A = a; B = b; C = c; D = d }
        
        /// Zero quaternion
        let zero = create 0 0 0 0
        
        /// Unit quaternion
        let one = create 1 0 0 0
        
        /// Quaternion addition
        let add q1 q2 = {
            A = q1.A + q2.A
            B = q1.B + q2.B
            C = q1.C + q2.C
            D = q1.D + q2.D
        }
        
        /// Quaternion multiplication (non-commutative)
        let multiply q1 q2 = {
            A = q1.A * q2.A - q1.B * q2.B - q1.C * q2.C - q1.D * q2.D
            B = q1.A * q2.B + q1.B * q2.A + q1.C * q2.D - q1.D * q2.C
            C = q1.A * q2.C - q1.B * q2.D + q1.C * q2.A + q1.D * q2.B
            D = q1.A * q2.D + q1.B * q2.C - q1.C * q2.B + q1.D * q2.A
        }
        
        /// Quaternion conjugate
        let conjugate q = { A = q.A; B = -q.B; C = -q.C; D = -q.D }
        
        /// Quaternion norm (N(q) = a² + b² + c² + d²)
        let norm q = q.A * q.A + q.B * q.B + q.C * q.C + q.D * q.D
        
        /// Check if quaternion is a unit (norm = 1)
        let isUnit q = norm q = 1

    /// Prime testing for Hurwitz quaternions
    module PrimeTesting =
        
        /// Simple primality test for integers
        let isPrime n =
            if n < 2 then false
            elif n = 2 then true
            elif n % 2 = 0 then false
            else
                let limit = int (sqrt (float n))
                let rec check i =
                    if i > limit then true
                    elif n % i = 0 then false
                    else check (i + 2)
                check 3
        
        /// Check if Hurwitz quaternion has prime norm
        let hasPrimeNorm q = 
            let n = Operations.norm q
            isPrime n
        
        /// Generate Hurwitz quaternions with prime norms up to bound
        let generatePrimeNormQuaternions bound =
            let mutable results = []
            for a in -bound..bound do
                for b in -bound..bound do
                    for c in -bound..bound do
                        for d in -bound..bound do
                            let q = Operations.create a b c d
                            if hasPrimeNorm q then
                                results <- q :: results
            results |> List.rev

    /// 4D lattice operations for cognitive reasoning
    module Lattice =
        
        /// Lattice point in 4D space
        type LatticePoint = {
            Quaternion: HurwitzQuaternion
            Coordinates: float * float * float * float
            Norm: int
            IsPrime: bool
        }
        
        /// Convert quaternion to 4D coordinates
        let toCoordinates q = 
            (float q.A, float q.B, float q.C, float q.D)
        
        /// Create lattice point from quaternion
        let createLatticePoint q = {
            Quaternion = q
            Coordinates = toCoordinates q
            Norm = Operations.norm q
            IsPrime = PrimeTesting.hasPrimeNorm q
        }
        
        /// Generate 4D prime lattice within radius
        let generatePrimeLattice radius =
            let bound = int (sqrt (float radius))
            PrimeTesting.generatePrimeNormQuaternions bound
            |> List.map createLatticePoint
            |> List.filter (fun p -> p.Norm <= radius)
        
        /// Distance between two lattice points
        let distance p1 p2 =
            let (x1, y1, z1, w1) = p1.Coordinates
            let (x2, y2, z2, w2) = p2.Coordinates
            sqrt ((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2) + (w1-w2)*(w1-w2))

    /// Geometric belief encoding using quaternions
    module BeliefEncoding =
        
        /// Belief state in quaternionic space
        type QuaternionicBelief = {
            Belief: string
            Quaternion: HurwitzQuaternion
            Confidence: float
            Timestamp: DateTime
            Context: string
        }
        
        /// Encode belief as quaternion based on semantic properties
        let encodeBelief (belief: string) (confidence: float) (context: string) =
            // Simple encoding: hash belief string to quaternion coefficients
            let hash = belief.GetHashCode()
            let a = (hash >>> 24) &&& 0xFF |> int
            let b = (hash >>> 16) &&& 0xFF |> int  
            let c = (hash >>> 8) &&& 0xFF |> int
            let d = hash &&& 0xFF |> int
            
            // Normalize to reasonable range
            let normalize x = (x % 20) - 10
            
            {
                Belief = belief
                Quaternion = Operations.create (normalize a) (normalize b) (normalize c) (normalize d)
                Confidence = confidence
                Timestamp = DateTime.UtcNow
                Context = context
            }
        
        /// Measure belief similarity using quaternion distance
        let beliefSimilarity b1 b2 =
            let q1 = b1.Quaternion
            let q2 = b2.Quaternion
            let diff = Operations.add q1 (Operations.conjugate q2)
            let distance = float (Operations.norm diff)
            1.0 / (1.0 + distance)  // Similarity decreases with distance
        
        /// Detect contradictory beliefs using quaternion properties
        let detectContradiction beliefs =
            beliefs
            |> List.pairwise
            |> List.filter (fun (b1, b2) -> beliefSimilarity b1 b2 < 0.1)
            |> List.map (fun (b1, b2) -> sprintf "Contradiction: '%s' vs '%s'" b1.Belief b2.Belief)

    /// Rotation-based evolution for agent mutations
    module Evolution =
        
        /// Rotation quaternion for 4D transformations
        type RotationQuaternion = {
            Axis: float * float * float  // 3D rotation axis
            Angle: float                 // Rotation angle
            Quaternion: HurwitzQuaternion
        }
        
        /// Create rotation quaternion from axis and angle
        let createRotation (axis: float * float * float) (angle: float) =
            let (x, y, z) = axis
            let halfAngle = angle / 2.0
            let sinHalf = sin halfAngle
            let cosHalf = cos halfAngle
            
            // Convert to Hurwitz quaternion (approximate)
            let a = int (cosHalf * 100.0)
            let b = int (x * sinHalf * 100.0)
            let c = int (y * sinHalf * 100.0)
            let d = int (z * sinHalf * 100.0)
            
            {
                Axis = axis
                Angle = angle
                Quaternion = Operations.create a b c d
            }
        
        /// Apply rotation to belief state
        let rotateBelief rotation belief =
            let rotated = Operations.multiply rotation.Quaternion belief.Quaternion
            { belief with 
                Quaternion = rotated
                Timestamp = DateTime.UtcNow }
        
        /// Generate mutation through quaternionic rotation
        let mutateBelief belief mutationStrength =
            // Random rotation for mutation
            let random = Random()
            let axis = (random.NextDouble() - 0.5, random.NextDouble() - 0.5, random.NextDouble() - 0.5)
            let angle = mutationStrength * Math.PI * 2.0
            let rotation = createRotation axis angle
            rotateBelief rotation belief

    /// Integration with TARS cognitive architecture
    module TarsIntegration =
        
        /// TARS cognitive state using quaternions
        type TarsQuaternionicState = {
            Beliefs: BeliefEncoding.QuaternionicBelief list
            PrimeLattice: Lattice.LatticePoint list
            CurrentFocus: HurwitzQuaternion
            EvolutionHistory: (DateTime * HurwitzQuaternion) list
            Contradictions: string list
        }
        
        /// Initialize TARS quaternionic reasoning system
        let initializeTarsQuaternions() = {
            Beliefs = []
            PrimeLattice = Lattice.generatePrimeLattice 100
            CurrentFocus = Operations.one
            EvolutionHistory = [(DateTime.UtcNow, Operations.one)]
            Contradictions = []
        }
        
        /// Add belief to TARS quaternionic state
        let addBelief state belief confidence context =
            let qBelief = BeliefEncoding.encodeBelief belief confidence context
            let newContradictions = BeliefEncoding.detectContradiction (qBelief :: state.Beliefs)
            
            { state with 
                Beliefs = qBelief :: state.Beliefs
                Contradictions = state.Contradictions @ newContradictions
                EvolutionHistory = (DateTime.UtcNow, qBelief.Quaternion) :: state.EvolutionHistory }
        
        /// Evolve TARS state through quaternionic mutations
        let evolveTarsState state mutationStrength =
            let mutatedBeliefs = 
                state.Beliefs 
                |> List.map (Evolution.mutateBelief mutationStrength)
            
            { state with 
                Beliefs = mutatedBeliefs
                EvolutionHistory = (DateTime.UtcNow, state.CurrentFocus) :: state.EvolutionHistory }
        
        /// Analyze quaternionic patterns in TARS evolution
        let analyzeEvolutionPatterns state =
            let norms = state.EvolutionHistory |> List.map (fun (_, q) -> Operations.norm q)
            let primeNorms = norms |> List.filter PrimeTesting.isPrime
            let averageNorm = if norms.IsEmpty then 0.0 else norms |> List.map float |> List.average
            
            {|
                TotalEvolutions = state.EvolutionHistory.Length
                PrimeEvolutions = primeNorms.Length
                AverageNorm = averageNorm
                PrimeRatio = if norms.IsEmpty then 0.0 else float primeNorms.Length / float norms.Length
                CurrentBeliefs = state.Beliefs.Length
                ActiveContradictions = state.Contradictions.Length
            |}

    /// Non-commutative reasoning patterns
    module NonCommutativeReasoning =
        
        /// Reasoning step with quaternionic transformation
        type ReasoningStep = {
            Description: string
            InputQuaternion: HurwitzQuaternion
            Transformation: HurwitzQuaternion
            OutputQuaternion: HurwitzQuaternion
            IsCommutative: bool
            Timestamp: DateTime
        }
        
        /// Apply non-commutative reasoning transformation
        let applyReasoning input transformation description =
            let output1 = Operations.multiply input transformation
            let output2 = Operations.multiply transformation input
            let isCommutative = output1 = output2
            
            {
                Description = description
                InputQuaternion = input
                Transformation = transformation
                OutputQuaternion = output1
                IsCommutative = isCommutative
                Timestamp = DateTime.UtcNow
            }
        
        /// Chain multiple reasoning steps
        let chainReasoning steps =
            steps
            |> List.fold (fun acc step -> 
                applyReasoning acc.OutputQuaternion step.Transformation step.Description
            ) (List.head steps)
        
        /// Detect non-commutative insights
        let detectNonCommutativeInsights steps =
            steps
            |> List.filter (fun s -> not s.IsCommutative)
            |> List.map (fun s -> sprintf "Non-commutative insight: %s" s.Description)

    /// Guitar Alchemist Integration - Musical Quaternions
    module MusicalQuaternions =

        /// Musical interval encoded as quaternion
        type MusicalQuaternion = {
            Interval: string
            Frequency: float
            Quaternion: HurwitzQuaternion
            Harmonic: int
        }

        /// Encode musical interval as quaternion
        let encodeMusicalInterval interval frequency =
            // Map frequency to quaternion coefficients
            let logFreq = log frequency
            let a = int (logFreq * 10.0) % 20 - 10
            let b = int (frequency / 100.0) % 20 - 10
            let c = int (frequency * 0.01) % 20 - 10
            let d = int (frequency * 0.001) % 20 - 10

            {
                Interval = interval
                Frequency = frequency
                Quaternion = Operations.create a b c d
                Harmonic = int (frequency / 440.0)  // Relative to A440
            }

        /// Calculate harmonic relationships using quaternions
        let harmonicRelationship m1 m2 =
            let product = Operations.multiply m1.Quaternion m2.Quaternion
            let norm = Operations.norm product
            if PrimeTesting.isPrime norm then
                sprintf "Prime harmonic relationship: %s + %s" m1.Interval m2.Interval
            else
                sprintf "Composite harmonic: %s + %s (norm: %d)" m1.Interval m2.Interval norm
