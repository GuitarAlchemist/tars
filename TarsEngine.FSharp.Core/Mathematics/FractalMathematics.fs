// Fractal Mathematics - Advanced Multi-Scale Operations for TARS
// Implements Takagi functions, Rham curves, dual quaternions, and Lie algebra operations

namespace TarsEngine.FSharp.Core.Mathematics

open System
open System.Threading.Tasks

/// Complex number for advanced mathematical operations
type ComplexNumber = {
    Real: float
    Imaginary: float
} with
    static member (+) (a, b) = { Real = a.Real + b.Real; Imaginary = a.Imaginary + b.Imaginary }
    static member (-) (a, b) = { Real = a.Real - b.Real; Imaginary = a.Imaginary - b.Imaginary }
    static member (*) (a, b) = { 
        Real = a.Real * b.Real - a.Imaginary * b.Imaginary
        Imaginary = a.Real * b.Imaginary + a.Imaginary * b.Real 
    }
    static member Zero = { Real = 0.0; Imaginary = 0.0 }
    static member One = { Real = 1.0; Imaginary = 0.0 }
    static member I = { Real = 0.0; Imaginary = 1.0 }
    member this.Magnitude = sqrt(this.Real * this.Real + this.Imaginary * this.Imaginary)
    member this.Conjugate = { Real = this.Real; Imaginary = -this.Imaginary }

/// Quaternion for 3D rotations
type Quaternion = {
    W: float
    X: float
    Y: float
    Z: float
} with
    static member (*) (q1, q2) = {
        W = q1.W*q2.W - q1.X*q2.X - q1.Y*q2.Y - q1.Z*q2.Z
        X = q1.W*q2.X + q1.X*q2.W + q1.Y*q2.Z - q1.Z*q2.Y
        Y = q1.W*q2.Y - q1.X*q2.Z + q1.Y*q2.W + q1.Z*q2.X
        Z = q1.W*q2.Z + q1.X*q2.Y - q1.Y*q2.X + q1.Z*q2.W
    }
    static member Identity = { W = 1.0; X = 0.0; Y = 0.0; Z = 0.0 }
    member this.Norm = sqrt(this.W*this.W + this.X*this.X + this.Y*this.Y + this.Z*this.Z)
    member this.Conjugate = { W = this.W; X = -this.X; Y = -this.Y; Z = -this.Z }

/// Dual quaternion for rigid body transformations
type DualQuaternion = {
    Real: Quaternion      // Rotation
    Dual: Quaternion      // Translation
} with
    static member (*) (dq1, dq2) = {
        Real = dq1.Real * dq2.Real
        Dual = dq1.Real * dq2.Dual + dq1.Dual * dq2.Real
    }
    static member Identity = { Real = Quaternion.Identity; Dual = { W = 0.0; X = 0.0; Y = 0.0; Z = 0.0 } }

/// Lie algebra element (tangent space vector)
type LieAlgebraElement = {
    X: float
    Y: float
    Z: float
} with
    member this.Norm = sqrt(this.X*this.X + this.Y*this.Y + this.Z*this.Z)

/// Fractal curve point
type FractalPoint = {
    X: float
    Y: float
    Parameter: float
    Iteration: int
}

/// Takagi function parameters
type TakagiParameters = {
    Depth: int
    Amplitude: float
    Scale: float
    Roughness: float
}

/// Rham curve parameters
type RhamParameters = {
    Depth: int
    Roughness: float
    Smoothness: float
    InterpolationPoints: FractalPoint[]
}

/// Fractal Mathematics Module
module FractalMathematics =
    
    // ============================================================================
    // TAKAGI FUNCTIONS - MULTI-SCALE FRACTAL NOISE
    // ============================================================================
    
    /// Takagi function: multi-scale fractal noise generator
    let takagi (x: float) (params: TakagiParameters) =
        let rec aux n acc currentAmp =
            if n >= params.Depth then acc
            else 
                // Compute triangular wave contribution: distance from the nearest half-integer
                let scaledX = (params.Scale ** float n) * x
                let triangularWave = abs (scaledX % 1.0 - 0.5)
                let value = currentAmp * triangularWave
                aux (n + 1) (acc + value) (currentAmp * params.Roughness)
        aux 0 0.0 params.Amplitude
    
    /// Create Takagi function closure
    let createTakagiFunction depth amplitude scale roughness =
        let params = { Depth = depth; Amplitude = amplitude; Scale = scale; Roughness = roughness }
        fun x -> takagi x params
    
    /// Multi-dimensional Takagi noise
    let takagiNoise (vector: float[]) (params: TakagiParameters) =
        vector |> Array.map (fun x -> takagi x params)
    
    /// Takagi-based perturbation for optimization
    let takagiPerturbation (vector: float[]) (learningRate: float) (params: TakagiParameters) =
        vector |> Array.map (fun x -> x + learningRate * takagi x params)
    
    // ============================================================================
    // RHAM CURVES - SMOOTH RECURSIVE INTERPOLATION
    // ============================================================================
    
    /// Rham curve recursive interpolation
    let rec rhamCurve (t: float) (points: FractalPoint[]) (depth: int) (roughness: float) =
        if depth = 0 || points.Length < 2 then
            // Base case: linear interpolation
            if points.Length = 0 then { X = 0.0; Y = 0.0; Parameter = t; Iteration = depth }
            elif points.Length = 1 then points.[0]
            else
                let p1, p2 = points.[0], points.[1]
                let alpha = t
                { 
                    X = p1.X + alpha * (p2.X - p1.X)
                    Y = p1.Y + alpha * (p2.Y - p1.Y)
                    Parameter = t
                    Iteration = depth
                }
        else
            // Recursive case: subdivide and add perturbation
            let midPoints = Array.zeroCreate (points.Length - 1)
            for i in 0..points.Length-2 do
                let p1, p2 = points.[i], points.[i+1]
                let midX = (p1.X + p2.X) / 2.0
                let midY = (p1.Y + p2.Y) / 2.0
                
                // Add fractal perturbation
                let perturbationX = (Random().NextDouble() - 0.5) * roughness
                let perturbationY = (Random().NextDouble() - 0.5) * roughness
                
                midPoints.[i] <- {
                    X = midX + perturbationX
                    Y = midY + perturbationY
                    Parameter = (p1.Parameter + p2.Parameter) / 2.0
                    Iteration = depth
                }
            
            // Recursively interpolate with reduced roughness
            let newPoints = Array.concat [points; midPoints] |> Array.sortBy (fun p -> p.Parameter)
            rhamCurve t newPoints (depth - 1) (roughness * 0.5)
    
    /// Create Rham curve closure
    let createRhamCurve initialPoints depth roughness smoothness =
        let params = { 
            Depth = depth
            Roughness = roughness
            Smoothness = smoothness
            InterpolationPoints = initialPoints
        }
        fun t -> rhamCurve t params.InterpolationPoints params.Depth params.Roughness
    
    /// Generate Rham curve path
    let generateRhamPath (startPoint: FractalPoint) (endPoint: FractalPoint) (depth: int) (roughness: float) (steps: int) =
        async {
            let initialPoints = [|startPoint; endPoint|]
            let rhamFunc = createRhamCurve initialPoints depth roughness 1.0
            
            let path = Array.init steps (fun i ->
                let t = float i / float (steps - 1)
                rhamFunc t)
            
            return {|
                Path = path
                StartPoint = startPoint
                EndPoint = endPoint
                Depth = depth
                Roughness = roughness
                Steps = steps
            |}
        }
    
    // ============================================================================
    // DUAL QUATERNIONS - ADVANCED SPATIAL TRANSFORMATIONS
    // ============================================================================
    
    /// Create dual quaternion from rotation and translation
    let createDualQuaternion (rotation: Quaternion) (translation: float[]) =
        if translation.Length <> 3 then failwith "Translation must be 3D vector"
        
        let translationQuat = { W = 0.0; X = translation.[0]; Y = translation.[1]; Z = translation.[2] }
        let dualPart = { 
            W = -0.5 * (rotation.X * translation.[0] + rotation.Y * translation.[1] + rotation.Z * translation.[2])
            X = 0.5 * (rotation.W * translation.[0] + rotation.Y * translation.[2] - rotation.Z * translation.[1])
            Y = 0.5 * (rotation.W * translation.[1] + rotation.Z * translation.[0] - rotation.X * translation.[2])
            Z = 0.5 * (rotation.W * translation.[2] + rotation.X * translation.[1] - rotation.Y * translation.[0])
        }
        
        { Real = rotation; Dual = dualPart }
    
    /// Apply dual quaternion transformation to point
    let transformPoint (dq: DualQuaternion) (point: float[]) =
        if point.Length <> 3 then failwith "Point must be 3D"
        
        // Convert point to quaternion
        let pointQuat = { W = 0.0; X = point.[0]; Y = point.[1]; Z = point.[2] }
        
        // Apply rotation: q * p * q*
        let rotatedPoint = dq.Real * pointQuat * dq.Real.Conjugate
        
        // Extract translation from dual part (simplified)
        let translatedPoint = {
            W = rotatedPoint.W
            X = rotatedPoint.X + 2.0 * dq.Dual.X
            Y = rotatedPoint.Y + 2.0 * dq.Dual.Y
            Z = rotatedPoint.Z + 2.0 * dq.Dual.Z
        }
        
        [|translatedPoint.X; translatedPoint.Y; translatedPoint.Z|]
    
    /// Interpolate between dual quaternions (SLERP)
    let interpolateDualQuaternions (dq1: DualQuaternion) (dq2: DualQuaternion) (t: float) =
        // Simplified linear interpolation (proper SLERP would be more complex)
        let interpReal = {
            W = dq1.Real.W + t * (dq2.Real.W - dq1.Real.W)
            X = dq1.Real.X + t * (dq2.Real.X - dq1.Real.X)
            Y = dq1.Real.Y + t * (dq2.Real.Y - dq1.Real.Y)
            Z = dq1.Real.Z + t * (dq2.Real.Z - dq1.Real.Z)
        }
        
        let interpDual = {
            W = dq1.Dual.W + t * (dq2.Dual.W - dq1.Dual.W)
            X = dq1.Dual.X + t * (dq2.Dual.X - dq1.Dual.X)
            Y = dq1.Dual.Y + t * (dq2.Dual.Y - dq1.Dual.Y)
            Z = dq1.Dual.Z + t * (dq2.Dual.Z - dq1.Dual.Z)
        }
        
        { Real = interpReal; Dual = interpDual }
    
    // ============================================================================
    // LIE ALGEBRA OPERATIONS - SMOOTH MANIFOLD TRANSITIONS
    // ============================================================================
    
    /// Exponential map from Lie algebra to quaternion
    let expMap (lieElement: LieAlgebraElement) =
        let theta = lieElement.Norm
        if theta < 1e-6 then 
            Quaternion.Identity
        else
            let s = sin(theta / 2.0) / theta
            {
                W = cos(theta / 2.0)
                X = lieElement.X * s
                Y = lieElement.Y * s
                Z = lieElement.Z * s
            }
    
    /// Logarithmic map from quaternion to Lie algebra
    let logMap (quaternion: Quaternion) =
        let norm = quaternion.Norm
        if norm < 1e-6 then
            { X = 0.0; Y = 0.0; Z = 0.0 }
        else
            let theta = 2.0 * acos(abs(quaternion.W))
            if theta < 1e-6 then
                { X = 0.0; Y = 0.0; Z = 0.0 }
            else
                let s = theta / sin(theta / 2.0)
                {
                    X = quaternion.X * s
                    Y = quaternion.Y * s
                    Z = quaternion.Z * s
                }
    
    /// Lie bracket operation [X, Y] = XY - YX
    let lieBracket (x: LieAlgebraElement) (y: LieAlgebraElement) =
        {
            X = x.Y * y.Z - x.Z * y.Y
            Y = x.Z * y.X - x.X * y.Z
            Z = x.X * y.Y - x.Y * y.X
        }
    
    /// Smooth interpolation on manifold using Lie algebra
    let manifoldInterpolation (q1: Quaternion) (q2: Quaternion) (t: float) =
        let relativeRotation = q2 * q1.Conjugate
        let lieElement = logMap relativeRotation
        let scaledLie = { X = t * lieElement.X; Y = t * lieElement.Y; Z = t * lieElement.Z }
        let interpolatedRotation = expMap scaledLie
        q1 * interpolatedRotation
    
    // ============================================================================
    // FRACTAL MATHEMATICS CLOSURES
    // ============================================================================
    
    /// Create fractal noise generator closure
    let createFractalNoiseGenerator depth amplitude scale roughness =
        fun (input: float[]) ->
            async {
                let params = { Depth = depth; Amplitude = amplitude; Scale = scale; Roughness = roughness }
                let noisyOutput = takagiNoise input params
                
                return {|
                    NoiseType = "Takagi Fractal"
                    Input = input
                    Output = noisyOutput
                    Parameters = params
                    Complexity = float depth * amplitude
                |}
            }
    
    /// Create fractal path generator closure
    let createFractalPathGenerator depth roughness =
        fun (startPoint: float[]) (endPoint: float[]) ->
            async {
                if startPoint.Length <> 2 || endPoint.Length <> 2 then
                    return {| Error = "Points must be 2D for fractal path generation" |}
                else
                    let start = { X = startPoint.[0]; Y = startPoint.[1]; Parameter = 0.0; Iteration = 0 }
                    let endPt = { X = endPoint.[0]; Y = endPoint.[1]; Parameter = 1.0; Iteration = 0 }
                    
                    let! pathResult = generateRhamPath start endPt depth roughness 100
                    
                    return {|
                        PathType = "Rham Fractal Curve"
                        StartPoint = startPoint
                        EndPoint = endPoint
                        Path = pathResult.Path |> Array.map (fun p -> [|p.X; p.Y|])
                        Depth = depth
                        Roughness = roughness
                        PathLength = pathResult.Path.Length
                    |} :> obj
            }
    
    /// Create dual quaternion transformation closure
    let createDualQuaternionTransformer () =
        fun (rotation: float[]) (translation: float[]) ->
            async {
                if rotation.Length <> 4 || translation.Length <> 3 then
                    return {| Error = "Rotation must be 4D quaternion, translation must be 3D vector" |}
                else
                    let quat = { W = rotation.[0]; X = rotation.[1]; Y = rotation.[2]; Z = rotation.[3] }
                    let dualQuat = createDualQuaternion quat translation
                    
                    return {|
                        TransformationType = "Dual Quaternion"
                        Rotation = rotation
                        Translation = translation
                        DualQuaternion = dualQuat
                        Transform = fun (point: float[]) -> transformPoint dualQuat point
                    |} :> obj
            }
    
    /// Create Lie algebra manifold interpolator closure
    let createLieAlgebraInterpolator () =
        fun (quaternion1: float[]) (quaternion2: float[]) ->
            async {
                if quaternion1.Length <> 4 || quaternion2.Length <> 4 then
                    return {| Error = "Quaternions must be 4D" |}
                else
                    let q1 = { W = quaternion1.[0]; X = quaternion1.[1]; Y = quaternion1.[2]; Z = quaternion1.[3] }
                    let q2 = { W = quaternion2.[0]; X = quaternion2.[1]; Y = quaternion2.[2]; Z = quaternion2.[3] }
                    
                    let interpolator = fun t ->
                        let result = manifoldInterpolation q1 q2 t
                        [|result.W; result.X; result.Y; result.Z|]
                    
                    return {|
                        InterpolationType = "Lie Algebra Manifold"
                        Quaternion1 = quaternion1
                        Quaternion2 = quaternion2
                        Interpolator = interpolator
                        SmoothTransition = true
                    |} :> obj
            }
    
    /// Create fractal perturbation closure for optimization
    let createFractalPerturbationOptimizer depth amplitude scale roughness learningRate =
        fun (parameters: float[]) ->
            async {
                let takagiParams = { Depth = depth; Amplitude = amplitude; Scale = scale; Roughness = roughness }
                let perturbedParams = takagiPerturbation parameters learningRate takagiParams
                
                return {|
                    OptimizationType = "Fractal Perturbation"
                    OriginalParameters = parameters
                    PerturbedParameters = perturbedParams
                    LearningRate = learningRate
                    FractalParameters = takagiParams
                    PerturbationMagnitude = Array.map2 (fun orig pert -> abs(pert - orig)) parameters perturbedParams |> Array.average
                |}
            }
