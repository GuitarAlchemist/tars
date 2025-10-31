// ================================================
// 🔄 Hurwitz Quaternion Demo
// ================================================
// Real Hurwitz quaternions for 3D rotation optimization

namespace TarsEngine.FSharp.Cli.Commands

open System
open System.IO
open Spectre.Console

module HurwitzQuaternionDemo =

    // Real Hurwitz quaternion structure (integer or half-integer coefficients)
    type HurwitzQuaternion = {
        W: float  // Real part
        X: float  // i component
        Y: float  // j component
        Z: float  // k component
    }

    type Vector3D = {
        X: float
        Y: float
        Z: float
    }

    type RotationMatrix = {
        M11: float; M12: float; M13: float
        M21: float; M22: float; M23: float
        M31: float; M32: float; M33: float
    }

    type RobotJoint = {
        Name: string
        CurrentOrientation: HurwitzQuaternion
        TargetOrientation: HurwitzQuaternion
        Position: Vector3D
    }

    // Validate Hurwitz quaternion (coefficients must be integers or half-integers)
    let isHurwitzQuaternion (q: HurwitzQuaternion) : bool =
        let isIntegerOrHalfInteger x =
            let doubled = x * 2.0
            abs (doubled - round doubled) < 1e-10
        
        isIntegerOrHalfInteger q.W &&
        isIntegerOrHalfInteger q.X &&
        isIntegerOrHalfInteger q.Y &&
        isIntegerOrHalfInteger q.Z

    // Create Hurwitz quaternion with validation
    let createHurwitzQuaternion w x y z : HurwitzQuaternion option =
        let q = { W = w; X = x; Y = y; Z = z }
        if isHurwitzQuaternion q then Some q else None

    // Quaternion operations
    let norm (q: HurwitzQuaternion) : float =
        sqrt (q.W * q.W + q.X * q.X + q.Y * q.Y + q.Z * q.Z)

    let normalize (q: HurwitzQuaternion) : HurwitzQuaternion =
        let n = norm q
        if n > 1e-10 then
            { W = q.W / n; X = q.X / n; Y = q.Y / n; Z = q.Z / n }
        else
            { W = 1.0; X = 0.0; Y = 0.0; Z = 0.0 }

    let conjugate (q: HurwitzQuaternion) : HurwitzQuaternion =
        { W = q.W; X = -q.X; Y = -q.Y; Z = -q.Z }

    let multiply (q1: HurwitzQuaternion) (q2: HurwitzQuaternion) : HurwitzQuaternion =
        {
            W = q1.W * q2.W - q1.X * q2.X - q1.Y * q2.Y - q1.Z * q2.Z
            X = q1.W * q2.X + q1.X * q2.W + q1.Y * q2.Z - q1.Z * q2.Y
            Y = q1.W * q2.Y - q1.X * q2.Z + q1.Y * q2.W + q1.Z * q2.X
            Z = q1.W * q2.Z + q1.X * q2.Y - q1.Y * q2.X + q1.Z * q2.W
        }

    // Convert quaternion to rotation matrix
    let toRotationMatrix (q: HurwitzQuaternion) : RotationMatrix =
        let q = normalize q
        let w, x, y, z = q.W, q.X, q.Y, q.Z
        
        {
            M11 = 1.0 - 2.0 * (y * y + z * z)
            M12 = 2.0 * (x * y - w * z)
            M13 = 2.0 * (x * z + w * y)
            M21 = 2.0 * (x * y + w * z)
            M22 = 1.0 - 2.0 * (x * x + z * z)
            M23 = 2.0 * (y * z - w * x)
            M31 = 2.0 * (x * z - w * y)
            M32 = 2.0 * (y * z + w * x)
            M33 = 1.0 - 2.0 * (x * x + y * y)
        }

    // Calculate angular distance between quaternions
    let angularDistance (q1: HurwitzQuaternion) (q2: HurwitzQuaternion) : float =
        let q1 = normalize q1
        let q2 = normalize q2
        let dot = q1.W * q2.W + q1.X * q2.X + q1.Y * q2.Y + q1.Z * q2.Z
        let clampedDot = max -1.0 (min 1.0 (abs dot))
        2.0 * acos clampedDot

    // Spherical Linear Interpolation (SLERP) for smooth rotation
    let slerp (q1: HurwitzQuaternion) (q2: HurwitzQuaternion) (t: float) : HurwitzQuaternion =
        let q1 = normalize q1
        let q2 = normalize q2
        let dot = q1.W * q2.W + q1.X * q2.X + q1.Y * q2.Y + q1.Z * q2.Z
        
        let (q2Final, dotFinal) =
            if dot < 0.0 then
                ({ W = -q2.W; X = -q2.X; Y = -q2.Y; Z = -q2.Z }, -dot)
            else
                (q2, dot)
        
        if dotFinal > 0.9995 then
            // Linear interpolation for very close quaternions
            let result = {
                W = q1.W + t * (q2Final.W - q1.W)
                X = q1.X + t * (q2Final.X - q1.X)
                Y = q1.Y + t * (q2Final.Y - q1.Y)
                Z = q1.Z + t * (q2Final.Z - q1.Z)
            }
            normalize result
        else
            let theta = acos dotFinal
            let sinTheta = sin theta
            let factor1 = sin ((1.0 - t) * theta) / sinTheta
            let factor2 = sin (t * theta) / sinTheta
            
            {
                W = factor1 * q1.W + factor2 * q2Final.W
                X = factor1 * q1.X + factor2 * q2Final.X
                Y = factor1 * q1.Y + factor2 * q2Final.Y
                Z = factor1 * q1.Z + factor2 * q2Final.Z
            }

    // Generate optimal Hurwitz quaternion for rotation
    let findOptimalHurwitzRotation (from: HurwitzQuaternion) (target: HurwitzQuaternion) : HurwitzQuaternion list =
        // Generate candidate Hurwitz quaternions near the optimal rotation
        let candidates = [
            for w in [-1.0; -0.5; 0.0; 0.5; 1.0] do
                for x in [-1.0; -0.5; 0.0; 0.5; 1.0] do
                    for y in [-1.0; -0.5; 0.0; 0.5; 1.0] do
                        for z in [-1.0; -0.5; 0.0; 0.5; 1.0] do
                            match createHurwitzQuaternion w x y z with
                            | Some q when norm q > 0.1 -> Some (normalize q)
                            | _ -> None
        ]
        
        candidates
        |> List.choose id
        |> List.distinct
        |> List.sortBy (fun q -> angularDistance from q + angularDistance q target)
        |> List.take 10

    // Calculate energy cost for rotation sequence
    let calculateRotationEnergy (rotations: HurwitzQuaternion list) : float =
        if rotations.Length < 2 then 0.0
        else
            rotations
            |> List.pairwise
            |> List.map (fun (q1, q2) -> 
                let distance = angularDistance q1 q2
                distance * distance) // Energy proportional to square of angular distance
            |> List.sum

    // Apply rotation to 3D vector
    let rotateVector (q: HurwitzQuaternion) (v: Vector3D) : Vector3D =
        let q = normalize q
        let vq = { W = 0.0; X = v.X; Y = v.Y; Z = v.Z }
        let result = multiply (multiply q vq) (conjugate q)
        { X = result.X; Y = result.Y; Z = result.Z }

    // Create sample robot joints for demonstration
    let createSampleRobotArm () : RobotJoint list =
        [
            {
                Name = "Base Joint"
                CurrentOrientation = { W = 1.0; X = 0.0; Y = 0.0; Z = 0.0 }
                TargetOrientation = { W = 0.5; X = 0.5; Y = 0.5; Z = 0.5 }
                Position = { X = 0.0; Y = 0.0; Z = 0.0 }
            }
            {
                Name = "Shoulder Joint"
                CurrentOrientation = { W = 1.0; X = 0.0; Y = 0.0; Z = 0.0 }
                TargetOrientation = { W = 0.0; X = 1.0; Y = 0.0; Z = 0.0 }
                Position = { X = 0.0; Y = 0.0; Z = 10.0 }
            }
            {
                Name = "Elbow Joint"
                CurrentOrientation = { W = 0.5; X = 0.5; Y = 0.5; Z = 0.5 }
                TargetOrientation = { W = 1.0; X = 0.0; Y = 0.0; Z = 0.0 }
                Position = { X = 0.0; Y = 0.0; Z = 20.0 }
            }
        ]

    // Optimize robot arm movement using Hurwitz quaternions
    let optimizeRobotMovement (joints: RobotJoint list) : (RobotJoint * HurwitzQuaternion list * float) list =
        joints
        |> List.map (fun joint ->
            let optimalPath = findOptimalHurwitzRotation joint.CurrentOrientation joint.TargetOrientation
            let energy = calculateRotationEnergy (joint.CurrentOrientation :: optimalPath @ [joint.TargetOrientation])
            (joint, optimalPath, energy))
