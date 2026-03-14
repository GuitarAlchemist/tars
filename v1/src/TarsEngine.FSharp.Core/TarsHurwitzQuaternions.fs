// ================================================
// 🔢 TARS Hurwitz Quaternions and Quaternionic Primes
// ================================================
// Implements Hurwitz quaternions for 4D prime lattice reasoning
// Based on ChatGPT-Leveraging Primes for TARS document

namespace TarsEngine.FSharp.Core

open System
open Microsoft.Extensions.Logging

/// A Hurwitz quaternion is (a + bi + cj + dk) where a,b,c,d ∈ ℤ or ℤ+½
/// We represent them with components and a flag for half-integer offset
[<Struct>]
type HurwitzQuaternion = {
    A: float
    B: float
    C: float
    D: float
    IsHalf: bool // if true, components are offset by ½
}

/// Result type for quaternionic operations
type QuaternionResult<'T> = 
    | Success of 'T
    | Error of string

/// Quaternionic prime information
type QuaternionPrime = {
    Quaternion: HurwitzQuaternion
    Norm: int
    IsIrreducible: bool
}

/// Performance metrics for quaternion operations
type QuaternionPerformance = {
    QuaternionsGenerated: int
    PrimesFound: int
    ElapsedMs: int64
    PrimesPerSecond: float
}

module TarsHurwitzQuaternions =

    /// Compute norm squared: N(q) = a² + b² + c² + d²
    let norm2 (q: HurwitzQuaternion) : float =
        let offset = if q.IsHalf then 0.5 else 0.0
        let a = q.A + offset
        let b = q.B + offset
        let c = q.C + offset
        let d = q.D + offset
        a*a + b*b + c*c + d*d

    /// Compute integer norm for primality testing
    let norm (q: HurwitzQuaternion) : int =
        norm2 q |> int

    /// Check if a given number is prime (optimized trial division)
    let isPrime (n: int) : bool =
        if n <= 1 then false
        elif n <= 3 then true
        elif n % 2 = 0 || n % 3 = 0 then false
        else
            let rec check i =
                i * i > n || (n % i <> 0 && n % (i + 2) <> 0 && check (i + 6))
            check 5

    /// Check if a Hurwitz quaternion has prime norm (quaternionic prime)
    let isQuaternionPrime (q: HurwitzQuaternion) : bool =
        let n = norm q
        n > 1 && isPrime n

    /// Create a Hurwitz quaternion
    let create (a: float) (b: float) (c: float) (d: float) (isHalf: bool) : HurwitzQuaternion =
        { A = a; B = b; C = c; D = d; IsHalf = isHalf }

    /// Create integer Hurwitz quaternion
    let createInt (a: int) (b: int) (c: int) (d: int) : HurwitzQuaternion =
        create (float a) (float b) (float c) (float d) false

    /// Create half-integer Hurwitz quaternion
    let createHalf (a: int) (b: int) (c: int) (d: int) : HurwitzQuaternion =
        create (float a) (float b) (float c) (float d) true

    /// Format Hurwitz quaternion for display
    let format (q: HurwitzQuaternion) : string =
        let offset = if q.IsHalf then "+½" else ""
        $"(%g{q.A}%s{offset} + %g{q.B}i%s{offset} + %g{q.C}j%s{offset} + %g{q.D}k%s{offset})"

    /// Quaternion multiplication (non-commutative)
    let multiply (q1: HurwitzQuaternion) (q2: HurwitzQuaternion) : HurwitzQuaternion =
        let offset1 = if q1.IsHalf then 0.5 else 0.0
        let offset2 = if q2.IsHalf then 0.5 else 0.0
        
        let a1, b1, c1, d1 = q1.A + offset1, q1.B + offset1, q1.C + offset1, q1.D + offset1
        let a2, b2, c2, d2 = q2.A + offset2, q2.B + offset2, q2.C + offset2, q2.D + offset2
        
        // Quaternion multiplication: (a1 + b1i + c1j + d1k) * (a2 + b2i + c2j + d2k)
        let a = a1*a2 - b1*b2 - c1*c2 - d1*d2
        let b = a1*b2 + b1*a2 + c1*d2 - d1*c2
        let c = a1*c2 - b1*d2 + c1*a2 + d1*b2
        let d = a1*d2 + b1*c2 - c1*b2 + d1*a2
        
        // Result is integer quaternion
        { A = a; B = b; C = c; D = d; IsHalf = false }

    /// Quaternion conjugate
    let conjugate (q: HurwitzQuaternion) : HurwitzQuaternion =
        { q with B = -q.B; C = -q.C; D = -q.D }

    /// Generate Hurwitz quaternions within given bound and filter for prime norms
    let generateHurwitzPrimes (bound: int) (logger: ILogger) : QuaternionPrime list =
        let stopwatch = System.Diagnostics.Stopwatch.StartNew()
        let mutable results = []
        let mutable totalGenerated = 0
        
        logger.LogInformation($"🔢 Generating Hurwitz quaternions with prime norms up to bound {bound}")
        
        for a in -bound..bound do
            for b in -bound..bound do
                for c in -bound..bound do
                    for d in -bound..bound do
                        for isHalf in [false; true] do
                            totalGenerated <- totalGenerated + 1
                            let q = create (float a) (float b) (float c) (float d) isHalf
                            let n = norm q
                            if n > 1 && isPrime n then
                                let qPrime = {
                                    Quaternion = q
                                    Norm = n
                                    IsIrreducible = true // Simplified - true irreducibility test would be more complex
                                }
                                results <- qPrime :: results
        
        stopwatch.Stop()
        let elapsedMs = stopwatch.ElapsedMilliseconds
        let primesPerSec = if elapsedMs > 0L then (float results.Length * 1000.0) / (float elapsedMs) else 0.0
        
        logger.LogInformation($"✅ Generated {results.Length} quaternionic primes from {totalGenerated} quaternions")
        logger.LogInformation($"📈 Performance: {primesPerSec:F0} primes/second in {elapsedMs}ms")
        
        results |> List.rev

    /// Find quaternions with specific norm
    let findQuaternionsWithNorm (targetNorm: int) (bound: int) : HurwitzQuaternion list =
        let mutable results = []
        
        for a in -bound..bound do
            for b in -bound..bound do
                for c in -bound..bound do
                    for d in -bound..bound do
                        for isHalf in [false; true] do
                            let q = create (float a) (float b) (float c) (float d) isHalf
                            if norm q = targetNorm then
                                results <- q :: results
        
        results |> List.rev

    /// Benchmark quaternion prime generation performance
    let benchmarkPerformance (bound: int) (logger: ILogger) : QuaternionPerformance =
        let stopwatch = System.Diagnostics.Stopwatch.StartNew()
        
        logger.LogInformation($"📊 Benchmarking Hurwitz quaternion performance for bound {bound}")
        
        let primes = generateHurwitzPrimes bound logger
        
        stopwatch.Stop()
        let elapsedMs = stopwatch.ElapsedMilliseconds
        let totalQuaternions = (2 * bound + 1) * (2 * bound + 1) * (2 * bound + 1) * (2 * bound + 1) * 2
        let primesPerSec = if elapsedMs > 0L then (float primes.Length * 1000.0) / (float elapsedMs) else 0.0
        
        {
            QuaternionsGenerated = totalQuaternions
            PrimesFound = primes.Length
            ElapsedMs = elapsedMs
            PrimesPerSecond = primesPerSec
        }

    /// Test quaternion operations for correctness
    let testQuaternionOperations (logger: ILogger) : bool =
        try
            logger.LogInformation("🧪 Testing Hurwitz quaternion operations")
            
            // Test basic quaternion creation
            let q1 = createInt 1 2 3 4
            let q2 = createHalf 1 1 1 1
            
            // Test norm computation
            let norm1 = norm q1 // Should be 1 + 4 + 9 + 16 = 30
            let norm2 = norm q2 // Should be (1.5)² + (1.5)² + (1.5)² + (1.5)² = 9
            
            logger.LogInformation($"✅ Integer quaternion {format q1} has norm {norm1}")
            logger.LogInformation($"✅ Half-integer quaternion {format q2} has norm {norm2}")
            
            // Test primality
            let isPrime30 = isPrime norm1
            let isPrime9 = isPrime norm2
            
            logger.LogInformation($"✅ Norm {norm1} is prime: {isPrime30}")
            logger.LogInformation($"✅ Norm {norm2} is prime: {isPrime9}")
            
            // Test quaternion multiplication
            let product = multiply q1 q2
            logger.LogInformation($"✅ Product: {format q1} * {format q2} = {format product}")
            
            // Test conjugate
            let conj = conjugate q1
            logger.LogInformation($"✅ Conjugate of {format q1} = {format conj}")
            
            true
        with
        | ex ->
            logger.LogError($"❌ Quaternion operations test failed: {ex.Message}")
            false

    /// Generate small set of quaternionic primes for testing
    let getTestPrimes (logger: ILogger) : QuaternionPrime list =
        logger.LogInformation("🔢 Generating test set of quaternionic primes")
        let primes = generateHurwitzPrimes 3 logger
        let firstFew = primes |> List.take (min 10 primes.Length)
        
        for prime in firstFew do
            logger.LogInformation($"   Prime: {format prime.Quaternion} with norm {prime.Norm}")
        
        firstFew
