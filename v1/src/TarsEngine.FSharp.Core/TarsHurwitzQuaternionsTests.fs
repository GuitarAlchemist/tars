// ================================================
// 🧪 TARS Hurwitz Quaternions Tests
// ================================================
// Comprehensive test suite for Hurwitz quaternion operations

namespace TarsEngine.FSharp.Core.Tests

open System
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.TarsHurwitzQuaternions

module TarsHurwitzQuaternionsTests =

    /// Create a test logger that outputs to console
    let createTestLogger () =
        let loggerFactory = LoggerFactory.Create(fun builder ->
            builder.AddConsole() |> ignore
        )
        loggerFactory.CreateLogger<unit>()

    /// Test basic quaternion creation and properties
    let testQuaternionCreation (logger: ILogger) : bool =
        try
            logger.LogInformation("🧪 Testing quaternion creation")
            
            // Test integer quaternion
            let q1 = createInt 1 2 3 4
            assert (q1.A = 1.0 && q1.B = 2.0 && q1.C = 3.0 && q1.D = 4.0 && not q1.IsHalf)
            
            // Test half-integer quaternion
            let q2 = createHalf 1 1 1 1
            assert (q2.A = 1.0 && q2.B = 1.0 && q2.C = 1.0 && q2.D = 1.0 && q2.IsHalf)
            
            // Test general creation
            let q3 = create 2.5 -1.5 0.0 3.5 false
            assert (q3.A = 2.5 && q3.B = -1.5 && q3.C = 0.0 && q3.D = 3.5 && not q3.IsHalf)
            
            logger.LogInformation("✅ Quaternion creation tests passed")
            true
        with
        | ex ->
            logger.LogError($"❌ Quaternion creation test failed: {ex.Message}")
            false

    /// Test norm computation
    let testNormComputation (logger: ILogger) : bool =
        try
            logger.LogInformation("🧪 Testing norm computation")
            
            // Test integer quaternion norm: (1,2,3,4) -> 1² + 2² + 3² + 4² = 30
            let q1 = createInt 1 2 3 4
            let norm1 = norm q1
            assert (norm1 = 30)
            
            // Test half-integer quaternion norm: (1.5,1.5,1.5,1.5) -> 4 * (1.5)² = 9
            let q2 = createHalf 1 1 1 1
            let norm2 = norm q2
            assert (norm2 = 9)
            
            // Test zero quaternion
            let q3 = createInt 0 0 0 0
            let norm3 = norm q3
            assert (norm3 = 0)
            
            // Test unit quaternion: (1,0,0,0) -> 1
            let q4 = createInt 1 0 0 0
            let norm4 = norm q4
            assert (norm4 = 1)
            
            logger.LogInformation("✅ Norm computation tests passed")
            true
        with
        | ex ->
            logger.LogError($"❌ Norm computation test failed: {ex.Message}")
            false

    /// Test primality checking
    let testPrimalityChecking (logger: ILogger) : bool =
        try
            logger.LogInformation("🧪 Testing primality checking")
            
            // Test known primes
            assert (isPrime 2)
            assert (isPrime 3)
            assert (isPrime 5)
            assert (isPrime 7)
            assert (isPrime 11)
            assert (isPrime 13)
            assert (isPrime 17)
            assert (isPrime 19)
            assert (isPrime 23)
            
            // Test known composites
            assert (not (isPrime 1))
            assert (not (isPrime 4))
            assert (not (isPrime 6))
            assert (not (isPrime 8))
            assert (not (isPrime 9))
            assert (not (isPrime 10))
            assert (not (isPrime 12))
            assert (not (isPrime 15))
            assert (not (isPrime 16))
            
            // Test edge cases
            assert (not (isPrime 0))
            assert (not (isPrime -1))
            
            logger.LogInformation("✅ Primality checking tests passed")
            true
        with
        | ex ->
            logger.LogError($"❌ Primality checking test failed: {ex.Message}")
            false

    /// Test quaternion multiplication
    let testQuaternionMultiplication (logger: ILogger) : bool =
        try
            logger.LogInformation("🧪 Testing quaternion multiplication")
            
            // Test multiplication with unit quaternion
            let unit = createInt 1 0 0 0
            let q = createInt 2 3 4 5
            let result1 = multiply unit q
            let result2 = multiply q unit
            
            // Multiplication by unit should preserve the quaternion (approximately)
            assert (abs(result1.A - q.A) < 1e-10)
            assert (abs(result1.B - q.B) < 1e-10)
            assert (abs(result1.C - q.C) < 1e-10)
            assert (abs(result1.D - q.D) < 1e-10)
            
            // Test i * i = -1 (where i = (0,1,0,0))
            let i = createInt 0 1 0 0
            let i_squared = multiply i i
            assert (abs(i_squared.A - (-1.0)) < 1e-10)
            assert (abs(i_squared.B) < 1e-10)
            assert (abs(i_squared.C) < 1e-10)
            assert (abs(i_squared.D) < 1e-10)
            
            // Test j * j = -1 (where j = (0,0,1,0))
            let j = createInt 0 0 1 0
            let j_squared = multiply j j
            assert (abs(j_squared.A - (-1.0)) < 1e-10)
            
            // Test k * k = -1 (where k = (0,0,0,1))
            let k = createInt 0 0 0 1
            let k_squared = multiply k k
            assert (abs(k_squared.A - (-1.0)) < 1e-10)
            
            logger.LogInformation("✅ Quaternion multiplication tests passed")
            true
        with
        | ex ->
            logger.LogError($"❌ Quaternion multiplication test failed: {ex.Message}")
            false

    /// Test quaternion conjugate
    let testQuaternionConjugate (logger: ILogger) : bool =
        try
            logger.LogInformation("🧪 Testing quaternion conjugate")
            
            let q = createInt 1 2 3 4
            let conj = conjugate q
            
            // Conjugate should negate i, j, k components
            assert (conj.A = q.A)
            assert (conj.B = -q.B)
            assert (conj.C = -q.C)
            assert (conj.D = -q.D)
            assert (conj.IsHalf = q.IsHalf)
            
            // Test conjugate of conjugate is original
            let double_conj = conjugate conj
            assert (double_conj.A = q.A)
            assert (double_conj.B = q.B)
            assert (double_conj.C = q.C)
            assert (double_conj.D = q.D)
            
            logger.LogInformation("✅ Quaternion conjugate tests passed")
            true
        with
        | ex ->
            logger.LogError($"❌ Quaternion conjugate test failed: {ex.Message}")
            false

    /// Test quaternionic prime detection
    let testQuaternionicPrimes (logger: ILogger) : bool =
        try
            logger.LogInformation("🧪 Testing quaternionic prime detection")
            
            // Test some known cases
            let q1 = createInt 1 1 0 0  // Norm = 2 (prime)
            assert (isQuaternionPrime q1)
            
            let q2 = createInt 1 1 1 0  // Norm = 3 (prime)
            assert (isQuaternionPrime q2)
            
            let q3 = createInt 2 0 0 0  // Norm = 4 (not prime)
            assert (not (isQuaternionPrime q3))
            
            let q4 = createInt 0 0 0 0  // Norm = 0 (not prime)
            assert (not (isQuaternionPrime q4))
            
            let q5 = createInt 1 0 0 0  // Norm = 1 (not prime)
            assert (not (isQuaternionPrime q5))
            
            logger.LogInformation("✅ Quaternionic prime detection tests passed")
            true
        with
        | ex ->
            logger.LogError($"❌ Quaternionic prime detection test failed: {ex.Message}")
            false

    /// Test quaternion formatting
    let testQuaternionFormatting (logger: ILogger) : bool =
        try
            logger.LogInformation("🧪 Testing quaternion formatting")
            
            let q1 = createInt 1 2 3 4
            let formatted1 = format q1
            assert (formatted1.Contains("1") && formatted1.Contains("2i") && formatted1.Contains("3j") && formatted1.Contains("4k"))
            
            let q2 = createHalf 1 1 1 1
            let formatted2 = format q2
            assert (formatted2.Contains("+½"))
            
            logger.LogInformation($"✅ Formatted quaternions: {formatted1}, {formatted2}")
            true
        with
        | ex ->
            logger.LogError($"❌ Quaternion formatting test failed: {ex.Message}")
            false

    /// Test performance benchmarking
    let testPerformanceBenchmark (logger: ILogger) : bool =
        try
            logger.LogInformation("🧪 Testing performance benchmark")
            
            let performance = benchmarkPerformance 2 logger
            
            assert (performance.QuaternionsGenerated > 0)
            assert (performance.PrimesFound >= 0)
            assert (performance.ElapsedMs >= 0L)
            assert (performance.PrimesPerSecond >= 0.0)
            
            logger.LogInformation($"✅ Benchmark: {performance.PrimesFound} primes from {performance.QuaternionsGenerated} quaternions")
            logger.LogInformation($"✅ Performance: {performance.PrimesPerSecond:F0} primes/second")
            
            true
        with
        | ex ->
            logger.LogError($"❌ Performance benchmark test failed: {ex.Message}")
            false

    /// Run all Hurwitz quaternion tests
    let runAllTests (logger: ILogger) : bool =
        logger.LogInformation("🚀 Running comprehensive Hurwitz quaternion tests")
        
        let tests = [
            ("Quaternion Creation", testQuaternionCreation)
            ("Norm Computation", testNormComputation)
            ("Primality Checking", testPrimalityChecking)
            ("Quaternion Multiplication", testQuaternionMultiplication)
            ("Quaternion Conjugate", testQuaternionConjugate)
            ("Quaternionic Primes", testQuaternionicPrimes)
            ("Quaternion Formatting", testQuaternionFormatting)
            ("Performance Benchmark", testPerformanceBenchmark)
        ]
        
        let mutable allPassed = true
        let mutable passedCount = 0
        
        for (testName, testFunc) in tests do
            try
                if testFunc logger then
                    passedCount <- passedCount + 1
                    logger.LogInformation($"✅ {testName}: PASSED")
                else
                    allPassed <- false
                    logger.LogError($"❌ {testName}: FAILED")
            with
            | ex ->
                allPassed <- false
                logger.LogError($"❌ {testName}: EXCEPTION - {ex.Message}")
        
        logger.LogInformation($"🎯 Test Results: {passedCount}/{tests.Length} tests passed")
        
        if allPassed then
            logger.LogInformation("🎉 All Hurwitz quaternion tests PASSED!")
        else
            logger.LogError("💥 Some Hurwitz quaternion tests FAILED!")
        
        allPassed
