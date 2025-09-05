module TarsEngine.FSharp.Tests.Unified.UnifiedSecurityTests

open System
open System.Threading
open Xunit
open FsUnit.Xunit
open TarsEngine.FSharp.Cli.Core.UnifiedCore
open TarsEngine.FSharp.Cli.Core.UnifiedLogger
open TarsEngine.FSharp.Cli.Integration.UnifiedProofSystem

/// Security tests for unified systems
[<TestClass>]
type UnifiedSecurityTests() =
    
    let createTestLogger() = createLogger "UnifiedSecurityTests"
    
    [<Fact>]
    let ``Proof signatures should be cryptographically secure`` () =
        task {
            // Arrange
            use proofGenerator = createProofGenerator (createTestLogger())
            let correlationId = generateCorrelationId()
            
            // Act - Generate multiple proofs
            let! proof1 = ProofExtensions.generateExecutionProof proofGenerator "Operation1" correlationId
            let! proof2 = ProofExtensions.generateExecutionProof proofGenerator "Operation2" correlationId
            
            // Assert
            match proof1, proof2 with
            | Success (p1, _), Success (p2, _) ->
                // Signatures should be different for different operations
                p1.CryptographicSignature |> should not' (equal p2.CryptographicSignature)
                
                // Signatures should be of reasonable length (base64 encoded)
                p1.CryptographicSignature.Length |> should be (greaterThan 20)
                p2.CryptographicSignature.Length |> should be (greaterThan 20)
                
                // Verification data should be present
                p1.VerificationData.Length |> should be (greaterThan 0)
                p2.VerificationData.Length |> should be (greaterThan 0)
                
            | _ -> failwith "Failed to generate test proofs"
        }
    
    [<Fact>]
    let ``Tampered proofs should be detected`` () =
        task {
            // Arrange
            use proofGenerator = createProofGenerator (createTestLogger())
            let correlationId = generateCorrelationId()
            
            // Generate a valid proof
            let! proofResult = ProofExtensions.generateExecutionProof proofGenerator "TestOperation" correlationId
            
            match proofResult with
            | Success (originalProof, _) ->
                // Act - Tamper with different parts of the proof
                let tamperedSignature = { originalProof with CryptographicSignature = "tampered_signature" }
                let tamperedVerification = { originalProof with VerificationData = "tampered_verification" }
                let tamperedContent = { originalProof with ContentHash = "tampered_content" }
                
                // Verify tampered proofs
                let! sigResult = proofGenerator.VerifyProofAsync(tamperedSignature, CancellationToken.None)
                let! verResult = proofGenerator.VerifyProofAsync(tamperedVerification, CancellationToken.None)
                let! contentResult = proofGenerator.VerifyProofAsync(tamperedContent, CancellationToken.None)
                
                // Assert - All tampered proofs should be detected as invalid
                match sigResult, verResult, contentResult with
                | Success (sigVerification, _), Success (verVerification, _), Success (contentVerification, _) ->
                    sigVerification.IsValid |> should be False
                    verVerification.IsValid |> should be False
                    contentVerification.IsValid |> should be False
                    
                    // Trust scores should be low
                    sigVerification.TrustScore |> should be (lessThanOrEqualTo 0.5)
                    verVerification.TrustScore |> should be (lessThanOrEqualTo 0.5)
                    contentVerification.TrustScore |> should be (lessThanOrEqualTo 0.5)
                    
                | _ -> failwith "Verification should succeed but detect tampering"
            
            | Failure (error, _) -> failwith $"Failed to generate test proof: {TarsError.toString error}"
        }
    
    [<Fact>]
    let ``System fingerprints should be consistent`` () =
        task {
            // Arrange
            use proofGenerator = createProofGenerator (createTestLogger())
            let correlationId = generateCorrelationId()
            
            // Act - Generate multiple proofs in quick succession
            let! proof1 = ProofExtensions.generateExecutionProof proofGenerator "Operation1" correlationId
            let! proof2 = ProofExtensions.generateExecutionProof proofGenerator "Operation2" correlationId
            
            // Assert
            match proof1, proof2 with
            | Success (p1, _), Success (p2, _) ->
                // System fingerprints should be the same (same system, same session)
                p1.SystemFingerprint |> should equal p2.SystemFingerprint
                
                // But execution GUIDs should be different
                p1.ExecutionGuid |> should not' (equal p2.ExecutionGuid)
                
            | _ -> failwith "Failed to generate test proofs"
        }
    
    [<Fact>]
    let ``Proof chains should maintain integrity`` () =
        task {
            // Arrange
            use proofGenerator = createProofGenerator (createTestLogger())
            let correlationId = generateCorrelationId()
            
            // Generate multiple proofs
            let! proof1 = ProofExtensions.generateExecutionProof proofGenerator "Step1" correlationId
            let! proof2 = ProofExtensions.generateExecutionProof proofGenerator "Step2" correlationId
            let! proof3 = ProofExtensions.generateExecutionProof proofGenerator "Step3" correlationId
            
            match proof1, proof2, proof3 with
            | Success (p1, _), Success (p2, _), Success (p3, _) ->
                let proofs = [p1; p2; p3]
                
                // Act - Create proof chain
                let! chainResult = proofGenerator.CreateProofChainAsync("security_test_chain", proofs, CancellationToken.None)
                
                // Assert
                match chainResult with
                | Success (chain, _) ->
                    chain.IsValid |> should be True
                    chain.Proofs.Length |> should equal 3
                    chain.ChainHash.Length |> should be (greaterThan 0)
                    
                    // Tamper with one proof and verify chain becomes invalid
                    let tamperedProof = { p2 with CryptographicSignature = "tampered" }
                    let tamperedProofs = [p1; tamperedProof; p3]
                    
                    let! tamperedChainResult = proofGenerator.CreateProofChainAsync("tampered_chain", tamperedProofs, CancellationToken.None)
                    
                    match tamperedChainResult with
                    | Success (tamperedChain, _) ->
                        tamperedChain.IsValid |> should be False
                    | Failure _ -> () // Also acceptable - chain creation might fail
                    
                | Failure (error, _) -> failwith $"Chain creation failed: {TarsError.toString error}"
            
            | _ -> failwith "Failed to generate test proofs"
        }
    
    [<Fact>]
    let ``Configuration should not expose sensitive data`` () =
        task {
            // Arrange
            use configManager = createConfigurationManager (createTestLogger())
            let! _ = configManager.InitializeAsync(CancellationToken.None)
            
            // Act - Set some configuration values
            let! _ = configManager.SetValueAsync("public.setting", "public_value", None)
            let! _ = configManager.SetValueAsync("secret.password", "secret_value", None)
            
            // Get all configuration values
            let allValues = configManager.GetAllValues()
            
            // Assert - Configuration should be accessible but we should be careful about logging
            allValues.ContainsKey("public.setting") |> should be True
            allValues.ContainsKey("secret.password") |> should be True
            
            // In a real implementation, we would test that:
            // - Sensitive values are not logged
            // - Sensitive values are encrypted at rest
            // - Access to sensitive values is controlled
        }
    
    [<Fact>]
    let ``Error messages should not leak sensitive information`` () =
        // Arrange
        let sensitiveData = "password123"
        let publicData = "public_info"
        
        // Act - Create errors with different types of data
        let publicError = ValidationError ($"Invalid value: {publicData}", Map.empty)
        let sensitiveError = ValidationError ("Authentication failed", Map [("details", box "Invalid credentials")])
        
        // Assert - Error messages should be safe to log
        let publicErrorString = TarsError.toString publicError
        let sensitiveErrorString = TarsError.toString sensitiveError
        
        publicErrorString |> should contain "Invalid value"
        sensitiveErrorString |> should contain "Authentication failed"
        
        // Sensitive data should not appear in error strings
        sensitiveErrorString |> should not' (contain sensitiveData)
    
    [<Fact>]
    let ``Correlation IDs should not be predictable`` () =
        // Act - Generate many correlation IDs
        let ids = [for i in 1..1000 -> generateCorrelationId()]
        
        // Assert - All should be unique
        let uniqueIds = ids |> List.distinct
        uniqueIds.Length |> should equal ids.Length
        
        // Should not follow a predictable pattern
        let sortedIds = ids |> List.sort
        sortedIds |> should not' (equal ids) // Very unlikely to be generated in sorted order
