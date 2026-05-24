module TarsEngine.FSharp.Tests.Unified.UnifiedProofSystemTests

open System
open System.Threading
open Xunit
open FsUnit.Xunit
open TarsEngine.FSharp.Cli.Core.UnifiedCore
open TarsEngine.FSharp.Cli.Core.UnifiedLogger
open TarsEngine.FSharp.Cli.Integration.UnifiedProofSystem

/// Tests for the Unified Proof System
[<TestClass>]
type UnifiedProofSystemTests() =
    
    let createTestLogger() = createLogger "UnifiedProofSystemTests"
    
    [<Fact>]
    let ``Should generate execution proof successfully`` () =
        task {
            // Arrange
            use proofGenerator = createProofGenerator (createTestLogger())
            let correlationId = generateCorrelationId()
            
            // Act
            let! result = ProofExtensions.generateExecutionProof proofGenerator "TestOperation" correlationId
            
            // Assert
            match result with
            | Success (proof, metadata) ->
                proof.ProofType |> should be (ofCase <@ ExecutionProof @>)
                proof.CorrelationId |> should equal correlationId
                proof.ProofId.Length |> should be (greaterThan 0)
                proof.CryptographicSignature.Length |> should be (greaterThan 0)
                proof.VerificationData.Length |> should be (greaterThan 0)
                metadata.ContainsKey("proofId") |> should be True
            | Failure (error, _) -> failwith $"Proof generation failed: {TarsError.toString error}"
        }
    
    [<Fact>]
    let ``Should generate state change proof successfully`` () =
        task {
            // Arrange
            use proofGenerator = createProofGenerator (createTestLogger())
            let correlationId = generateCorrelationId()
            
            // Act
            let! result = ProofExtensions.generateStateChangeProof proofGenerator "state_v1" "state_v2" correlationId
            
            // Assert
            match result with
            | Success (proof, metadata) ->
                match proof.ProofType with
                | StateChangeProof (before, after) ->
                    before |> should equal "state_v1"
                    after |> should equal "state_v2"
                | _ -> failwith "Expected StateChangeProof"
                
                proof.CorrelationId |> should equal correlationId
                proof.CryptographicSignature.Length |> should be (greaterThan 0)
            | Failure (error, _) -> failwith $"Proof generation failed: {TarsError.toString error}"
        }
    
    [<Fact>]
    let ``Should generate agent action proof successfully`` () =
        task {
            // Arrange
            use proofGenerator = createProofGenerator (createTestLogger())
            let correlationId = generateCorrelationId()
            
            // Act
            let! result = ProofExtensions.generateAgentActionProof proofGenerator "agent_001" "ProcessTask" correlationId
            
            // Assert
            match result with
            | Success (proof, metadata) ->
                match proof.ProofType with
                | AgentActionProof (agentId, action) ->
                    agentId |> should equal "agent_001"
                    action |> should equal "ProcessTask"
                | _ -> failwith "Expected AgentActionProof"
                
                proof.CorrelationId |> should equal correlationId
            | Failure (error, _) -> failwith $"Proof generation failed: {TarsError.toString error}"
        }
    
    [<Fact>]
    let ``Should generate performance proof successfully`` () =
        task {
            // Arrange
            use proofGenerator = createProofGenerator (createTestLogger())
            let correlationId = generateCorrelationId()
            
            // Act
            let! result = ProofExtensions.generatePerformanceProof proofGenerator "ResponseTime" 0.245 correlationId
            
            // Assert
            match result with
            | Success (proof, metadata) ->
                match proof.ProofType with
                | PerformanceProof (benchmark, result) ->
                    benchmark |> should equal "ResponseTime"
                    Math.Abs(result - 0.245) |> should be (lessThan 0.00001)
                | _ -> failwith "Expected PerformanceProof"
                
                proof.CorrelationId |> should equal correlationId
            | Failure (error, _) -> failwith $"Proof generation failed: {TarsError.toString error}"
        }
    
    [<Fact>]
    let ``Should generate system health proof successfully`` () =
        task {
            // Arrange
            use proofGenerator = createProofGenerator (createTestLogger())
            let correlationId = generateCorrelationId()
            let metrics = Map [
                ("cpu_usage", box 45.2)
                ("memory_usage", box 67.8)
                ("active_agents", box 12)
            ]
            
            // Act
            let! result = ProofExtensions.generateSystemHealthProof proofGenerator metrics correlationId
            
            // Assert
            match result with
            | Success (proof, metadata) ->
                match proof.ProofType with
                | SystemHealthProof healthMetrics ->
                    healthMetrics.["cpu_usage"] :?> float |> should equal 45.2
                    healthMetrics.["memory_usage"] :?> float |> should equal 67.8
                    healthMetrics.["active_agents"] :?> int |> should equal 12
                | _ -> failwith "Expected SystemHealthProof"
                
                proof.CorrelationId |> should equal correlationId
            | Failure (error, _) -> failwith $"Proof generation failed: {TarsError.toString error}"
        }
    
    [<Fact>]
    let ``Should verify proof successfully`` () =
        task {
            // Arrange
            use proofGenerator = createProofGenerator (createTestLogger())
            let correlationId = generateCorrelationId()
            
            // Generate a proof first
            let! proofResult = ProofExtensions.generateExecutionProof proofGenerator "TestOperation" correlationId
            
            match proofResult with
            | Success (proof, _) ->
                // Act
                let! verificationResult = proofGenerator.VerifyProofAsync(proof, CancellationToken.None)
                
                // Assert
                match verificationResult with
                | Success (verification, metadata) ->
                    verification.IsValid |> should be True
                    verification.ProofId |> should equal proof.ProofId
                    verification.TrustScore |> should be (greaterThan 0.0)
                    verification.Issues |> should be Empty
                    metadata.ContainsKey("proofId") |> should be True
                | Failure (error, _) -> failwith $"Proof verification failed: {TarsError.toString error}"
            
            | Failure (error, _) -> failwith $"Proof generation failed: {TarsError.toString error}"
        }
    
    [<Fact>]
    let ``Should create proof chain successfully`` () =
        task {
            // Arrange
            use proofGenerator = createProofGenerator (createTestLogger())
            let correlationId = generateCorrelationId()
            
            // Generate multiple proofs
            let! proof1Result = ProofExtensions.generateExecutionProof proofGenerator "Operation1" correlationId
            let! proof2Result = ProofExtensions.generateExecutionProof proofGenerator "Operation2" correlationId
            let! proof3Result = ProofExtensions.generateExecutionProof proofGenerator "Operation3" correlationId
            
            match proof1Result, proof2Result, proof3Result with
            | Success (proof1, _), Success (proof2, _), Success (proof3, _) ->
                let proofs = [proof1; proof2; proof3]
                
                // Act
                let! chainResult = proofGenerator.CreateProofChainAsync("test_chain", proofs, CancellationToken.None)
                
                // Assert
                match chainResult with
                | Success (proofChain, metadata) ->
                    proofChain.ChainId |> should equal "test_chain"
                    proofChain.Proofs.Length |> should equal 3
                    proofChain.IsValid |> should be True
                    proofChain.ChainHash.Length |> should be (greaterThan 0)
                    metadata.ContainsKey("chainId") |> should be True
                | Failure (error, _) -> failwith $"Proof chain creation failed: {TarsError.toString error}"
            
            | _ -> failwith "Failed to generate test proofs"
        }
    
    [<Fact>]
    let ``Should retrieve proof by ID`` () =
        task {
            // Arrange
            use proofGenerator = createProofGenerator (createTestLogger())
            let correlationId = generateCorrelationId()
            
            // Generate a proof first
            let! proofResult = ProofExtensions.generateExecutionProof proofGenerator "TestOperation" correlationId
            
            match proofResult with
            | Success (originalProof, _) ->
                // Act
                let! retrievalResult = proofGenerator.GetProofAsync(originalProof.ProofId, CancellationToken.None)
                
                // Assert
                match retrievalResult with
                | Success (Some retrievedProof, metadata) ->
                    retrievedProof.ProofId |> should equal originalProof.ProofId
                    retrievedProof.CryptographicSignature |> should equal originalProof.CryptographicSignature
                    retrievedProof.CorrelationId |> should equal originalProof.CorrelationId
                    metadata.ContainsKey("proofId") |> should be True
                | Success (None, _) -> failwith "Proof not found"
                | Failure (error, _) -> failwith $"Proof retrieval failed: {TarsError.toString error}"
            
            | Failure (error, _) -> failwith $"Proof generation failed: {TarsError.toString error}"
        }
    
    [<Fact>]
    let ``Should retrieve proofs by correlation ID`` () =
        task {
            // Arrange
            use proofGenerator = createProofGenerator (createTestLogger())
            let correlationId = generateCorrelationId()
            
            // Generate multiple proofs with same correlation ID
            let! proof1Result = ProofExtensions.generateExecutionProof proofGenerator "Operation1" correlationId
            let! proof2Result = ProofExtensions.generateExecutionProof proofGenerator "Operation2" correlationId
            let! proof3Result = ProofExtensions.generateExecutionProof proofGenerator "Operation3" correlationId
            
            match proof1Result, proof2Result, proof3Result with
            | Success _, Success _, Success _ ->
                // Act
                let! retrievalResult = proofGenerator.GetProofsByCorrelationAsync(correlationId, CancellationToken.None)
                
                // Assert
                match retrievalResult with
                | Success (proofs, metadata) ->
                    proofs.Length |> should equal 3
                    proofs |> List.forall (fun p -> p.CorrelationId = correlationId) |> should be True
                    metadata.ContainsKey("correlationId") |> should be True
                    metadata.["count"] :?> int |> should equal 3
                | Failure (error, _) -> failwith $"Proof retrieval failed: {TarsError.toString error}"
            
            | _ -> failwith "Failed to generate test proofs"
        }
    
    [<Fact>]
    let ``Should provide proof statistics`` () =
        task {
            // Arrange
            use proofGenerator = createProofGenerator (createTestLogger())
            let correlationId = generateCorrelationId()
            
            // Generate some proofs
            let! _ = ProofExtensions.generateExecutionProof proofGenerator "Operation1" correlationId
            let! _ = ProofExtensions.generateStateChangeProof proofGenerator "state1" "state2" correlationId
            let! _ = ProofExtensions.generateAgentActionProof proofGenerator "agent1" "action1" correlationId
            
            // Act
            let statistics = proofGenerator.GetProofStatistics()
            
            // Assert
            statistics.ContainsKey("totalProofs") |> should be True
            statistics.ContainsKey("totalChains") |> should be True
            statistics.ContainsKey("proofsByType") |> should be True
            statistics.ContainsKey("lastUpdate") |> should be True
            
            let totalProofs = statistics.["totalProofs"] :?> int
            totalProofs |> should be (greaterThanOrEqualTo 3)
            
            let proofsByType = statistics.["proofsByType"] :?> Map<string, int>
            proofsByType.ContainsKey("Execution") |> should be True
            proofsByType.ContainsKey("StateChange") |> should be True
            proofsByType.ContainsKey("AgentAction") |> should be True
        }
    
    [<Fact>]
    let ``Should handle concurrent proof generation`` () =
        task {
            // Arrange
            use proofGenerator = createProofGenerator (createTestLogger())
            let correlationId = generateCorrelationId()
            
            // Act - Generate proofs concurrently
            let tasks = [
                for i in 1..10 do
                    yield ProofExtensions.generateExecutionProof proofGenerator $"Operation{i}" correlationId
            ]
            
            let! results = System.Threading.Tasks.Task.WhenAll(tasks)
            
            // Assert
            results |> Array.forall (function | Success _ -> true | Failure _ -> false) |> should be True
            
            // All proofs should have unique IDs
            let proofIds = 
                results 
                |> Array.choose (function | Success (proof, _) -> Some proof.ProofId | Failure _ -> None)
            
            let uniqueIds = proofIds |> Array.distinct
            uniqueIds.Length |> should equal proofIds.Length
        }
    
    [<Fact>]
    let ``Should detect tampered proof`` () =
        task {
            // Arrange
            use proofGenerator = createProofGenerator (createTestLogger())
            let correlationId = generateCorrelationId()
            
            // Generate a proof
            let! proofResult = ProofExtensions.generateExecutionProof proofGenerator "TestOperation" correlationId
            
            match proofResult with
            | Success (originalProof, _) ->
                // Tamper with the proof
                let tamperedProof = { originalProof with CryptographicSignature = "tampered_signature" }
                
                // Act
                let! verificationResult = proofGenerator.VerifyProofAsync(tamperedProof, CancellationToken.None)
                
                // Assert
                match verificationResult with
                | Success (verification, _) ->
                    verification.IsValid |> should be False
                    verification.Issues |> should not' (be Empty)
                    verification.TrustScore |> should equal 0.0
                | Failure (error, _) -> failwith $"Verification should succeed but detect tampering: {TarsError.toString error}"
            
            | Failure (error, _) -> failwith $"Proof generation failed: {TarsError.toString error}"
        }
    
    [<Fact>]
    let ``Should handle all proof types`` () =
        task {
            // Arrange
            use proofGenerator = createProofGenerator (createTestLogger())
            let correlationId = generateCorrelationId()
            
            // Act - Generate all types of proofs
            let! executionProof = proofGenerator.GenerateProofAsync(ExecutionProof "test", correlationId, None)
            let! stateProof = proofGenerator.GenerateProofAsync(StateChangeProof ("old", "new"), correlationId, None)
            let! agentProof = proofGenerator.GenerateProofAsync(AgentActionProof ("agent", "action"), correlationId, None)
            let! dataProof = proofGenerator.GenerateProofAsync(DataIntegrityProof "hash123", correlationId, None)
            let! healthProof = proofGenerator.GenerateProofAsync(SystemHealthProof (Map [("cpu", box 50.0)]), correlationId, None)
            let! complianceProof = proofGenerator.GenerateProofAsync(ComplianceProof ("req", "evidence"), correlationId, None)
            let! perfProof = proofGenerator.GenerateProofAsync(PerformanceProof ("bench", 1.23), correlationId, None)
            let! securityProof = proofGenerator.GenerateProofAsync(SecurityProof ("check", true), correlationId, None)
            
            // Assert
            let results = [executionProof; stateProof; agentProof; dataProof; healthProof; complianceProof; perfProof; securityProof]
            results |> List.forall (function | Success _ -> true | Failure _ -> false) |> should be True
            
            // Verify each proof type
            match results with
            | [Success (exec, _); Success (state, _); Success (agent, _); Success (data, _); 
               Success (health, _); Success (compliance, _); Success (perf, _); Success (security, _)] ->
                
                exec.ProofType |> should be (ofCase <@ ExecutionProof @>)
                state.ProofType |> should be (ofCase <@ StateChangeProof @>)
                agent.ProofType |> should be (ofCase <@ AgentActionProof @>)
                data.ProofType |> should be (ofCase <@ DataIntegrityProof @>)
                health.ProofType |> should be (ofCase <@ SystemHealthProof @>)
                compliance.ProofType |> should be (ofCase <@ ComplianceProof @>)
                perf.ProofType |> should be (ofCase <@ PerformanceProof @>)
                security.ProofType |> should be (ofCase <@ SecurityProof @>)
            
            | _ -> failwith "Not all proof types generated successfully"
        }
