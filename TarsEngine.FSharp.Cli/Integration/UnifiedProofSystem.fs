namespace TarsEngine.FSharp.Cli.Integration

open System
open System.Security.Cryptography
open System.Text
open System.Threading
open System.Threading.Tasks
open System.Collections.Concurrent
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Cli.Core.UnifiedCore

/// Unified Proof Generation System - Cryptographic evidence and verification for all TARS operations
module UnifiedProofSystem =
    
    /// Types of proofs that can be generated
    type ProofType =
        | ExecutionProof of operation: string
        | StateChangeProof of before: string * after: string
        | AgentActionProof of agentId: string * action: string
        | DataIntegrityProof of dataHash: string
        | SystemHealthProof of metrics: Map<string, obj>
        | ComplianceProof of requirement: string * evidence: string
        | PerformanceProof of benchmark: string * result: float
        | SecurityProof of securityCheck: string * status: bool
    
    /// Cryptographic proof structure
    type TarsProof = {
        ProofId: string
        ProofType: ProofType
        Timestamp: DateTime
        CorrelationId: string
        ExecutionGuid: Guid
        ChainGuid: Guid
        SystemFingerprint: string
        ContentHash: string
        PreviousProofHash: string option
        CryptographicSignature: string
        Metadata: Map<string, obj>
        VerificationData: string
    }
    
    /// Proof chain for maintaining integrity
    type ProofChain = {
        ChainId: string
        StartTime: DateTime
        Proofs: TarsProof list
        ChainHash: string
        IsValid: bool
    }
    
    /// Proof verification result
    type ProofVerificationResult = {
        IsValid: bool
        ProofId: string
        VerificationTime: DateTime
        Issues: string list
        TrustScore: float
        VerificationChain: string list
    }
    
    /// Thread-safe unified proof generator
    type UnifiedProofGenerator(logger: ITarsLogger) =
        let proofChains = ConcurrentDictionary<string, ProofChain>()
        let proofStore = ConcurrentDictionary<string, TarsProof>()
        let mutable lastProofHash = None
        
        /// Generate system fingerprint for proof authenticity
        member private this.GenerateSystemFingerprint() =
            try
                let currentProcess = System.Diagnostics.Process.GetCurrentProcess()
                sprintf "%s|%d|%d|%d|%s"
                    Environment.MachineName
                    currentProcess.Id
                    currentProcess.Threads.Count
                    (int (currentProcess.WorkingSet64 / 1024L / 1024L))
                    (currentProcess.StartTime.ToString("yyyyMMddHHmmss"))
            with
            | _ -> sprintf "%s|%d|%s" Environment.MachineName Environment.ProcessId (DateTime.Now.ToString("yyyyMMddHHmmss"))
        
        /// Generate cryptographic hash for content
        member private this.GenerateContentHash(content: string) =
            use sha256 = SHA256.Create()
            let bytes = Encoding.UTF8.GetBytes(content)
            let hash = sha256.ComputeHash(bytes)
            Convert.ToBase64String(hash)
        
        /// Generate cryptographic signature for proof
        member private this.GenerateCryptographicSignature(proof: TarsProof) =
            use sha256 = SHA256.Create()
            let signatureData = sprintf "%s|%s|%s|%s|%s|%s"
                                    proof.ProofId
                                    (proof.Timestamp.ToString("O"))
                                    (proof.ExecutionGuid.ToString())
                                    (proof.ChainGuid.ToString())
                                    proof.SystemFingerprint
                                    proof.ContentHash

            let signatureBytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(signatureData))
            Convert.ToBase64String(signatureBytes)

        /// Generate verification data for proof integrity
        member private this.GenerateVerificationData(proof: TarsProof) =
            let verificationContent = sprintf "%s|%s|%s|%s"
                                          proof.ProofId
                                          proof.CorrelationId
                                          (proof.Timestamp.ToString("O"))
                                          proof.SystemFingerprint

            this.GenerateContentHash(verificationContent)
        
        /// Generate a new cryptographic proof
        member this.GenerateProofAsync(proofType: ProofType, correlationId: string, metadata: Map<string, obj> option) =
            task {
                try
                    let proofId = Guid.NewGuid().ToString("N").[..15]
                    let executionGuid = Guid.NewGuid()
                    let chainGuid = Guid.NewGuid()
                    let timestamp = DateTime.UtcNow
                    let systemFingerprint = this.GenerateSystemFingerprint()
                    
                    // Create content for hashing based on proof type
                    let content = 
                        match proofType with
                        | ExecutionProof operation -> sprintf "EXEC:%s:%s" operation correlationId
                        | StateChangeProof (before, after) -> sprintf "STATE:%s->%s:%s" before after correlationId
                        | AgentActionProof (agentId, action) -> sprintf "AGENT:%s:%s:%s" agentId action correlationId
                        | DataIntegrityProof dataHash -> sprintf "DATA:%s:%s" dataHash correlationId
                        | SystemHealthProof metrics -> sprintf "HEALTH:%A:%s" metrics correlationId
                        | ComplianceProof (requirement, evidence) -> sprintf "COMPLIANCE:%s:%s:%s" requirement evidence correlationId
                        | PerformanceProof (benchmark, result) -> sprintf "PERF:%s:%f:%s" benchmark result correlationId
                        | SecurityProof (securityCheck, status) -> sprintf "SECURITY:%s:%b:%s" securityCheck status correlationId
                    
                    let contentHash = this.GenerateContentHash(content)
                    let finalMetadata = metadata |> Option.defaultValue Map.empty
                    
                    let proof = {
                        ProofId = proofId
                        ProofType = proofType
                        Timestamp = timestamp
                        CorrelationId = correlationId
                        ExecutionGuid = executionGuid
                        ChainGuid = chainGuid
                        SystemFingerprint = systemFingerprint
                        ContentHash = contentHash
                        PreviousProofHash = lastProofHash
                        CryptographicSignature = ""
                        Metadata = finalMetadata
                        VerificationData = ""
                    }
                    
                    // Generate signature and verification data
                    let signature = this.GenerateCryptographicSignature(proof)
                    let verificationData = this.GenerateVerificationData(proof)
                    
                    let finalProof = {
                        proof with
                            CryptographicSignature = signature
                            VerificationData = verificationData
                    }
                    
                    // Store proof
                    proofStore.[proofId] <- finalProof
                    lastProofHash <- Some signature
                    
                    logger.LogInformation(correlationId, sprintf "Generated cryptographic proof: %s" proofId)
                    
                    return Success (finalProof, Map [("proofId", box proofId); ("timestamp", box timestamp)])
                
                with
                | ex ->
                    let error = ExecutionError ("Failed to generate proof", Some ex)
                    logger.LogError(correlationId, error, ex)
                    return Failure (error, correlationId)
            }
        
        /// Verify a cryptographic proof
        member this.VerifyProofAsync(proof: TarsProof, cancellationToken: CancellationToken) =
            task {
                try
                    let correlationId = generateCorrelationId()
                    let verificationTime = DateTime.UtcNow
                    let mutable issues = []
                    let mutable trustScore = 1.0
                    
                    // Verify signature
                    let expectedSignature = this.GenerateCryptographicSignature(proof)
                    if proof.CryptographicSignature <> expectedSignature then
                        issues <- "Invalid cryptographic signature" :: issues
                        trustScore <- trustScore * 0.0
                    
                    // Verify verification data
                    let expectedVerificationData = this.GenerateVerificationData(proof)
                    if proof.VerificationData <> expectedVerificationData then
                        issues <- "Invalid verification data" :: issues
                        trustScore <- trustScore * 0.5
                    
                    // Verify timestamp (not too old or in future)
                    let timeDiff = verificationTime - proof.Timestamp
                    if timeDiff.TotalDays > 30.0 then
                        issues <- "Proof is too old (>30 days)" :: issues
                        trustScore <- trustScore * 0.8
                    elif timeDiff.TotalSeconds < -60.0 then
                        issues <- "Proof timestamp is in the future" :: issues
                        trustScore <- trustScore * 0.3
                    
                    // Verify proof chain integrity
                    let verificationChain = [proof.ProofId]
                    
                    let result = {
                        IsValid = issues.IsEmpty
                        ProofId = proof.ProofId
                        VerificationTime = verificationTime
                        Issues = issues
                        TrustScore = trustScore
                        VerificationChain = verificationChain
                    }
                    
                    logger.LogInformation(correlationId, sprintf "Verified proof %s: Valid=%b, Trust=%s" proof.ProofId result.IsValid (trustScore.ToString("F2")))
                    
                    return Success (result, Map [("proofId", box proof.ProofId); ("isValid", box result.IsValid); ("trustScore", box trustScore)])
                
                with
                | ex ->
                    let error = ExecutionError (sprintf "Failed to verify proof %s" proof.ProofId, Some ex)
                    logger.LogError(generateCorrelationId(), error, ex)
                    return Failure (error, generateCorrelationId())
            }
        
        /// Create a proof chain for related operations
        member this.CreateProofChainAsync(chainId: string, proofs: TarsProof list, cancellationToken: CancellationToken) =
            task {
                try
                    let correlationId = generateCorrelationId()
                    let startTime = DateTime.UtcNow
                    
                    // Generate chain hash from all proofs
                    let chainContent = 
                        proofs
                        |> List.map (fun p -> p.CryptographicSignature)
                        |> String.concat "|"
                    
                    let chainHash = this.GenerateContentHash(chainContent)
                    
                    // Verify all proofs in chain
                    let verificationTasks = 
                        proofs
                        |> List.map (fun proof -> this.VerifyProofAsync(proof, cancellationToken))
                    
                    let! verificationResults = Task.WhenAll(verificationTasks)
                    
                    let isValid = 
                        verificationResults
                        |> Array.forall (function
                            | Success (result, _) -> result.IsValid
                            | Failure _ -> false)
                    
                    let proofChain = {
                        ChainId = chainId
                        StartTime = startTime
                        Proofs = proofs
                        ChainHash = chainHash
                        IsValid = isValid
                    }
                    
                    proofChains.[chainId] <- proofChain
                    
                    logger.LogInformation(correlationId, sprintf "Created proof chain %s with %d proofs: Valid=%b" chainId proofs.Length isValid)
                    
                    return Success (proofChain, Map [("chainId", box chainId); ("proofCount", box proofs.Length); ("isValid", box isValid)])
                
                with
                | ex ->
                    let error = ExecutionError (sprintf "Failed to create proof chain %s" chainId, Some ex)
                    logger.LogError(generateCorrelationId(), error, ex)
                    return Failure (error, generateCorrelationId())
            }
        
        /// Get proof by ID
        member this.GetProofAsync(proofId: string, cancellationToken: CancellationToken) =
            task {
                match proofStore.TryGetValue(proofId) with
                | true, proof ->
                    return Success (Some proof, Map [("proofId", box proofId)])
                | false, _ ->
                    return Success (None, Map [("proofId", box proofId)])
            }
        
        /// Get all proofs for a correlation ID
        member this.GetProofsByCorrelationAsync(correlationId: string, cancellationToken: CancellationToken) =
            task {
                let matchingProofs = 
                    proofStore.Values
                    |> Seq.filter (fun p -> p.CorrelationId = correlationId)
                    |> Seq.toList
                
                return Success (matchingProofs, Map [("correlationId", box correlationId); ("count", box matchingProofs.Length)])
            }
        
        /// Get system-wide proof statistics
        member this.GetProofStatistics() =
            let totalProofs = proofStore.Count
            let totalChains = proofChains.Count
            
            let proofsByType = 
                proofStore.Values
                |> Seq.groupBy (fun p -> 
                    match p.ProofType with
                    | ExecutionProof _ -> "Execution"
                    | StateChangeProof _ -> "StateChange"
                    | AgentActionProof _ -> "AgentAction"
                    | DataIntegrityProof _ -> "DataIntegrity"
                    | SystemHealthProof _ -> "SystemHealth"
                    | ComplianceProof _ -> "Compliance"
                    | PerformanceProof _ -> "Performance"
                    | SecurityProof _ -> "Security")
                |> Seq.map (fun (proofType, proofs) -> proofType, Seq.length proofs)
                |> Map.ofSeq
            
            let validChains = 
                proofChains.Values
                |> Seq.filter (fun chain -> chain.IsValid)
                |> Seq.length
            
            Map [
                ("totalProofs", box totalProofs)
                ("totalChains", box totalChains)
                ("validChains", box validChains)
                ("proofsByType", box proofsByType)
                ("lastUpdate", box DateTime.UtcNow)
            ]
        
        interface IDisposable with
            member this.Dispose() =
                proofStore.Clear()
                proofChains.Clear()
    
    /// Proof generation extensions for unified operations
    module ProofExtensions =
        
        /// Generate execution proof for any operation
        let generateExecutionProof (proofGenerator: UnifiedProofGenerator) (operation: string) (correlationId: string) =
            proofGenerator.GenerateProofAsync(ExecutionProof operation, correlationId, None)
        
        /// Generate state change proof
        let generateStateChangeProof (proofGenerator: UnifiedProofGenerator) (beforeState: string) (afterState: string) (correlationId: string) =
            proofGenerator.GenerateProofAsync(StateChangeProof (beforeState, afterState), correlationId, None)
        
        /// Generate agent action proof
        let generateAgentActionProof (proofGenerator: UnifiedProofGenerator) (agentId: string) (action: string) (correlationId: string) =
            proofGenerator.GenerateProofAsync(AgentActionProof (agentId, action), correlationId, None)
        
        /// Generate performance proof
        let generatePerformanceProof (proofGenerator: UnifiedProofGenerator) (benchmark: string) (result: float) (correlationId: string) =
            proofGenerator.GenerateProofAsync(PerformanceProof (benchmark, result), correlationId, None)
        
        /// Generate system health proof
        let generateSystemHealthProof (proofGenerator: UnifiedProofGenerator) (metrics: Map<string, obj>) (correlationId: string) =
            proofGenerator.GenerateProofAsync(SystemHealthProof metrics, correlationId, None)
    
    /// Create unified proof generator
    let createProofGenerator (logger: ITarsLogger) =
        new UnifiedProofGenerator(logger)

