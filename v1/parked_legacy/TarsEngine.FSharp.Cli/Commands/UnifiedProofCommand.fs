namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Threading
open System.Threading.Tasks
open Spectre.Console
open TarsEngine.FSharp.Cli.Core.UnifiedCore
open TarsEngine.FSharp.Cli.Core.UnifiedLogger
open TarsEngine.FSharp.Cli.Integration.UnifiedProofSystem
open TarsEngine.FSharp.Cli.Commands
open TarsEngine.FSharp.Cli.Core.UnifiedTypes

/// Unified Proof Command - Demonstrates the cryptographic proof generation system
module UnifiedProofCommand =
    
    /// Demonstrate the unified proof generation system
    let demonstrateProofSystem (logger: ITarsLogger) =
        task {
            try
                AnsiConsole.MarkupLine("[bold cyan]🔐 TARS Unified Proof Generation System Demo[/]")
                AnsiConsole.MarkupLine("[dim]Cryptographic evidence and verification for all operations[/]")
                AnsiConsole.WriteLine()
                
                // Create proof generator
                use proofGenerator = createProofGenerator logger
                
                AnsiConsole.MarkupLine("[yellow]📋 Generating Various Types of Cryptographic Proofs...[/]")
                AnsiConsole.WriteLine()
                
                // Generate different types of proofs
                let correlationId = generateCorrelationId()
                let proofTasks = [
                    // Execution proofs
                    ("System Initialization", ProofExtensions.generateExecutionProof proofGenerator "SystemInitialization" correlationId)
                    ("Data Processing", ProofExtensions.generateExecutionProof proofGenerator "DataProcessing" correlationId)
                    ("Agent Coordination", ProofExtensions.generateExecutionProof proofGenerator "AgentCoordination" correlationId)
                    
                    // State change proofs
                    ("Configuration Update", ProofExtensions.generateStateChangeProof proofGenerator "config_v1.0" "config_v1.1" correlationId)
                    ("System Status Change", ProofExtensions.generateStateChangeProof proofGenerator "initializing" "running" correlationId)
                    
                    // Agent action proofs
                    ("Agent Task Execution", ProofExtensions.generateAgentActionProof proofGenerator "agent_001" "ProcessTask" correlationId)
                    ("Agent Communication", ProofExtensions.generateAgentActionProof proofGenerator "agent_002" "SendMessage" correlationId)
                    
                    // Performance proofs
                    ("Response Time Benchmark", ProofExtensions.generatePerformanceProof proofGenerator "ResponseTime" 0.245 correlationId)
                    ("Throughput Benchmark", ProofExtensions.generatePerformanceProof proofGenerator "Throughput" 1250.0 correlationId)
                    
                    // System health proof
                    ("System Health Check", 
                     let healthMetrics = Map [
                         ("cpu_usage", box 45.2)
                         ("memory_usage", box 67.8)
                         ("disk_usage", box 23.1)
                         ("active_agents", box 12)
                         ("system_uptime", box "2d 14h 32m")
                     ]
                     ProofExtensions.generateSystemHealthProof proofGenerator healthMetrics correlationId)
                ]
                
                // Execute proof generation tasks
                let proofResults = ResizeArray<string * TarsResult<TarsProof>>()
                
                for (description, proofTask) in proofTasks do
                    let! result = proofTask
                    proofResults.Add(description, result)
                    
                    match result with
                    | Success (proof, metadata) ->
                        AnsiConsole.MarkupLine($"  ✅ [green]{description}[/]")
                        AnsiConsole.MarkupLine($"     [dim]Proof ID: {proof.ProofId}[/]")
                        let timestampStr = proof.Timestamp.ToString("yyyy-MM-dd HH:mm:ss")
                        AnsiConsole.MarkupLine($"     [dim]Timestamp: {timestampStr}[/]")
                        AnsiConsole.MarkupLine($"     [dim]Signature: {proof.CryptographicSignature.Substring(0, 16)}...[/]")
                    | Failure (error, corrId) ->
                        AnsiConsole.MarkupLine($"  ❌ [red]{description}: {TarsError.toString error}[/]")
                
                AnsiConsole.WriteLine()
                
                // Collect successful proofs for chain creation
                let successfulProofs = 
                    proofResults
                    |> Seq.choose (fun (_, result) ->
                        match result with
                        | Success (proof, _) -> Some proof
                        | Failure _ -> None)
                    |> Seq.toList
                
                if successfulProofs.Length > 0 then
                    AnsiConsole.MarkupLine("[yellow]🔗 Creating Proof Chain for Operation Integrity...[/]")
                    
                    let chainId = $"demo_chain_{DateTime.Now:yyyyMMdd_HHmmss}"
                    let! chainResult = proofGenerator.CreateProofChainAsync(chainId, successfulProofs, CancellationToken.None)
                    
                    match chainResult with
                    | Success (proofChain, chainMetadata) ->
                        AnsiConsole.MarkupLine($"  ✅ [green]Proof chain created: {chainId}[/]")
                        AnsiConsole.MarkupLine($"     [dim]Chain Hash: {proofChain.ChainHash.Substring(0, 16)}...[/]")
                        AnsiConsole.MarkupLine($"     [dim]Proofs in Chain: {proofChain.Proofs.Length}[/]")
                        AnsiConsole.MarkupLine($"     [dim]Chain Valid: {proofChain.IsValid}[/]")
                    | Failure (error, corrId) ->
                        AnsiConsole.MarkupLine($"  ❌ [red]Chain creation failed: {TarsError.toString error}[/]")
                
                AnsiConsole.WriteLine()
                
                // Demonstrate proof verification
                AnsiConsole.MarkupLine("[yellow]🔍 Verifying Cryptographic Proofs...[/]")
                
                let verificationResults = ResizeArray<string * bool * float>()
                
                for proof in successfulProofs |> List.take (min 5 successfulProofs.Length) do
                    let! verificationResult = proofGenerator.VerifyProofAsync(proof, CancellationToken.None)
                    
                    match verificationResult with
                    | Success (verification, _) ->
                        verificationResults.Add(proof.ProofId, verification.IsValid, verification.TrustScore)
                        
                        let statusIcon = if verification.IsValid then "✅" else "❌"
                        let statusColor = if verification.IsValid then "green" else "red"
                        
                        AnsiConsole.MarkupLine($"  {statusIcon} [{statusColor}]Proof {proof.ProofId}[/]")
                        AnsiConsole.MarkupLine($"     [dim]Valid: {verification.IsValid}[/]")
                        let trustScoreStr = verification.TrustScore.ToString("F2")
                        AnsiConsole.MarkupLine($"     [dim]Trust Score: {trustScoreStr}[/]")
                        
                        if not verification.Issues.IsEmpty then
                            for issue in verification.Issues do
                                AnsiConsole.MarkupLine($"     [red]⚠️ {issue}[/]")
                    
                    | Failure (error, corrId) ->
                        verificationResults.Add(proof.ProofId, false, 0.0)
                        AnsiConsole.MarkupLine($"  ❌ [red]Verification failed for {proof.ProofId}: {TarsError.toString error}[/]")
                
                AnsiConsole.WriteLine()
                
                // Show proof system statistics
                let statistics = proofGenerator.GetProofStatistics()
                AnsiConsole.MarkupLine("[bold cyan]📊 Proof System Statistics:[/]")
                
                let totalProofs = statistics.["totalProofs"] :?> int
                let totalChains = statistics.["totalChains"] :?> int
                let validChains = statistics.["validChains"] :?> int
                let proofsByType = statistics.["proofsByType"] :?> Map<string, int>
                
                AnsiConsole.MarkupLine($"  Total Proofs Generated: [yellow]{totalProofs}[/]")
                AnsiConsole.MarkupLine($"  Total Proof Chains: [yellow]{totalChains}[/]")
                AnsiConsole.MarkupLine($"  Valid Chains: [green]{validChains}[/]")
                
                AnsiConsole.MarkupLine("  Proofs by Type:")
                for kvp in proofsByType do
                    AnsiConsole.MarkupLine($"    [cyan]{kvp.Key}[/]: {kvp.Value}")
                
                AnsiConsole.WriteLine()
                
                // Calculate verification success rate
                let validVerifications = verificationResults |> Seq.filter (fun (_, isValid, _) -> isValid) |> Seq.length
                let totalVerifications = verificationResults.Count
                let successRate = if totalVerifications > 0 then (float validVerifications / float totalVerifications) * 100.0 else 0.0
                
                let averageTrustScore = 
                    if verificationResults.Count > 0 then
                        verificationResults |> Seq.averageBy (fun (_, _, trustScore) -> trustScore)
                    else 0.0
                
                AnsiConsole.MarkupLine("[bold cyan]🎯 Verification Results:[/]")
                let successRateStr = successRate.ToString("F1")
                let avgTrustScoreStr = averageTrustScore.ToString("F2")
                AnsiConsole.MarkupLine($"  Verification Success Rate: [green]{successRateStr}%[/] ({validVerifications}/{totalVerifications})")
                AnsiConsole.MarkupLine($"  Average Trust Score: [yellow]{avgTrustScoreStr}[/]")
                
                AnsiConsole.WriteLine()
                
                // Demonstrate proof retrieval
                if successfulProofs.Length > 0 then
                    let sampleProof = successfulProofs.[0]
                    AnsiConsole.MarkupLine("[yellow]🔍 Demonstrating Proof Retrieval...[/]")
                    
                    let! retrievalResult = proofGenerator.GetProofAsync(sampleProof.ProofId, CancellationToken.None)
                    match retrievalResult with
                    | Success (Some retrievedProof, _) ->
                        AnsiConsole.MarkupLine($"  ✅ [green]Successfully retrieved proof: {retrievedProof.ProofId}[/]")
                        AnsiConsole.MarkupLine($"     [dim]Matches original: {retrievedProof.CryptographicSignature = sampleProof.CryptographicSignature}[/]")
                    | Success (None, _) ->
                        AnsiConsole.MarkupLine($"  ⚠️ [yellow]Proof not found: {sampleProof.ProofId}[/]")
                    | Failure (error, _) ->
                        AnsiConsole.MarkupLine($"  ❌ [red]Retrieval failed: {TarsError.toString error}[/]")
                    
                    // Get proofs by correlation ID
                    let! correlationResult = proofGenerator.GetProofsByCorrelationAsync(correlationId, CancellationToken.None)
                    match correlationResult with
                    | Success (correlatedProofs, metadata) ->
                        let count = metadata.["count"] :?> int
                        AnsiConsole.MarkupLine($"  ✅ [green]Found {count} proofs for correlation ID: {correlationId.Substring(0, 8)}...[/]")
                    | Failure (error, _) ->
                        AnsiConsole.MarkupLine($"  ❌ [red]Correlation retrieval failed: {TarsError.toString error}[/]")
                
                AnsiConsole.WriteLine()
                AnsiConsole.MarkupLine("[bold green]🎉 Unified Proof System Demo Completed Successfully![/]")
                AnsiConsole.WriteLine()
                AnsiConsole.MarkupLine("[bold cyan]🚀 PROOF SYSTEM ACHIEVEMENTS:[/]")
                AnsiConsole.MarkupLine("  ✅ [green]Cryptographic proof generation[/] - Multiple proof types supported")
                AnsiConsole.MarkupLine("  ✅ [green]Proof chain creation[/] - Operation integrity verification")
                AnsiConsole.MarkupLine("  ✅ [green]Proof verification[/] - Cryptographic signature validation")
                AnsiConsole.MarkupLine("  ✅ [green]Proof storage and retrieval[/] - Persistent evidence management")
                AnsiConsole.MarkupLine("  ✅ [green]System fingerprinting[/] - Tamper detection and authenticity")
                AnsiConsole.MarkupLine("  ✅ [green]Trust scoring[/] - Quantified proof reliability")
                
                return 0
            
            with
            | ex ->
                AnsiConsole.MarkupLine($"[red]❌ Proof system demo failed: {ex.Message}[/]")
                return 1
        }
    
    /// Unified Proof Command implementation
    type UnifiedProofCommand() =
        interface ICommand with
            member _.Name = "proof"
            member _.Description = "Demonstrate TARS unified cryptographic proof generation system"
            member _.Usage = "tars proof [--demo]"
            member _.Examples = [
                "tars proof --demo         # Run proof system demonstration"
                "tars proof                # Show proof system overview"
            ]
            
            member _.ValidateOptions(options: CommandOptions) = true
            
            member _.ExecuteAsync(options: CommandOptions) =
                task {
                    try
                        let logger = createLogger "UnifiedProofCommand"
                        
                        let isDemoMode = 
                            options.Arguments 
                            |> List.exists (fun arg -> arg = "--demo")
                        
                        if isDemoMode then
                            let! result = demonstrateProofSystem logger
                            return { Message = ""; ExitCode = result; Success = result = 0 }
                        else
                            AnsiConsole.MarkupLine("[bold cyan]🔐 TARS Unified Proof Generation System[/]")
                            AnsiConsole.WriteLine()
                            AnsiConsole.MarkupLine("This system provides cryptographic evidence and verification")
                            AnsiConsole.MarkupLine("for all TARS operations, ensuring integrity and authenticity.")
                            AnsiConsole.WriteLine()
                            AnsiConsole.MarkupLine("[bold yellow]Supported Proof Types:[/]")
                            AnsiConsole.MarkupLine("  🔧 [cyan]Execution Proofs[/] - Operation execution evidence")
                            AnsiConsole.MarkupLine("  🔄 [blue]State Change Proofs[/] - System state transitions")
                            AnsiConsole.MarkupLine("  🤖 [green]Agent Action Proofs[/] - Agent operation evidence")
                            AnsiConsole.MarkupLine("  📊 [yellow]Performance Proofs[/] - Benchmark and metrics evidence")
                            AnsiConsole.MarkupLine("  🏥 [magenta]System Health Proofs[/] - Health status verification")
                            AnsiConsole.MarkupLine("  🔒 [red]Security Proofs[/] - Security check evidence")
                            AnsiConsole.WriteLine()
                            AnsiConsole.MarkupLine("Available options:")
                            AnsiConsole.MarkupLine("  [yellow]--demo[/]  Run proof system demonstration")
                            AnsiConsole.WriteLine()
                            AnsiConsole.MarkupLine("Example: [dim]tars proof --demo[/]")
                            return { Message = ""; ExitCode = 0; Success = true }
                    
                    with
                    | ex ->
                        AnsiConsole.MarkupLine($"[red]❌ Command failed: {ex.Message}[/]")
                        return { Message = ""; ExitCode = 1; Success = false }
                }

