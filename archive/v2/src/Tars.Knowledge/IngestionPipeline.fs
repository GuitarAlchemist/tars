namespace Tars.Knowledge

open System
open Tars.Core
open Tars.Llm

/// Orchestrates the full ingestion pipeline: Fetch → Extract → Verify → Write
module IngestionPipeline =

    type IngestionStats = {
        ArticleTitle: string
        SourceUrl: Uri
        ProposalsExtracted: int
        ProposalsAccepted: int
        ProposalsDenied: int
        ContradictionsFound: int
        DurationMs: float
    }

    type IngestionResult = {
        Stats: IngestionStats
        AcceptedBeliefs: Tars.Knowledge.Belief list
        RejectedProposals: (ProposedAssertion * string) list
    }

    /// Process a single proposal through verification and assertion
    let private processProposal 
        (ledger: KnowledgeLedger) 
        (verifier: VerifierAgent)
        (url: string)
        (proposal: ProposedAssertion) 
        : Async<Microsoft.FSharp.Core.Result<Tars.Knowledge.Belief, string>> =
        async {
            let log = Logging.withCategory "Ingestion"
            
            // Step 1: Verify
            let! decision = verifier.Verify(proposal)
            
            match decision with
            | Accepted confidence ->
                // Convert to Belief
                let predicate = 
                    match proposal.Predicate.ToLowerInvariant() with
                    | "is" -> RelationType.IsA
                    | "has" -> RelationType.HasProperty
                    | "part_of" | "part of" -> RelationType.PartOf
                    | "invented" | "created" -> RelationType.DerivedFrom
                    | "located_in" | "located in" -> RelationType.PartOf
                    | "causes" -> RelationType.Causes
                    | "prevents" -> RelationType.Prevents
                    | "supports" -> RelationType.Supports
                    | custom -> RelationType.Custom custom

                let provenance = Provenance.FromExternal(Uri(url), None, confidence)
                let belief = Tars.Knowledge.Belief.create proposal.Subject predicate proposal.Object provenance
                
                // Step 2: Assert
                let! assertResult = ledger.Assert(belief, AgentId.System) |> Async.AwaitTask
                
                return 
                    match assertResult with
                    | Ok _ -> Ok belief
                    | Error err -> Error err
                    
            | Denied reason ->
                return Error reason
                
            | Conflict (conflictingBelief, score) ->
                let reason = $"Conflicts with: {conflictingBelief.TripleString} (score: {score:F2})"
                return Error reason
        }

    /// Run the complete ingestion pipeline
    let ingest 
        (llmService: ILlmService) 
        (ledger: KnowledgeLedger) 
        (verifier: VerifierAgent)
        (url: string) 
        : Async<Microsoft.FSharp.Core.Result<IngestionResult, string>> =
        async {
            let sw = System.Diagnostics.Stopwatch.StartNew()
            let log = Logging.withCategory "Ingestion"

            try
                // Step 1: Fetch Wikipedia content
                log.Info $"📥 Fetching content from: {url}"
                
                use client = new System.Net.Http.HttpClient()
                let! htmlResponse = client.GetStringAsync(url) |> Async.AwaitTask
                
                // Extract title from URL
                let uri = Uri(url)
                let title = uri.Segments |> Array.last |> fun (s: string) -> s.Replace("_", " ").TrimEnd('/')
                
                log.Info $"📄 Article: {title}"
                log.Info $"   Length: {htmlResponse.Length} chars"

                // Simple text extraction (remove HTML tags)
                let textContent = 
                    System.Text.RegularExpressions.Regex.Replace(htmlResponse, "<[^>]+>", " ")
                    |> fun s -> System.Text.RegularExpressions.Regex.Replace(s, @"\s+", " ")
                    |> fun (s: string) -> s.Trim()

                // Step 2: Extract beliefs using LLM
                log.Info "🧠 Extracting beliefs via LLM..."
                let! proposals = WikipediaExtractor.extractFromArticle llmService title textContent
                
                log.Info $"   Extracted {proposals.Length} candidate beliefs"

                // Step 3: Process each proposal (verify + assert)
                log.Info "✔️ Processing beliefs..."
                
                let! results = 
                    proposals
                    |> List.map (processProposal ledger verifier url)
                    |> Async.Sequential
                
                // Separate successes from failures
                let accepted = 
                    results 
                    |> Array.choose (function | Ok belief -> Some belief | Error _ -> None)
                    |> Array.toList
                    
                let denied =
                    results
                    |> Array.mapi (fun i result ->
                        match result with
                        | Error reason -> Some (proposals.[i], reason)
                        | Ok _ -> None)
                    |> Array.choose id
                    |> Array.toList
                
                let contradictions =
                    denied
                    |> List.filter (fun (_, reason) -> 
                        reason.Contains("Inconsistent") || reason.Contains("Conflicts"))
                    |> List.length

                sw.Stop()

                let stats = {
                    ArticleTitle = title
                    SourceUrl = Uri(url)
                    ProposalsExtracted = proposals.Length
                    ProposalsAccepted = accepted.Length
                    ProposalsDenied = denied.Length
                    ContradictionsFound = contradictions
                    DurationMs = sw.Elapsed.TotalMilliseconds
                }

                log.Info ""
                log.Info $"✅ Ingestion complete in {stats.DurationMs:F0}ms"
                log.Info $"   📊 Extracted: {stats.ProposalsExtracted}"
                log.Info $"   ✔️ Accepted: {stats.ProposalsAccepted}"
                log.Info $"   ❌ Denied: {stats.ProposalsDenied}"
                log.Info $"   ⚠️ Contradictions: {stats.ContradictionsFound}"

                return Ok {
                    Stats = stats
                    AcceptedBeliefs = accepted
                    RejectedProposals = denied
                }

            with ex ->
                sw.Stop()
                log.Error("Ingestion failed", ex)
                return Error ex.Message
        }
