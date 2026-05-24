/// <summary>
/// Phase 11: Cognitive Grounding & Production Intelligence
/// =========================================================
/// Ensures TARS grounds its reasoning in verified facts and maintains
/// calibrated confidence. This is the "reality check" layer that prevents
/// hallucination and ensures reliable production-grade reasoning.
///
/// Key Components:
/// - Fact Verification Pipeline: LLM claims → Ledger check
/// - Confidence Calibration: Track and adjust prediction accuracy
/// - Source Attribution: Every claim links to evidence
/// - Hallucination Detection: Identify and recover from confabulation
/// - Grounded Reasoning: Constrain LLM output to known facts
/// </summary>
namespace Tars.Cortex

open System
open System.Collections.Concurrent
open System.Text.RegularExpressions
open Tars.Core
open Tars.Knowledge
open Tars.Llm
open Tars.Llm.LlmService

/// Cognitive grounding for reliable, verifiable reasoning
module CognitiveGrounding =

    // =========================================================================
    // Core Types
    // =========================================================================

    /// A claim extracted from LLM output
    type Claim = {
        Id: Guid
        Statement: string
        Source: string
        Confidence: float
        Timestamp: DateTime
    }

    /// Verification status of a claim
    type VerificationStatus =
        | Verified of evidence: string * confidence: float
        | Contradicted of counterEvidence: string
        | Unverifiable of reason: string
        | Pending

    /// A verified or refuted claim
    type VerifiedClaim = {
        Claim: Claim
        Status: VerificationStatus
        VerifiedAt: DateTime
        Sources: string list
    }

    /// Calibration record for confidence tracking
    type CalibrationRecord = {
        PredictedConfidence: float
        WasCorrect: bool
        Domain: string
        Timestamp: DateTime
    }

    /// Hallucination detection result
    type HallucinationAnalysis = {
        IsLikelyHallucination: bool
        Confidence: float
        Indicators: string list
        Suggestion: string
    }

    /// A grounded response with citations
    type GroundedResponse = {
        Answer: string
        Claims: VerifiedClaim list
        OverallConfidence: float
        Citations: Map<int, string>  // Citation number -> source
        Warnings: string list
    }

    // =========================================================================
    // Claim Extraction
    // =========================================================================

    /// Extracts factual claims from LLM text
    let extractClaims (text: string) : Claim list =
        // Split into sentences and filter for factual claims
        let sentences = 
            Regex.Split(text, @"(?<=[.!?])\s+")
            |> Array.filter (fun s -> s.Length > 10)
        
        sentences
        |> Array.mapi (fun i sentence ->
            // Filter for declarative statements (likely factual claims)
            let isFactual = 
                not (sentence.StartsWith("I think")) &&
                not (sentence.StartsWith("Perhaps")) &&
                not (sentence.StartsWith("Maybe")) &&
                not (sentence.Contains("?")) &&
                (sentence.Contains(" is ") || 
                 sentence.Contains(" are ") ||
                 sentence.Contains(" was ") ||
                 sentence.Contains(" were ") ||
                 sentence.Contains(" has ") ||
                 sentence.Contains(" have "))
            
            if isFactual then
                Some {
                    Id = Guid.NewGuid()
                    Statement = sentence.Trim()
                    Source = "LLM"
                    Confidence = 0.7 // Default confidence
                    Timestamp = DateTime.UtcNow
                }
            else None)
        |> Array.choose id
        |> Array.toList

    // =========================================================================
    // Fact Verification Pipeline
    // =========================================================================

    /// Verifies a claim against the knowledge ledger
    let verifyClaim (ledger: KnowledgeLedger) (claim: Claim) : VerifiedClaim =
        // Search for related beliefs in the ledger
        let related = ledger.GetRelevantBeliefs(claim.Statement, limit = 5)
        
        if related.IsEmpty then
            {
                Claim = claim
                Status = Unverifiable "No related facts found in knowledge ledger"
                VerifiedAt = DateTime.UtcNow
                Sources = []
            }
        else
            // Check for supporting or contradicting evidence
            let supporting = 
                related 
                |> List.filter (fun b -> b.Confidence > 0.6)
                |> List.tryHead
            
            let contradicting =
                related
                |> List.tryFind (fun b -> 
                    b.Object.Value.ToLowerInvariant().Contains("not") ||
                    b.Object.Value.ToLowerInvariant().Contains("false"))
            
            match contradicting, supporting with
            | Some c, _ ->
                let sourceStr = sprintf "%A" c.Provenance.Source
                {
                    Claim = claim
                    Status = Contradicted (sprintf "Contradicted by: %s %A %s" c.Subject.Value c.Predicate c.Object.Value)
                    VerifiedAt = DateTime.UtcNow
                    Sources = [sourceStr]
                }
            | None, Some s ->
                let sourceStr = sprintf "%A" s.Provenance.Source
                {
                    Claim = claim
                    Status = Verified (sprintf "Supported by: %s %A %s" s.Subject.Value s.Predicate s.Object.Value, s.Confidence)
                    VerifiedAt = DateTime.UtcNow
                    Sources = [sourceStr]
                }
            | None, None ->
                {
                    Claim = { claim with Confidence = 0.5 } // Lower confidence
                    Status = Unverifiable "Related facts found but no clear support or contradiction"
                    VerifiedAt = DateTime.UtcNow
                    Sources = related |> List.map (fun b -> sprintf "%A" b.Provenance.Source)
                }

    /// Verify all claims in a text against the ledger
    let verifyText (ledger: KnowledgeLedger) (text: string) : VerifiedClaim list =
        text
        |> extractClaims
        |> List.map (verifyClaim ledger)

    // =========================================================================
    // Confidence Calibration
    // =========================================================================

    /// Thread-safe calibration history
    type CalibrationTracker() =
        let history = ConcurrentQueue<CalibrationRecord>()
        let maxHistory = 1000
        
        /// Record a prediction and its outcome
        member _.Record(predictedConfidence: float, wasCorrect: bool, domain: string) =
            history.Enqueue({
                PredictedConfidence = predictedConfidence
                WasCorrect = wasCorrect
                Domain = domain
                Timestamp = DateTime.UtcNow
            })
            
            // Trim history if too large
            while history.Count > maxHistory do
                history.TryDequeue() |> ignore
        
        /// Calculate calibration error (how well confidence predicts accuracy)
        member _.CalibrationError(domain: string option) =
            let records = 
                history.ToArray()
                |> Array.filter (fun r -> domain.IsNone || Some r.Domain = domain)
            
            if records.Length < 10 then
                None // Not enough data
            else
                // Group by confidence buckets
                let buckets = 
                    records
                    |> Array.groupBy (fun r -> Math.Round(r.PredictedConfidence * 10.0) / 10.0)
                    |> Array.map (fun (bucket, items) ->
                        let accuracy = items |> Array.filter (fun r -> r.WasCorrect) |> Array.length |> float |> fun x -> x / float items.Length
                        abs(bucket - accuracy))
                
                Some (buckets |> Array.average)
        
        /// Get adjusting factor for a domain based on calibration
        member this.GetAdjustmentFactor(domain: string) =
            match this.CalibrationError(Some domain) with
            | None -> 1.0 // No data, no adjustment
            | Some error when error < 0.1 -> 1.0 // Well calibrated
            | Some error when error < 0.2 -> 0.9 // Slightly overconfident
            | Some error -> 0.8 // Significantly overconfident

    // =========================================================================
    // Hallucination Detection
    // =========================================================================

    /// Indicators that suggest hallucination
    let private hallucinationIndicators = [
        // Very specific numbers without source
        @"\b\d{4,}\b", "Large specific numbers without context"
        // Made-up quotes
        "\"[^\"]{50,}\"", "Long direct quotes"
        // Overly specific dates in distant past
        @"in \d{3,4} (BC|BCE|AD|CE)", "Very specific historical dates"
        // Fake-sounding names
        @"Dr\. [A-Z][a-z]+ [A-Z][a-z]+son", "Potentially fabricated expert names"
        // URLs that look made up
        @"https?://[a-z]+\.[a-z]{2,4}/\w{20,}", "Suspiciously specific URLs"
    ]

    /// Analyze text for hallucination indicators
    let detectHallucination (claims: Claim list) (verifiedClaims: VerifiedClaim list) : HallucinationAnalysis =
        let indicators = ResizeArray<string>()
        
        // Check verification results
        let unverifiedRatio = 
            let unverified = verifiedClaims |> List.filter (fun vc -> 
                match vc.Status with Unverifiable _ -> true | _ -> false) |> List.length
            float unverified / float (max 1 verifiedClaims.Length)
        
        if unverifiedRatio > 0.7 then
            indicators.Add("High ratio of unverifiable claims")
        
        // Check for contradictions
        let contradictions = 
            verifiedClaims 
            |> List.filter (fun vc -> match vc.Status with Contradicted _ -> true | _ -> false)
            |> List.length
        
        if contradictions > 0 then
            indicators.Add($"Found {contradictions} contradicted claims")
        
        // Check for hallucination patterns in text
        let allText = claims |> List.map (fun c -> c.Statement) |> String.concat " "
        for pattern, description in hallucinationIndicators do
            if Regex.IsMatch(allText, pattern) then
                indicators.Add(description)
        
        // Calculate overall hallucination likelihood
        let score = 
            (if unverifiedRatio > 0.7 then 0.3 else 0.0) +
            (float contradictions * 0.2) +
            (float indicators.Count * 0.1)
            |> min 1.0
        
        {
            IsLikelyHallucination = score > 0.5
            Confidence = score
            Indicators = indicators |> Seq.toList
            Suggestion = 
                if score > 0.5 then "Consider regenerating with stricter grounding constraints"
                elif score > 0.3 then "Verify key claims before using"
                else "Response appears reasonably grounded"
        }

    // =========================================================================
    // Grounded Response Generation
    // =========================================================================

    /// Generate a response grounded in the knowledge ledger
    let groundedQuery (llm: ILlmService) (ledger: KnowledgeLedger) (query: string) =
        task {
            // 1. Get relevant facts from ledger
            let relevantBeliefs = ledger.GetRelevantBeliefs(query, limit = 10)
            
            let factsContext = 
                if relevantBeliefs.IsEmpty then ""
                else
                    let facts = 
                        relevantBeliefs
                        |> List.mapi (fun i b -> 
                            sprintf "[%d] %s %A %s (Confidence: %.0f%%)" (i+1) b.Subject.Value b.Predicate b.Object.Value (b.Confidence * 100.0))
                        |> String.concat "\n"
                    sprintf "\n\n[VERIFIED FACTS FROM KNOWLEDGE LEDGER]\n%s\n" facts
            
            // 2. Create grounded prompt
            let systemPrompt = """You are a factual assistant that only states what you know to be true.
Rules:
1. Base your answer ONLY on the verified facts provided
2. Use [N] citations when referencing facts (e.g., "Paris is the capital [1]")
3. If the facts don't cover something, say "I don't have verified information about..."
4. Never make up facts or statistics
5. Express uncertainty when appropriate"""

            let userPrompt = query + factsContext
            
            // 3. Generate response
            let request = {
                LlmRequest.Default with
                    SystemPrompt = Some systemPrompt
                    Messages = [{ Role = Role.User; Content = userPrompt }]
                    Temperature = Some 0.3 // Lower for factuality
            }
            
            let! response = llm.CompleteAsync request
            
            // 4. Verify claims in response
            let claims = extractClaims response.Text
            let verifiedClaims = claims |> List.map (verifyClaim ledger)
            
            // 5. Analyze for hallucination
            let hallAnalysis = detectHallucination claims verifiedClaims
            
            // 6. Build citations map
            let citations = 
                relevantBeliefs
                |> List.mapi (fun i b -> i + 1, sprintf "%A" b.Provenance.Source)
                |> Map.ofList
            
            // 7. Calculate overall confidence
            let verifiedCount = 
                verifiedClaims 
                |> List.filter (fun vc -> match vc.Status with Verified _ -> true | _ -> false)
                |> List.length
            let overallConfidence = 
                if claims.IsEmpty then 0.8
                else float verifiedCount / float claims.Length
            
            // 8. Build warnings
            let warnings = ResizeArray<string>()
            if hallAnalysis.IsLikelyHallucination then
                warnings.Add("⚠️ Response may contain hallucinated content")
            for vc in verifiedClaims do
                match vc.Status with
                | Contradicted evidence ->
                    warnings.Add($"⚠️ Claim contradicted: {vc.Claim.Statement}")
                | _ -> ()
            
            return {
                Answer = response.Text
                Claims = verifiedClaims
                OverallConfidence = overallConfidence
                Citations = citations
                Warnings = warnings |> Seq.toList
            }
        }

    // =========================================================================
    // Source Attribution
    // =========================================================================

    /// Attribute claims in text to their sources
    let attributeSources (text: string) (verifiedClaims: VerifiedClaim list) : string =
        let mutable annotated = text
        
        for vc in verifiedClaims do
            let sources = vc.Sources |> String.concat ", "
            let attribution = 
                match vc.Status with
                | Verified (_, conf) -> $" [Source: {sources}, Confidence: {conf:P0}]"
                | Contradicted _ -> $" [⚠️ CONTRADICTED by: {sources}]"
                | Unverifiable reason -> $" [Unverified: {reason}]"
                | Pending -> " [Pending verification]"
            
            // Find and annotate the claim
            let claimText = vc.Claim.Statement
            if annotated.Contains(claimText) then
                annotated <- annotated.Replace(claimText, claimText + attribution)
        
        annotated

    // =========================================================================
    // Production Monitoring
    // =========================================================================

    /// Metrics for monitoring grounding quality
    type GroundingMetrics = {
        TotalClaims: int
        VerifiedClaims: int
        ContradictedClaims: int
        UnverifiableClaims: int
        AverageConfidence: float
        HallucinationRate: float
    }

    /// Calculate metrics from a batch of grounded responses
    let calculateMetrics (responses: GroundedResponse list) : GroundingMetrics =
        let allClaims = responses |> List.collect (fun r -> r.Claims)
        
        let verified = allClaims |> List.filter (fun vc -> match vc.Status with Verified _ -> true | _ -> false)
        let contradicted = allClaims |> List.filter (fun vc -> match vc.Status with Contradicted _ -> true | _ -> false)
        let unverifiable = allClaims |> List.filter (fun vc -> match vc.Status with Unverifiable _ -> true | _ -> false)
        
        {
            TotalClaims = allClaims.Length
            VerifiedClaims = verified.Length
            ContradictedClaims = contradicted.Length
            UnverifiableClaims = unverifiable.Length
            AverageConfidence = responses |> List.averageBy (fun r -> r.OverallConfidence)
            HallucinationRate = 
                if responses.IsEmpty then 0.0
                else
                    let hallucinated = responses |> List.filter (fun r -> r.Warnings |> List.exists (fun w -> w.Contains("hallucinated")))
                    float hallucinated.Length / float responses.Length
        }
