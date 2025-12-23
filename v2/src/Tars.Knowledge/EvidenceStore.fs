/// TARS Evidence Store - Raw evidence before verification
/// "The Internet never writes beliefs directly. It only produces evidence candidates."
namespace Tars.Knowledge

open System
open System.Collections.Generic
open System.Security.Cryptography
open System.Text

/// Evidence store for managing evidence candidates
type EvidenceStore() =
    let candidates = Dictionary<Guid, EvidenceCandidate>()
    let byUrl = Dictionary<string, Guid>()
    let byStatus = Dictionary<EvidenceStatus, HashSet<Guid>>()

    /// Compute content hash
    static member ComputeHash(content: string) =
        use sha = SHA256.Create()
        let bytes = Encoding.UTF8.GetBytes(content)
        let hash = sha.ComputeHash(bytes)
        Convert.ToHexString(hash).ToLowerInvariant()

    /// Add a new evidence candidate
    member this.Add(url: Uri, content: string, segments: string list) =
        let hash = EvidenceStore.ComputeHash(content)
        let id = Guid.NewGuid()

        let candidate =
            { Id = id
              SourceUrl = url
              ContentHash = hash
              FetchedAt = DateTime.UtcNow
              RawContent = content
              Segments = segments
              ProposedAssertions = []
              Status = Pending
              VerifiedAt = None
              VerifiedBy = None
              RejectionReason = None }

        candidates.[id] <- candidate
        byUrl.[url.ToString()] <- id

        if not (byStatus.ContainsKey(Pending)) then
            byStatus.[Pending] <- HashSet<Guid>()

        byStatus.[Pending].Add(id) |> ignore

        id

    /// Add proposed assertions to a candidate
    member this.AddProposals(candidateId: Guid, proposals: ProposedAssertion list) =
        match candidates.TryGetValue(candidateId) with
        | true, c ->
            candidates.[candidateId] <-
                { c with
                    ProposedAssertions = c.ProposedAssertions @ proposals }

            true
        | false, _ -> false

    /// Get a candidate by ID
    member this.Get(id: Guid) : EvidenceCandidate option =
        match candidates.TryGetValue(id) with
        | true, c -> Some c
        | false, _ -> None

    /// Get candidate by URL
    member this.GetByUrl(url: string) : EvidenceCandidate option =
        match byUrl.TryGetValue(url) with
        | true, id -> this.Get(id)
        | false, _ -> None

    /// Get all candidates by status
    member this.GetByStatus(status: EvidenceStatus) : EvidenceCandidate seq =
        match byStatus.TryGetValue(status) with
        | true, ids -> ids |> Seq.choose (fun id -> this.Get(id))
        | false, _ -> Seq.empty

    /// Get all pending candidates
    member this.GetPending() = this.GetByStatus(Pending)

    /// Mark candidate as verified
    member this.Verify(id: Guid, verifierId: AgentId) =
        match candidates.TryGetValue(id) with
        | true, c ->
            // Update status index
            if byStatus.ContainsKey(c.Status) then
                byStatus.[c.Status].Remove(id) |> ignore

            let updated =
                { c with
                    Status = Verified
                    VerifiedAt = Some DateTime.UtcNow
                    VerifiedBy = Some verifierId }

            candidates.[id] <- updated

            if not (byStatus.ContainsKey(Verified)) then
                byStatus.[Verified] <- HashSet<Guid>()

            byStatus.[Verified].Add(id) |> ignore

            true
        | false, _ -> false

    /// Mark candidate as rejected
    member this.Reject(id: Guid, reason: string) =
        match candidates.TryGetValue(id) with
        | true, c ->
            // Update status index
            if byStatus.ContainsKey(c.Status) then
                byStatus.[c.Status].Remove(id) |> ignore

            let updated =
                { c with
                    Status = Rejected
                    RejectionReason = Some reason }

            candidates.[id] <- updated

            if not (byStatus.ContainsKey(Rejected)) then
                byStatus.[Rejected] <- HashSet<Guid>()

            byStatus.[Rejected].Add(id) |> ignore

            true
        | false, _ -> false

    /// Mark candidate as conflicting
    member this.MarkConflicting(id: Guid) =
        match candidates.TryGetValue(id) with
        | true, c ->
            if byStatus.ContainsKey(c.Status) then
                byStatus.[c.Status].Remove(id) |> ignore

            let updated = { c with Status = Conflicting }
            candidates.[id] <- updated

            if not (byStatus.ContainsKey(Conflicting)) then
                byStatus.[Conflicting] <- HashSet<Guid>()

            byStatus.[Conflicting].Add(id) |> ignore

            true
        | false, _ -> false

    /// Check if URL already fetched (based on hash)
    member this.IsDuplicate(url: string) = byUrl.ContainsKey(url)

    /// Get statistics
    member this.Stats() =
        let statusCounts =
            byStatus |> Seq.map (fun kvp -> (kvp.Key, kvp.Value.Count)) |> Seq.toList

        let totalProposals =
            candidates.Values |> Seq.sumBy (fun c -> c.ProposedAssertions.Length)

        {| TotalCandidates = candidates.Count
           ByStatus = statusCounts
           TotalProposals = totalProposals
           UniqueUrls = byUrl.Count |}

    /// Clear all data
    member this.Clear() =
        candidates.Clear()
        byUrl.Clear()
        byStatus.Clear()

/// Evidence pipeline for processing raw content into proposals
module EvidencePipeline =

    /// Segment content into paragraphs
    let segmentParagraphs (content: string) =
        content.Split([| "\n\n"; "\r\n\r\n" |], StringSplitOptions.RemoveEmptyEntries)
        |> Array.filter (fun s -> s.Trim().Length > 50)
        |> Array.toList

    /// Create a proposal from extracted information
    let createProposal
        (subject: string)
        (predicate: string)
        (obj: string)
        (section: string)
        (confidence: float)
        (agentId: AgentId)
        =
        { Id = Guid.NewGuid()
          Subject = subject
          Predicate = predicate
          Object = obj
          SourceSection = section
          Confidence = confidence
          ExtractorAgent = agentId
          ExtractedAt = DateTime.UtcNow }
