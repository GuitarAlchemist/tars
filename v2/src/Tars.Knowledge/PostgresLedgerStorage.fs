/// TARS Postgres Ledger Storage - Persistent event-sourced beliefs
/// "Law is written in stone, or at least in Postgres"
namespace Tars.Knowledge

open System
open System.Collections.Generic
open System.Threading.Tasks
open System.Text.Json
open System.Text.Json.Serialization
open Npgsql

/// Postgres implementation of knowledge ledger storage
type PostgresLedgerStorage(connectionString: string) =

    // JSON options with F# support
    let jsonOptions =
        let mutable opts = JsonSerializerOptions()
        opts.Converters.Add(JsonFSharpConverter())
        opts.WriteIndented <- false
        opts

    let executeNonQuery (sql: string) (parameters: (string * obj) list) =
        task {
            try
                use conn = new NpgsqlConnection(connectionString)
                do! conn.OpenAsync()
                use cmd = new NpgsqlCommand(sql, conn)

                for (name, value) in parameters do
                    cmd.Parameters.AddWithValue(name, value) |> ignore

                let! _ = cmd.ExecuteNonQueryAsync()
                return Ok()
            with
            | :? PostgresException as ex when ex.SqlState = "28P01" ->
                return
                    Error
                        "Postgres Authentication Failed (28P01). Check your TARS_POSTGRES_CONNECTION environment variable or 'Memory:PostgresConnectionString' in appsettings.json."
            | ex -> return Error ex.Message
        }

    let ensureSchema () =
        task {
            let sql =
                """
            -- Knowledge Ledger Event Log
            CREATE TABLE IF NOT EXISTS knowledge_ledger (
                id UUID PRIMARY KEY,
                belief_id UUID NOT NULL,
                event_type TEXT NOT NULL,
                event_data JSONB NOT NULL,
                agent_id TEXT NOT NULL,
                run_id UUID NULL,
                timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_ledger_belief_id ON knowledge_ledger(belief_id);
            CREATE INDEX IF NOT EXISTS idx_ledger_timestamp ON knowledge_ledger(timestamp);

            -- Evidence Candidates (Internet Ingestion)
            CREATE TABLE IF NOT EXISTS evidence_candidates (
                id UUID PRIMARY KEY,
                source_uri TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                raw_content TEXT NOT NULL,
                fetched_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                status TEXT NOT NULL, -- 'Pending', 'Verified', 'Rejected', 'Conflicting'
                segments JSONB NOT NULL DEFAULT '[]',
                metadata JSONB NOT NULL DEFAULT '{}'
            );

            ALTER TABLE evidence_candidates
            ADD COLUMN IF NOT EXISTS segments JSONB NOT NULL DEFAULT '[]';

            -- Proposed Assertions (Extracted from Evidence)
            CREATE TABLE IF NOT EXISTS proposed_assertions (
                id UUID PRIMARY KEY,
                evidence_id UUID NULL REFERENCES evidence_candidates(id),
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                confidence DOUBLE PRECISION NOT NULL,
                source_section TEXT NULL,
                extractor_agent TEXT NOT NULL,
                extracted_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            -- Plans (Materialized view of symbolic goals)
            CREATE TABLE IF NOT EXISTS plans (
                id UUID PRIMARY KEY,
                goal_description TEXT NOT NULL,
                status TEXT NOT NULL,
                priority INT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """

            return! executeNonQuery sql []
        }

    interface ILedgerStorage with
        member _.Append(entry) =
            task {
                let! schemaResult = ensureSchema ()

                match schemaResult with
                | Error e -> return Error e
                | Ok() ->
                    let sql =
                        """
                    INSERT INTO knowledge_ledger (id, belief_id, event_type, event_data, agent_id, run_id, timestamp)
                    VALUES (@id, @beliefId, @type, @data::jsonb, @agentId, @runId, @timestamp)
                    """

                    let eventType =
                        match entry.Event with
                        | Assert _ -> "Assert"
                        | Retract _ -> "Retract"
                        | Weaken _ -> "Weaken"
                        | Strengthen _ -> "Strengthen"
                        | Link _ -> "Link"
                        | Contradict _ -> "Contradict"
                        | SchemaEvolve _ -> "SchemaEvolve"

                    let beliefId =
                        match entry.Event with
                        | Assert b -> b.Id.Value
                        | Retract(id, _, _) -> id.Value
                        | Weaken(id, _, _) -> id.Value
                        | Strengthen(id, _, _) -> id.Value
                        | Link(s, _, _) -> s.Value
                        | Contradict(b1, _, _) -> b1.Value
                        | SchemaEvolve(_, _) -> Guid.Empty

                    let data = JsonSerializer.Serialize(entry.Event, jsonOptions)

                    let parameters =
                        [ "@id", box entry.EventId
                          "@beliefId", box beliefId
                          "@type", box eventType
                          "@data", box data
                          "@agentId", box entry.AgentId.Value
                          "@runId",
                          match entry.RunId with
                          | Some r -> box r.Value
                          | None -> box DBNull.Value
                          "@timestamp", box entry.Timestamp ]

                    return! executeNonQuery sql parameters
            }

        member _.GetEvents(since) =
            task {
                let! _ = ensureSchema ()

                let sql =
                    "SELECT id, event_data, agent_id, run_id, timestamp FROM knowledge_ledger "
                    + (if since.IsSome then "WHERE timestamp > @since " else "")
                    + "ORDER BY timestamp ASC"

                let results = ResizeArray<BeliefEventEntry>()

                use conn = new NpgsqlConnection(connectionString)
                do! conn.OpenAsync()
                use cmd = new NpgsqlCommand(sql, conn)

                if since.IsSome then
                    cmd.Parameters.AddWithValue("@since", since.Value) |> ignore

                use! reader = cmd.ExecuteReaderAsync()

                while reader.Read() do
                    let id = reader.GetGuid(0)
                    let data = reader.GetString(1)
                    let agentId = reader.GetString(2)

                    let runId =
                        if reader.IsDBNull(3) then
                            None
                        else
                            Some(RunId(reader.GetGuid(3)))

                    let timestamp = reader.GetDateTime(4)

                    try
                        let eventValue = JsonSerializer.Deserialize<BeliefEvent>(data, jsonOptions)

                        results.Add(
                            { EventId = id
                              Event = eventValue
                              AgentId = AgentId agentId
                              RunId = runId
                              Timestamp = timestamp
                              Metadata = Map.empty }
                        )
                    with _ ->
                        ()

                return results |> Seq.toList
            }

        member _.GetEventsByBelief(beliefId) =
            task {
                let! _ = ensureSchema ()

                let sql =
                    "SELECT id, event_data, agent_id, run_id, timestamp FROM knowledge_ledger WHERE belief_id = @bid ORDER BY timestamp ASC"

                let results = ResizeArray<BeliefEventEntry>()

                use conn = new NpgsqlConnection(connectionString)
                do! conn.OpenAsync()
                use cmd = new NpgsqlCommand(sql, conn)
                cmd.Parameters.AddWithValue("@bid", beliefId.Value) |> ignore

                use! reader = cmd.ExecuteReaderAsync()

                while reader.Read() do
                    let id = reader.GetGuid(0)
                    let data = reader.GetString(1)
                    let agentId = reader.GetString(2)

                    let runId =
                        if reader.IsDBNull(3) then
                            None
                        else
                            Some(RunId(reader.GetGuid(3)))

                    let timestamp = reader.GetDateTime(4)

                    try
                        let eventValue = JsonSerializer.Deserialize<BeliefEvent>(data, jsonOptions)

                        results.Add(
                            { EventId = id
                              Event = eventValue
                              AgentId = AgentId agentId
                              RunId = runId
                              Timestamp = timestamp
                              Metadata = Map.empty }
                        )
                    with _ ->
                        ()

                return results |> Seq.toList
            }

        member this.GetSnapshot() =
            task {
                let! events = (this :> ILedgerStorage).GetEvents(None)

                let beliefs = Dictionary<BeliefId, Belief>()

                for entry in events do
                    match entry.Event with
                    | Assert belief -> beliefs.[belief.Id] <- belief
                    | Retract(id, _, _) ->
                        match beliefs.TryGetValue(id) with
                        | true, b ->
                            beliefs.[id] <-
                                { b with
                                    InvalidAt = Some entry.Timestamp }
                        | false, _ -> ()
                    | Weaken(id, newConf, _) ->
                        match beliefs.TryGetValue(id) with
                        | true, b -> beliefs.[id] <- { b with Confidence = newConf }
                        | false, _ -> ()
                    | Strengthen(id, newConf, _) ->
                        match beliefs.TryGetValue(id) with
                        | true, b -> beliefs.[id] <- { b with Confidence = newConf }
                        | false, _ -> ()
                    | _ -> ()

                return beliefs.Values |> Seq.toList
            }

    interface IEvidenceStorage with
        member _.SaveCandidate(candidate) =
            task {
                let! _ = ensureSchema ()

                let sql =
                    """
                    INSERT INTO evidence_candidates (id, source_uri, content_hash, raw_content, fetched_at, status, segments, metadata)
                    VALUES (@id, @uri, @hash, @content, @fetched, @status, @segments::jsonb, @metadata::jsonb)
                    ON CONFLICT (id) DO UPDATE SET
                        status = EXCLUDED.status,
                        segments = EXCLUDED.segments,
                        metadata = EXCLUDED.metadata;
                """

                try
                    use conn = new NpgsqlConnection(connectionString)
                    do! conn.OpenAsync()
                    use cmd = new NpgsqlCommand(sql, conn)

                    cmd.Parameters.AddWithValue("@id", candidate.Id) |> ignore
                    cmd.Parameters.AddWithValue("@uri", candidate.SourceUrl.ToString()) |> ignore
                    cmd.Parameters.AddWithValue("@hash", candidate.ContentHash) |> ignore
                    cmd.Parameters.AddWithValue("@content", candidate.RawContent) |> ignore
                    cmd.Parameters.AddWithValue("@fetched", candidate.FetchedAt) |> ignore
                    cmd.Parameters.AddWithValue("@status", candidate.Status.ToString()) |> ignore

                    // Use proper JSON serialization for JSONB columns
                    let segmentsJson = JsonSerializer.Serialize(candidate.Segments, jsonOptions)
                    let metadataJson = JsonSerializer.Serialize(candidate.Metadata, jsonOptions)

                    cmd.Parameters.AddWithValue("@segments", NpgsqlTypes.NpgsqlDbType.Jsonb, segmentsJson)
                    |> ignore

                    cmd.Parameters.AddWithValue("@metadata", NpgsqlTypes.NpgsqlDbType.Jsonb, metadataJson)
                    |> ignore

                    let! _ = cmd.ExecuteNonQueryAsync()
                    return Ok()
                with
                | :? PostgresException as ex when ex.SqlState = "28P01" ->
                    return Error "Postgres Authentication Failed (28P01). Check your connection string."
                | ex -> return Error ex.Message
            }

        member _.SaveProposal(proposal, evidenceId) =
            task {
                let! _ = ensureSchema ()

                let sql =
                    """
                    INSERT INTO proposed_assertions (id, evidence_id, subject, predicate, object, confidence, source_section, extractor_agent, extracted_at)
                    VALUES (@id, @evId, @sub, @pred, @obj, @conf, @sec, @agent, @at)
                """

                let parameters =
                    [ "@id", box proposal.Id
                      "@evId",
                      match evidenceId with
                      | Some id -> box id
                      | None -> box DBNull.Value
                      "@sub", box proposal.Subject
                      "@pred", box proposal.Predicate
                      "@obj", box proposal.Object
                      "@conf", box proposal.Confidence
                      "@sec", box proposal.SourceSection
                      "@agent", box proposal.ExtractorAgent.Value
                      "@at", box proposal.ExtractedAt ]

                return! executeNonQuery sql parameters
            }

        member _.GetPendingCandidates() =
            task {
                let! _ = ensureSchema ()

                let sql =
                    "SELECT id, source_uri, content_hash, raw_content, fetched_at, status, segments, metadata FROM evidence_candidates WHERE status = 'Pending'"

                let results = ResizeArray<EvidenceCandidate>()
                use conn = new NpgsqlConnection(connectionString)
                do! conn.OpenAsync()
                use cmd = new NpgsqlCommand(sql, conn)
                use! reader = cmd.ExecuteReaderAsync()

                while reader.Read() do
                    results.Add(
                        { Id = reader.GetGuid(0)
                          SourceUrl = Uri(reader.GetString(1))
                          ContentHash = reader.GetString(2)
                          RawContent = reader.GetString(3)
                          FetchedAt = reader.GetDateTime(4)
                          Segments = JsonSerializer.Deserialize<string list>(reader.GetString(6), jsonOptions)
                          ProposedAssertions = [] // Loaded separately if needed
                          Metadata = JsonSerializer.Deserialize<Map<string, string>>(reader.GetString(7), jsonOptions)
                          Status =
                            match reader.GetString(5) with
                            | "Pending" -> Pending
                            | "Verified" -> Verified
                            | "Rejected" -> Rejected
                            | "Conflicting" -> Conflicting
                            | _ -> Pending
                          VerifiedAt = None
                          VerifiedBy = None
                          RejectionReason = None }
                    )

                return results |> Seq.toList
            }

        member _.GetProposalsByEvidence(evidenceId) =
            task {
                let! _ = ensureSchema ()

                let sql =
                    "SELECT id, subject, predicate, object, confidence, source_section, extractor_agent, extracted_at FROM proposed_assertions WHERE evidence_id = @evId"

                let results = ResizeArray<ProposedAssertion>()
                use conn = new NpgsqlConnection(connectionString)
                do! conn.OpenAsync()
                use cmd = new NpgsqlCommand(sql, conn)
                cmd.Parameters.AddWithValue("@evId", evidenceId) |> ignore
                use! reader = cmd.ExecuteReaderAsync()

                while reader.Read() do
                    results.Add(
                        { Id = reader.GetGuid(0)
                          Subject = reader.GetString(1)
                          Predicate = reader.GetString(2)
                          Object = reader.GetString(3)
                          Confidence = reader.GetDouble(4)
                          SourceSection = if reader.IsDBNull(5) then "" else reader.GetString(5)
                          ExtractorAgent = AgentId(reader.GetString(6))
                          ExtractedAt = reader.GetDateTime(7) }
                    )

                return results |> Seq.toList
            }

module PostgresLedgerStorage =

    let defaultConnectionString =
        "Host=localhost;Database=tars_memory;Username=postgres;Password=tars_password"

    let create () =
        let connStr =
            Environment.GetEnvironmentVariable("TARS_POSTGRES_CONNECTION")
            |> Option.ofObj
            |> Option.defaultValue defaultConnectionString

        PostgresLedgerStorage(connStr)

    let createWithConnectionString (connStr: string) = PostgresLedgerStorage(connStr)
