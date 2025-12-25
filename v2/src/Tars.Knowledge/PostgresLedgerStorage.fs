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

            -- Plans (Event-sourced symbolic goals and hypotheses)
            CREATE TABLE IF NOT EXISTS plans (
                id UUID PRIMARY KEY,
                goal TEXT NOT NULL,
                assumptions JSONB NOT NULL DEFAULT '[]',
                steps JSONB NOT NULL DEFAULT '[]',
                success_metrics JSONB NOT NULL DEFAULT '[]',
                risk_factors JSONB NOT NULL DEFAULT '[]',
                version INT NOT NULL DEFAULT 1,
                parent_version UUID NULL,
                status TEXT NOT NULL DEFAULT 'Draft',
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT NOT NULL,
                tags JSONB NOT NULL DEFAULT '[]'
            );

            -- Plan Events (Event log for plan lifecycle)
            CREATE TABLE IF NOT EXISTS plan_events (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                plan_id UUID NOT NULL,
                event_type TEXT NOT NULL,
                event_data JSONB NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_plans_status ON plans(status);
            CREATE INDEX IF NOT EXISTS idx_plans_created_by ON plans(created_by);
            CREATE INDEX IF NOT EXISTS idx_plan_events_plan_id ON plan_events(plan_id);

            -- Beliefs Snapshot (Materialized view of current belief state)
            -- This table provides fast access to current beliefs without replaying all events
            CREATE TABLE IF NOT EXISTS beliefs (
                id UUID PRIMARY KEY,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                confidence DOUBLE PRECISION NOT NULL,
                provenance_source TEXT NOT NULL,
                provenance_agent TEXT NOT NULL,
                provenance_run_id UUID NULL,
                provenance_confidence DOUBLE PRECISION NOT NULL,
                provenance_timestamp TIMESTAMPTZ NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT NOT NULL,
                tags JSONB NOT NULL DEFAULT '[]'
            );

            CREATE INDEX IF NOT EXISTS idx_beliefs_subject ON beliefs(subject);
            CREATE INDEX IF NOT EXISTS idx_beliefs_predicate ON beliefs(predicate);
            CREATE INDEX IF NOT EXISTS idx_beliefs_created_by ON beliefs(created_by);
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

    interface IPlanStorage with
        member _.SavePlan(plan) =
            task {
                let! _ = ensureSchema ()

                let sql =
                    """
                    INSERT INTO plans (id, goal, assumptions, steps, success_metrics, risk_factors, 
                                     version, parent_version, status, created_at, updated_at, created_by, tags)
                    VALUES (@id, @goal, @assumptions::jsonb, @steps::jsonb, @success_metrics::jsonb, @risk_factors::jsonb,
                            @version, @parent_version, @status, @created_at, @updated_at, @created_by, @tags::jsonb)
                """

                let parameters =
                    [ "@id", box plan.Id.Value
                      "@goal", box plan.Goal
                      "@assumptions", box (JsonSerializer.Serialize(plan.Assumptions, jsonOptions))
                      "@steps", box (JsonSerializer.Serialize(plan.Steps, jsonOptions))
                      "@success_metrics", box (JsonSerializer.Serialize(plan.SuccessMetrics, jsonOptions))
                      "@risk_factors", box (JsonSerializer.Serialize(plan.RiskFactors, jsonOptions))
                      "@version", box plan.Version
                      "@parent_version",
                      match plan.ParentVersion with
                      | Some pv -> box pv.Value
                      | None -> box DBNull.Value
                      "@status", box (plan.Status.ToString())
                      "@created_at", box plan.CreatedAt
                      "@updated_at", box plan.UpdatedAt
                      "@created_by", box plan.CreatedBy.Value
                      "@tags", box (JsonSerializer.Serialize(plan.Tags, jsonOptions)) ]

                return! executeNonQuery sql parameters
            }

        member _.UpdatePlan(plan) =
            task {
                let! _ = ensureSchema ()

                let sql =
                    """
                    UPDATE plans SET
                        goal = @goal,
                        assumptions = @assumptions::jsonb,
                        steps = @steps::jsonb,
                        success_metrics = @success_metrics::jsonb,
                        risk_factors = @risk_factors::jsonb,
                        status = @status,
                        updated_at = @updated_at,
                        tags = @tags::jsonb
                    WHERE id = @id
                """

                let parameters =
                    [ "@id", box plan.Id.Value
                      "@goal", box plan.Goal
                      "@assumptions", box (JsonSerializer.Serialize(plan.Assumptions, jsonOptions))
                      "@steps", box (JsonSerializer.Serialize(plan.Steps, jsonOptions))
                      "@success_metrics", box (JsonSerializer.Serialize(plan.SuccessMetrics, jsonOptions))
                      "@risk_factors", box (JsonSerializer.Serialize(plan.RiskFactors, jsonOptions))
                      "@status", box (plan.Status.ToString())
                      "@updated_at", box plan.UpdatedAt
                      "@tags", box (JsonSerializer.Serialize(plan.Tags, jsonOptions)) ]

                return! executeNonQuery sql parameters
            }

        member _.GetPlan(planId) =
            task {
                let! _ = ensureSchema ()

                let sql =
                    "SELECT id, goal, assumptions, steps, success_metrics, risk_factors, version, parent_version, status, created_at, updated_at, created_by, tags FROM plans WHERE id = @id"

                use conn = new NpgsqlConnection(connectionString)
                do! conn.OpenAsync()
                use cmd = new NpgsqlCommand(sql, conn)
                cmd.Parameters.AddWithValue("@id", planId.Value) |> ignore
                use! reader = cmd.ExecuteReaderAsync()

                if reader.Read() then
                    let plan =
                        { Id = PlanId(reader.GetGuid(0))
                          Goal = reader.GetString(1)
                          Assumptions = JsonSerializer.Deserialize<BeliefId list>(reader.GetString(2), jsonOptions)
                          Steps = JsonSerializer.Deserialize<PlanStep list>(reader.GetString(3), jsonOptions)
                          SuccessMetrics = JsonSerializer.Deserialize<string list>(reader.GetString(4), jsonOptions)
                          RiskFactors = JsonSerializer.Deserialize<string list>(reader.GetString(5), jsonOptions)
                          Version = reader.GetInt32(6)
                          ParentVersion =
                            if reader.IsDBNull(7) then
                                None
                            else
                                Some(PlanId(reader.GetGuid(7)))
                          Status =
                            match reader.GetString(8) with
                            | "Draft" -> PlanStatus.Draft
                            | "Active" -> PlanStatus.Active
                            | "Paused" -> PlanStatus.Paused
                            | "Completed" -> PlanStatus.Completed
                            | "Failed" -> PlanStatus.Failed
                            | "Superseded" -> PlanStatus.Superseded
                            | _ -> PlanStatus.Draft
                          CreatedAt = reader.GetDateTime(9)
                          UpdatedAt = reader.GetDateTime(10)
                          CreatedBy = AgentId(reader.GetString(11))
                          Tags = JsonSerializer.Deserialize<string list>(reader.GetString(12), jsonOptions) }

                    return Some plan
                else
                    return None
            }

        member _.GetPlansByStatus(status) =
            task {
                let! _ = ensureSchema ()

                let sql =
                    "SELECT id, goal, assumptions, steps, success_metrics, risk_factors, version, parent_version, status, created_at, updated_at, created_by, tags FROM plans WHERE status = @status"

                let results = ResizeArray<Plan>()
                use conn = new NpgsqlConnection(connectionString)
                do! conn.OpenAsync()
                use cmd = new NpgsqlCommand(sql, conn)
                cmd.Parameters.AddWithValue("@status", status.ToString()) |> ignore
                use! reader = cmd.ExecuteReaderAsync()

                while reader.Read() do
                    results.Add(
                        { Id = PlanId(reader.GetGuid(0))
                          Goal = reader.GetString(1)
                          Assumptions = JsonSerializer.Deserialize<BeliefId list>(reader.GetString(2), jsonOptions)
                          Steps = JsonSerializer.Deserialize<PlanStep list>(reader.GetString(3), jsonOptions)
                          SuccessMetrics = JsonSerializer.Deserialize<string list>(reader.GetString(4), jsonOptions)
                          RiskFactors = JsonSerializer.Deserialize<string list>(reader.GetString(5), jsonOptions)
                          Version = reader.GetInt32(6)
                          ParentVersion =
                            if reader.IsDBNull(7) then
                                None
                            else
                                Some(PlanId(reader.GetGuid(7)))
                          Status =
                            match reader.GetString(8) with
                            | "Draft" -> PlanStatus.Draft
                            | "Active" -> PlanStatus.Active
                            | "Paused" -> PlanStatus.Paused
                            | "Completed" -> PlanStatus.Completed
                            | "Failed" -> PlanStatus.Failed
                            | "Superseded" -> PlanStatus.Superseded
                            | _ -> PlanStatus.Draft
                          CreatedAt = reader.GetDateTime(9)
                          UpdatedAt = reader.GetDateTime(10)
                          CreatedBy = AgentId(reader.GetString(11))
                          Tags = JsonSerializer.Deserialize<string list>(reader.GetString(12), jsonOptions) }
                    )

                return results |> Seq.toList
            }

        member _.AppendEvent(event) =
            task {
                let! _ = ensureSchema ()

                let planId, eventType =
                    match event with
                    | PlanCreated p -> (p.Id, "PlanCreated")
                    | StepStarted(id, _) -> (id, "StepStarted")
                    | StepCompleted(id, _, _) -> (id, "StepCompleted")
                    | StepFailed(id, _, _) -> (id, "StepFailed")
                    | AssumptionInvalidated(id, _, _) -> (id, "AssumptionInvalidated")
                    | PlanForked(id, _) -> (id, "PlanForked")
                    | PlanCompleted id -> (id, "PlanCompleted")
                    | PlanFailed(id, _) -> (id, "PlanFailed")
                    | PlanSuperseded(id, _) -> (id, "PlanSuperseded")

                let eventData = JsonSerializer.Serialize(event, jsonOptions)

                let sql =
                    """
                    INSERT INTO plan_events (plan_id, event_type, event_data)
                    VALUES (@plan_id, @event_type, @event_data::jsonb)
                """

                let parameters =
                    [ "@plan_id", box planId.Value
                      "@event_type", box eventType
                      "@event_data", box eventData ]

                return! executeNonQuery sql parameters
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
