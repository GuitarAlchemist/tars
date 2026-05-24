namespace Tars.Tests

open System
open Xunit
open Tars.Core

module GraphitiTypesTests =

    [<Fact>]
    let TemporalValidity_now_creates_valid_entries () =
        let validity = TemporalValidityOps.now ()
        Assert.True(validity.InvalidAt.IsNone)
        Assert.True(validity.ExpiredAt.IsNone)
        Assert.True(validity.ValidFrom <= DateTime.UtcNow)

    [<Fact>]
    let TemporalValidity_invalidate_marks_as_invalid () =
        let validity = TemporalValidityOps.now ()
        let invalidated = TemporalValidityOps.invalidate validity
        Assert.True(invalidated.InvalidAt.IsSome)
        Assert.True(invalidated.ExpiredAt.IsSome)

    [<Fact>]
    let TemporalValidity_isValidAt_checks_time () =
        let validity =
            { ValidFrom = DateTime(2024, 1, 1)
              InvalidAt = Some(DateTime(2024, 6, 1))
              CreatedAt = DateTime(2024, 1, 1)
              ExpiredAt = None }

        Assert.True(TemporalValidityOps.isValidAt (DateTime(2024, 3, 1)) validity)
        Assert.False(TemporalValidityOps.isValidAt (DateTime(2024, 7, 1)) validity)

    [<Fact>]
    let Episode_timestamp_extracts_time () =
        let ts = DateTime.UtcNow
        let episode = AgentInteraction("agent1", "input", "output", ts)
        Assert.Equal(ts, Episode.timestamp episode)

    [<Fact>]
    let Episode_typeTag_returns_correct_tag () =
        let episode = CodeChange("file.fs", "diff", "author", DateTime.UtcNow)
        Assert.Equal("code_change", Episode.typeTag episode)

    [<Fact>]
    let TarsEntity_getId_stable_for_same_name () =
        let pattern1 =
            CodePatternE
                { Name = "TestPattern"
                  Category = Structural
                  Signature = "sig"
                  Occurrences = 5
                  FirstSeen = DateTime.UtcNow
                  LastSeen = DateTime.UtcNow }

        let pattern2 =
            CodePatternE
                { Name = "TestPattern"
                  Category = Structural
                  Signature = "sig"
                  Occurrences = 10
                  FirstSeen = DateTime.UtcNow
                  LastSeen = DateTime.UtcNow }

        Assert.Equal(TarsEntity.getId pattern1, TarsEntity.getId pattern2)

    [<Fact>]
    let TarsFact_source_extracts_entity () =
        let entity1 =
            ConceptE
                { Name = "A"
                  Description = ""
                  RelatedConcepts = [] }

        let entity2 =
            ConceptE
                { Name = "B"
                  Description = ""
                  RelatedConcepts = [] }

        let fact = TarsFact.DependsOn(entity1, entity2, 0.9)
        let sourceId = TarsEntity.getId (TarsFact.source fact)
        Assert.Equal("concept:a", sourceId)

    [<Fact>]
    let BelongsTo_fact_has_no_target () =
        let entity =
            ConceptE
                { Name = "A"
                  Description = ""
                  RelatedConcepts = [] }

        let fact = TarsFact.BelongsTo(entity, "community-1")
        Assert.True((TarsFact.target fact).IsNone)

module EpisodeStoreTests =

    [<Fact>]
    let EpisodeStore_ingest_and_retrieve () =
        let tempPath =
            System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"tars_test_{Guid.NewGuid():N}")

        let store = EpisodeStore(tempPath)

        let ts = DateTime.UtcNow
        let episode = AgentInteraction("agent1", "Hello", "Hi there!", ts)
        let id = store.Ingest(episode)

        let retrieved = store.Get(id)
        Assert.True(retrieved.IsSome)
        Assert.Equal(ts, Episode.timestamp retrieved.Value)

        store.Clear()

        if System.IO.Directory.Exists tempPath then
            System.IO.Directory.Delete(tempPath, true)

    [<Fact>]
    let EpisodeStore_query_by_type () =
        let tempPath =
            System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"tars_test_{Guid.NewGuid():N}")

        let store = EpisodeStore(tempPath)

        store.Ingest(AgentInteraction("a1", "in", "out", DateTime.UtcNow)) |> ignore
        store.Ingest(CodeChange("f.fs", "diff", "me", DateTime.UtcNow)) |> ignore
        store.Ingest(AgentInteraction("a2", "in2", "out2", DateTime.UtcNow)) |> ignore

        let agentEpisodes = store.GetByType("agent_interaction")
        Assert.Equal(2, agentEpisodes.Length)

        store.Clear()

        if System.IO.Directory.Exists tempPath then
            System.IO.Directory.Delete(tempPath, true)

    [<Fact>]
    let EpisodeStore_count_updates () =
        let tempPath =
            System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"tars_test_{Guid.NewGuid():N}")

        let store = EpisodeStore(tempPath)

        Assert.Equal(0, store.Count)
        store.Ingest(AgentInteraction("a1", "1", "1", DateTime.UtcNow)) |> ignore
        Assert.Equal(1, store.Count)

        store.Clear()

        if System.IO.Directory.Exists tempPath then
            System.IO.Directory.Delete(tempPath, true)

module TraceEntityTests =
    [<Fact>]
    let TarsEntity_getId_works_for_traces() =
        let runId = Guid.NewGuid()
        let runEnt = { Tars.Core.RunEntity.Id = runId; Goal = "Test"; Pattern = "WoT"; Timestamp = DateTime.UtcNow }
        Assert.Equal($"run:{runId}", TarsEntity.getId (RunE runEnt))
        
        let stepEnt = { Tars.Core.StepEntity.RunId = runId; StepId = "step1"; NodeType = "Generate"; Content = ""; Timestamp = DateTime.UtcNow }
        Assert.Equal($"step:{runId}:step1", TarsEntity.getId (StepE stepEnt))

    [<Fact>]
    let TarsFact_NextStep_source_target() =
        let runId = Guid.NewGuid()
        let s = StepE { Tars.Core.StepEntity.RunId = runId; StepId = "1"; NodeType="G"; Content=""; Timestamp=DateTime.UtcNow }
        let t = StepE { Tars.Core.StepEntity.RunId = runId; StepId = "2"; NodeType="G"; Content=""; Timestamp=DateTime.UtcNow }
        let fact = TarsFact.NextStep(s, t)
        
        Assert.Equal(s, TarsFact.source fact)
        Assert.Equal(Some t, TarsFact.target fact)
