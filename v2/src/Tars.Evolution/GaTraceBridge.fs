namespace Tars.Evolution

/// Reads GA trace artifacts from ~/.ga/traces/ and converts them
/// to PromotionPipeline.TraceArtifact records for cross-repo pattern discovery.
module GaTraceBridge =

    open System
    open System.IO
    open System.Text.Json

    // ── GA trace JSON shape ────────────────────────────────────────────

    type GaMetadata = {
        AgentId: string option
        Confidence: float option
        RoutingMethod: string option
        SkillName: string option
        OriginalQuery: string option
    }

    type GaTraceDto = {
        TaskId: string
        PatternName: string
        PatternTemplate: string
        Context: string
        Score: float
        Timestamp: DateTimeOffset
        RollbackExpansion: string option
        Source: string option
        EventType: string option
        GaMetadata: GaMetadata option
    }

    let private traceDir =
        Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".ga", "traces")

    let private jsonOptions =
        JsonSerializerOptions(
            PropertyNameCaseInsensitive = true,
            WriteIndented = true)

    // ── Read & convert ─────────────────────────────────────────────────

    /// Read all GA trace files, newest first.
    let readRawTraces (count: int) (sinceUtc: DateTime option) : GaTraceDto list =
        if not (Directory.Exists traceDir) then []
        else
            let files = Directory.GetFiles(traceDir, "*.json")
            Array.Sort(files)
            Array.Reverse(files) // newest first

            files
            |> Array.toList
            |> List.choose (fun file ->
                try
                    let json = File.ReadAllText(file)
                    let dto = JsonSerializer.Deserialize<GaTraceDto>(json, jsonOptions)
                    match sinceUtc with
                    | Some since when dto.Timestamp.UtcDateTime < since -> None
                    | _ -> Some dto
                with _ -> None)
            |> List.truncate count

    /// Convert a GA trace DTO to a TARS PromotionPipeline.TraceArtifact.
    let toTraceArtifact (dto: GaTraceDto) : PromotionPipeline.TraceArtifact =
        { TaskId = dto.TaskId
          PatternName = dto.PatternName
          PatternTemplate = dto.PatternTemplate
          Context = dto.Context
          Score = dto.Score
          Timestamp = dto.Timestamp.UtcDateTime
          RollbackExpansion = dto.RollbackExpansion }

    /// Read GA traces and convert to TARS TraceArtifacts ready for the promotion pipeline.
    let ingest (count: int) (sinceUtc: DateTime option) : PromotionPipeline.TraceArtifact list =
        readRawTraces count sinceUtc
        |> List.map toTraceArtifact

    /// Get stats about the GA trace directory.
    let stats () =
        if not (Directory.Exists traceDir) then
            {| FileCount = 0; Oldest = None; Newest = None; EventTypes = Map.empty |}
        else
            let files = Directory.GetFiles(traceDir, "*.json")
            if files.Length = 0 then
                {| FileCount = 0; Oldest = None; Newest = None; EventTypes = Map.empty |}
            else
                Array.Sort(files)
                let mutable oldest: DateTimeOffset option = None
                let mutable newest: DateTimeOffset option = None
                let mutable eventTypes = Map.empty<string, int>

                for file in files do
                    try
                        let json = File.ReadAllText(file)
                        let dto = JsonSerializer.Deserialize<GaTraceDto>(json, jsonOptions)

                        if oldest.IsNone then oldest <- Some dto.Timestamp
                        newest <- Some dto.Timestamp

                        match dto.EventType with
                        | Some et ->
                            let count = eventTypes |> Map.tryFind et |> Option.defaultValue 0
                            eventTypes <- eventTypes |> Map.add et (count + 1)
                        | None -> ()
                    with _ -> ()

                {| FileCount = files.Length
                   Oldest = oldest |> Option.map (fun d -> d.ToString("o"))
                   Newest = newest |> Option.map (fun d -> d.ToString("o"))
                   EventTypes = eventTypes |}
