namespace Tars.Evolution

/// MCP bridge: chatbot/GA-tool claim traces → TARS knowledge graph nodes.
///
/// Claims are emitted as per-day JSONL files by <c>GaMcpServer.ClaimTraceLog</c>
/// (one line per MCP tool invocation). This bridge reads those lines, transforms
/// each into a Claim-type graph node, and calls <c>GraphTools.graphAddNode</c>
/// to land it in the persistent TARS knowledge graph.
///
/// Cross-process path handling: the ingestion input takes an explicit
/// <c>claimsDir</c> parameter — there is no filesystem-inferred default because
/// the claims file lives near the ga repo root (walked up from the DLL) while
/// the TARS graph lives at the TARS process cwd. Callers must pass the path
/// they observe. See docs/plans/2026-04-19-tars-graph-persistence.md §
/// "Two trace destinations" for rationale.
module ChatbotClaimsBridge =

    open System
    open System.IO
    open System.Text.Json
    open System.Text.Json.Serialization
    open Tars.Core
    open Tars.Tools.Graph

    let private jsonOptions =
        let o = JsonSerializerOptions(WriteIndented = false)
        o.Converters.Add(JsonFSharpConverter())
        o

    // =========================================================================
    // Types
    // =========================================================================

    type IngestInput =
        { ClaimsDir: string option
          SinceIso: string option
          Limit: int option }

    type IngestResponse =
        { ClaimsRead: int
          NodesAdded: int
          Skipped: int
          Errors: string list
          GraphNodeCountAfter: int }

    // =========================================================================
    // Claim → node transform
    // =========================================================================

    /// Build a stable, human-readable node id from a claim. Same tool + same
    /// timestamp = same id, so replaying the same JSONL file is a no-op on the
    /// graph (the log grows but nodes are overwritten in place).
    let private nodeId (tool: string) (ts: string) =
        let safeTool = tool.Replace('.', '-').Replace('/', '-')
        let safeTs = ts.Replace(':', '-').Replace('.', '-').Replace('+', 'p')
        sprintf "c:%s:%s" safeTool safeTs

    /// Pull a representative subject value from the inputs dictionary.
    /// For single-input tools (parseChord(symbol="Cmaj7")) returns "Cmaj7".
    /// For multi-input tools returns "k1=v1; k2=v2" (stable order).
    let private subjectOf (inputs: JsonElement) =
        let props =
            inputs.EnumerateObject()
            |> Seq.map (fun p -> p.Name, p.Value.ToString())
            |> Seq.toList
        match props with
        | [] -> ""
        | [ _, v ] -> v
        | many ->
            many
            |> List.map (fun (k, v) -> sprintf "%s=%s" k v)
            |> String.concat "; "

    /// Transform one ClaimTraceRecord JSON line into the JSON payload expected
    /// by <c>graphAddNode</c>. Returns None if the line is malformed.
    let private claimToNodeJson (line: string) : string option =
        try
            use doc = JsonDocument.Parse line
            let root = doc.RootElement
            let tool = root.GetProperty("tool").GetString()
            let ts = root.GetProperty("ts").GetString()
            let session = root.GetProperty("session").GetString()
            let src = root.GetProperty("src").GetString()
            let truth = root.GetProperty("truth").GetString()
            let outcome = root.GetProperty("outcome").GetString()
            let inputs = root.GetProperty("inputs")
            let id = nodeId tool ts
            let subject = subjectOf inputs
            let label = sprintf "%s(%s) → %s" tool subject truth

            use ms = new MemoryStream()
            use writer = new Utf8JsonWriter(ms)
            writer.WriteStartObject()
            writer.WriteString("id", id)
            writer.WriteString("type", "Claim")
            writer.WriteString("label", label)
            writer.WriteNumber("confidence", if truth = "T" then 1.0 else 0.5)
            writer.WriteString("tool", tool)
            writer.WriteString("subject", subject)
            writer.WriteString("session", session)
            writer.WriteString("src", src)
            writer.WriteString("truth", truth)
            writer.WriteString("outcome", outcome)
            writer.WriteString("ts", ts)
            // Keep raw inputs for downstream extraction
            writer.WritePropertyName("inputs")
            inputs.WriteTo(writer)
            writer.WriteEndObject()
            writer.Flush()
            Some(System.Text.Encoding.UTF8.GetString(ms.ToArray()))
        with _ -> None

    // =========================================================================
    // File enumeration + filter
    // =========================================================================

    let private parseIso (s: string) =
        match DateTimeOffset.TryParse s with
        | true, dto -> Some dto
        | false, _ -> None

    let private readClaimLines (claimsDir: string) (since: DateTimeOffset option) (limit: int) =
        if not (Directory.Exists claimsDir) then
            []
        else
            let files = Directory.GetFiles(claimsDir, "*.jsonl") |> Array.sort
            let keep = ResizeArray<string>()
            let mutable stop = false
            for file in files do
                if not stop then
                    for line in File.ReadAllLines file do
                        if not stop && not (String.IsNullOrWhiteSpace line) then
                            let passesSince =
                                match since with
                                | None -> true
                                | Some threshold ->
                                    try
                                        use doc = JsonDocument.Parse line
                                        match doc.RootElement.TryGetProperty "ts" with
                                        | true, tsEl ->
                                            match parseIso (tsEl.GetString()) with
                                            | Some t -> t >= threshold
                                            | None -> true
                                        | _ -> true
                                    with _ -> false
                            if passesSince then
                                keep.Add line
                                if keep.Count >= limit then stop <- true
            keep |> Seq.toList

    // =========================================================================
    // Tool handler
    // =========================================================================

    /// Extract a string property case-insensitively (ClaimsDir / claimsDir both match).
    let private tryGetString (root: JsonElement) (name: string) =
        let mutable p = Unchecked.defaultof<JsonElement>
        if root.TryGetProperty(name, &p) && p.ValueKind = JsonValueKind.String then
            let s = p.GetString()
            if String.IsNullOrWhiteSpace s then None else Some s
        else
            // Try lowercased-first variant
            let alt = (string (Char.ToLower name.[0])) + name.Substring(1)
            if alt <> name && root.TryGetProperty(alt, &p) && p.ValueKind = JsonValueKind.String then
                let s = p.GetString()
                if String.IsNullOrWhiteSpace s then None else Some s
            else
                None

    let private tryGetInt (root: JsonElement) (name: string) =
        let mutable p = Unchecked.defaultof<JsonElement>
        if root.TryGetProperty(name, &p) && p.ValueKind = JsonValueKind.Number then Some(p.GetInt32())
        else
            let alt = (string (Char.ToLower name.[0])) + name.Substring(1)
            if alt <> name && root.TryGetProperty(alt, &p) && p.ValueKind = JsonValueKind.Number then Some(p.GetInt32())
            else None

    let private ingestChatbotClaims (input: string) : Result<string, string> =
        try
            // Parse input JSON manually — JsonFSharpConverter's default union
            // encoding doesn't deserialize `"ClaimsDir":"..."` → `Some "..."`
            // reliably across all F# record shapes. JsonDocument is unambiguous.
            use inputDoc = JsonDocument.Parse(if String.IsNullOrWhiteSpace input then "{}" else input)
            let inputRoot = inputDoc.RootElement

            let claimsDir = tryGetString inputRoot "ClaimsDir"
            let sinceIso = tryGetString inputRoot "SinceIso"
            let limitOpt = tryGetInt inputRoot "Limit"

            match claimsDir with
            | None ->
                Result.Error "ingest_chatbot_claims requires ClaimsDir (no default; claims live at <ga-repo>/state/claims/ when ga-mcp-server runs via its normal launch path)."
            | Some dir ->
                let since = sinceIso |> Option.bind parseIso
                let limit = limitOpt |> Option.defaultValue 500

                let lines = readClaimLines dir since limit
                let mutable added = 0
                let mutable skipped = 0
                let errors = ResizeArray<string>()

                for line in lines do
                    match claimToNodeJson line with
                    | Some nodeJson ->
                        let t = GraphTools.graphAddNode nodeJson
                        let result = t.GetAwaiter().GetResult()
                        if result.StartsWith("graph_add_node error") then
                            errors.Add result
                            skipped <- skipped + 1
                        else
                            added <- added + 1
                    | None ->
                        skipped <- skipped + 1
                        errors.Add "Malformed claim line skipped"

                // Query current graph size via graph_stats-style count.
                // GraphTools doesn't expose the dictionary; use a query round-trip.
                let statsTask = GraphTools.graphStats ""
                let statsText = statsTask.GetAwaiter().GetResult()
                let nodeCountAfter =
                    // Parse "**Nodes:** N\n" from the stats markdown.
                    try
                        let marker = "**Nodes:** "
                        let idx = statsText.IndexOf(marker)
                        if idx < 0 then -1
                        else
                            let rest = statsText.Substring(idx + marker.Length)
                            let endIdx = rest.IndexOfAny([| '\n'; '\r'; ' ' |])
                            let numStr = if endIdx < 0 then rest else rest.Substring(0, endIdx)
                            match Int32.TryParse numStr with
                            | true, n -> n
                            | false, _ -> -1
                    with _ -> -1

                let response =
                    { ClaimsRead = lines.Length
                      NodesAdded = added
                      Skipped = skipped
                      Errors = (errors |> Seq.truncate 10 |> Seq.toList)
                      GraphNodeCountAfter = nodeCountAfter }
                Result.Ok(JsonSerializer.Serialize(response, jsonOptions))
        with ex ->
            Result.Error $"Failed to ingest chatbot claims: {ex.Message}"

    // =========================================================================
    // Tool registration
    // =========================================================================

    /// Create chatbot-claims bridge MCP tools.
    let createTools () : Tool list =
        [ { Name = "ingest_chatbot_claims"
            Description =
                "Import MCP tool-call claim traces from a ClaimTraceLog JSONL directory into the TARS knowledge graph as Claim-type nodes. Nodes are addressed by (tool, ts) so repeated ingest of the same file is idempotent on the graph (log grows, nodes overwrite in place). Input: {\"ClaimsDir\": \"C:/path/to/ga/state/claims\", \"SinceIso\": \"2026-04-19T00:00:00Z\", \"Limit\": 500}. ClaimsDir is required; SinceIso and Limit are optional (default: no time filter, 500 lines)."
            Version = "1.0"
            ParentVersion = None
            CreatedAt = DateTime.UtcNow
            Execute = fun input -> async { return ingestChatbotClaims input } } ]
