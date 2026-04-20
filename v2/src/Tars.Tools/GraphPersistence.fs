namespace Tars.Tools.Graph

open System
open System.IO
open System.Collections.Concurrent
open System.Text.Json

/// Append-only JSONL persistence for the TARS knowledge graph.
///
/// Design rationale: see ix/docs/plans/2026-04-19-tars-graph-persistence.md.
/// Summary: logs at state/graph/nodes.jsonl + state/graph/edges.jsonl are the
/// authoritative store; the in-memory ConcurrentDictionaries in GraphTools are
/// a cache rebuilt on process start from the logs.
module GraphPersistence =

    let private schemaVersion = 1

    /// Root directory for graph storage. Defaults to `<cwd>/state/graph`.
    /// Override via `TARS_GRAPH_DIR` env var (primarily for tests).
    let private graphDir =
        let envDir = Environment.GetEnvironmentVariable("TARS_GRAPH_DIR")
        let dir =
            if not (String.IsNullOrWhiteSpace envDir) then envDir
            else Path.Combine(Directory.GetCurrentDirectory(), "state", "graph")
        Directory.CreateDirectory(dir) |> ignore
        dir

    let private nodesLogPath = Path.Combine(graphDir, "nodes.jsonl")
    let private edgesLogPath = Path.Combine(graphDir, "edges.jsonl")

    /// Single writer lock for both log files. Expected write rate is low
    /// (tens of ops/second at peak); a lock-free design would be premature.
    let private writeLock = obj ()

    /// Append a single JSONL envelope line: {"op":"...","ts":"iso8601","v":1,"data":{...}}
    let private writeEnvelope (logPath: string) (op: string) (payloadRaw: string) =
        lock writeLock (fun () ->
            let ts = DateTimeOffset.UtcNow.ToString("O")
            let line = sprintf "{\"op\":\"%s\",\"ts\":\"%s\",\"v\":%d,\"data\":%s}\n" op ts schemaVersion payloadRaw
            File.AppendAllText(logPath, line))

    /// Record an add_node operation. `payload` is the full node JSON (including id/type/label).
    let appendAddNode (payload: JsonElement) =
        writeEnvelope nodesLogPath "add_node" (payload.GetRawText())

    /// Record an add_edge operation. `payload` is the full edge JSON (including source/target/type).
    let appendAddEdge (payload: JsonElement) =
        writeEnvelope edgesLogPath "add_edge" (payload.GetRawText())

    /// Record a clear-all marker in a log. Log stays for audit; dictionary truncation happens in caller.
    let appendClearNodes () = writeEnvelope nodesLogPath "clear_all" "null"
    let appendClearEdges () = writeEnvelope edgesLogPath "clear_all" "null"

    /// Replay a log file, mutating the provided dictionary. Tolerates malformed lines.
    /// `keyExtract` pulls the dictionary key from the data payload (id for nodes; "src->tgt" for edges).
    let private replayLog
        (logPath: string)
        (store: ConcurrentDictionary<string, JsonElement>)
        (keyExtract: JsonElement -> string option)
        =
        if not (File.Exists logPath) then
            0
        else
            let mutable replayed = 0
            for line in File.ReadAllLines logPath do
                if not (String.IsNullOrWhiteSpace line) then
                    try
                        let doc = JsonDocument.Parse line
                        let root = doc.RootElement
                        let mutable opProp = Unchecked.defaultof<JsonElement>
                        let mutable dataProp = Unchecked.defaultof<JsonElement>
                        if root.TryGetProperty("op", &opProp) then
                            match opProp.GetString() with
                            | "clear_all" ->
                                store.Clear()
                                replayed <- replayed + 1
                            | "add_node"
                            | "add_edge" ->
                                if root.TryGetProperty("data", &dataProp) then
                                    match keyExtract dataProp with
                                    | Some key ->
                                        store.[key] <- dataProp.Clone()
                                        replayed <- replayed + 1
                                    | None -> ()
                            | _ -> () // forward-compat: unknown ops skipped
                    with _ -> () // malformed line — skip, don't abort replay
            replayed

    let private nodeKey (data: JsonElement) =
        let mutable p = Unchecked.defaultof<JsonElement>
        if data.TryGetProperty("id", &p) then Some(p.GetString()) else None

    let private edgeKey (data: JsonElement) =
        let mutable srcP = Unchecked.defaultof<JsonElement>
        let mutable tgtP = Unchecked.defaultof<JsonElement>
        if data.TryGetProperty("source", &srcP) && data.TryGetProperty("target", &tgtP) then
            Some(sprintf "%s->%s" (srcP.GetString()) (tgtP.GetString()))
        else
            None

    /// Hydrate both in-memory stores from the logs. Call once at process start,
    /// before any MCP tool is invoked. Idempotent: calling twice produces the
    /// same final state (log is deterministic).
    let hydrate
        (nodes: ConcurrentDictionary<string, JsonElement>)
        (edges: ConcurrentDictionary<string, JsonElement>)
        =
        let nodeOps = replayLog nodesLogPath nodes nodeKey
        let edgeOps = replayLog edgesLogPath edges edgeKey
        eprintfn
            "[TARS Graph] Hydrated %d nodes, %d edges (replayed %d node-ops, %d edge-ops) from %s"
            nodes.Count
            edges.Count
            nodeOps
            edgeOps
            graphDir

    /// Exposed for tests + diagnostics only.
    let internalPaths () = {| NodesLog = nodesLogPath; EdgesLog = edgesLogPath; GraphDir = graphDir |}
