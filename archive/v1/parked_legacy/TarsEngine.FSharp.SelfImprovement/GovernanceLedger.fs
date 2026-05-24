namespace TarsEngine.FSharp.SelfImprovement

open System
open System.Collections.Generic
open System.IO
open System.Text.Json
open Microsoft.Extensions.Logging

module GovernanceLedger =

    [<CLIMutable>]
    type CommandSnapshot =
        { Name: string
          Stage: string
          ExitCode: int
          DurationSeconds: float
          StartedAt: DateTime
          CompletedAt: DateTime }

    let private serializerOptions =
        let options = JsonSerializerOptions(PropertyNamingPolicy = JsonNamingPolicy.CamelCase, WriteIndented = true)
        options

    let private sanitizeValue (value: obj) =
        match value with
        | null -> null
        | :? string
        | :? bool
        | :? int
        | :? int16
        | :? int32
        | :? int64
        | :? uint16
        | :? uint32
        | :? uint64 as primitive -> box primitive
        | :? double as value -> box value
        | :? float32 as single -> box (float single)
        | :? decimal as dec -> box (float dec)
        | :? DateTime as dt -> box (dt.ToString("o"))
        | :? TimeSpan as span -> box span.TotalSeconds
        | :? Guid as guid -> box (guid.ToString("D"))
        | other -> box (other.ToString())

    let private sanitizeMetrics (metrics: Map<string, obj>) =
        let dict = Dictionary<string, obj>()
        metrics
        |> Map.iter (fun key value -> dict.[key] <- sanitizeValue value)
        dict

    let private sanitizeTelemetry (telemetry: Dictionary<string, obj>) =
        let dict = Dictionary<string, obj>()

        if not (isNull telemetry) then
            telemetry
            |> Seq.iter (fun kvp ->
                let key = kvp.Key
                let value = kvp.Value
                dict.[key] <- sanitizeValue value)

        dict

    let private ensureDirectory (path: string) =
        if not (Directory.Exists(path)) then
            Directory.CreateDirectory(path) |> ignore

    let private iterationsDirectory () =
        let root = Path.Combine(Environment.CurrentDirectory, ".specify")
        ensureDirectory root
        let ledgerRoot = Path.Combine(root, "ledger")
        ensureDirectory ledgerRoot
        let iterations = Path.Combine(ledgerRoot, "iterations")
        ensureDirectory iterations
        iterations

    let private safeFileName (timestamp: DateTime) (runId: Guid) =
        let timeStampText = timestamp.ToString("yyyyMMddHHmmss")
        $"{timeStampText}_{runId:N}.json"

    let recordIteration
        (logger: ILogger)
        (memoryEntry: PersistentAdaptiveMemory.MemoryEntry)
        (status: string)
        (metrics: Map<string, obj>)
        (artifacts: string list)
        (commands: CommandSnapshot list)
        (nextSteps: string list) =
        try
            let directory = iterationsDirectory ()
            let filePath = Path.Combine(directory, safeFileName memoryEntry.timestamp memoryEntry.runId)

            let metricPayload = sanitizeMetrics metrics
            let artifactPayload = artifacts |> List.distinct
            let commandPayload = commands |> List.toArray
            let nextStepsPayload = nextSteps |> List.distinct |> List.toArray

            let payload =
                {| runId = memoryEntry.runId
                   timestamp = memoryEntry.timestamp
                   specId = memoryEntry.specId
                   specPath = memoryEntry.specPath
                   description = memoryEntry.description
                   status = status
                   consensus = memoryEntry.consensus
                   critic = memoryEntry.critic
                   policyBefore = memoryEntry.policyBefore
                   policyAfter = memoryEntry.policyAfter
                   policyChanges = memoryEntry.policyChanges
                   agentFeedback = memoryEntry.agentFeedback
                   validatorFindings = memoryEntry.validatorFindings
                   validatorDisagreements = memoryEntry.validatorDisagreements
                   harness = memoryEntry.harness
                   inferenceTelemetry = sanitizeTelemetry memoryEntry.inferenceTelemetry
                   metrics = metricPayload
                   artifacts = artifactPayload
                   commands = commandPayload
                   nextSteps = nextStepsPayload
                   appendPath = PersistentAdaptiveMemory.getLastWritePath () |}

            let json = JsonSerializer.Serialize(payload, serializerOptions)
            File.WriteAllText(filePath, json)

            // Maintain a latest snapshot for quick inspection
            let latestPath = Path.Combine(directory, "latest.json")
            File.WriteAllText(latestPath, json)
        with ex ->
            logger.LogError(ex, "Failed to record governance ledger entry for run {RunId}", memoryEntry.runId)
