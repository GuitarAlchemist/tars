namespace TarsEngine.FSharp.Core.Services

open System
open System.Collections.Concurrent
open System.Globalization
open System.IO
open System.Text
open Microsoft.FSharp.Control
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Closures
open TarsEngine.FSharp.Core.Closures.UnifiedEvolutionaryClosureFactory
open TarsEngine.FSharp.Core.Metascript.FractalGrammarMetascripts
open TarsEngine.FSharp.Core.Metascript
open TarsEngine.FSharp.Core.Specs

/// Service providing a thin metascript façade over the evolutionary closure factory.
module MetascriptClosureIntegrationService =

    /// Supported execution modes for metascript commands.
    type ExecutionMode =
        | Synchronous
        | Asynchronous

    /// Parsed representation of a metascript closure command.
    type MetascriptClosureCommand =
        { CommandId: string
          ClosureType: string
          Name: string
          Parameters: Map<string, obj>
          Context: Map<string, obj>
          Mode: ExecutionMode }

    /// Result returned to the metascript host.
    type MetascriptClosureResult =
        { CommandId: string
          Success: bool
          Output: obj option
          OutputSummary: string
          Artifacts: string list
          EvolutionData: Map<string, obj>
          NextSteps: string list
          Error: string option
          ExecutionTime: TimeSpan }

    /// Metadata about DSL inputs supplied via metascript parameters.
    type DslResourceInfo =
        { Key: string
          Path: string
          Kind: string
          Content: string }

    type SpecContractInfo =
        { Path: string
          Summary: SpecKitSummary }

    /// Integration service bridging metascripts with the closure factory.
    type MetascriptClosureIntegrationService
        (
            logger: ILogger<MetascriptClosureIntegrationService>,
            closureFactory: UnifiedEvolutionaryClosureFactory
        ) =

        let activeCommands = ConcurrentDictionary<string, MetascriptClosureCommand>()
        let executionHistory = ConcurrentQueue<MetascriptClosureResult>()
        let fractalParser = FractalMetascriptParser()

        let determineResourceKind (path: string) =
            match Path.GetExtension(path).ToLowerInvariant() with
            | ".trsx" -> "trsx"
            | ".flux" -> "flux"
            | ".tars" -> "tars"
            | ".fsx" -> "fsx"
            | ".dsl" -> "dsl"
            | ".json" -> "json"
            | _ -> "text"

        let tryParseFractalScript (content: string) =
            try
                Some(fractalParser.ParseFractalMetascript(content))
            with
            | :? FormatException as ex ->
                logger.LogDebug(ex, "Failed to parse fractal metascript content; continuing with raw text.")
                None
            | :? ArgumentException as ex ->
                logger.LogDebug(ex, "Invalid fractal metascript content supplied.")
                None

        let toDynamicSpec (name: string) (block: FractalMetascriptBlock) (sourcePath: string option) (expectations: Map<string, string> option) =
            let rules =
                block.Rules
                |> List.map (fun rule ->
                    match rule with
                    | FractalRule.SpawnAgent(agentType, count, strategy) -> DynamicMetascriptRule.SpawnAgentRule(agentType, count, strategy)
                    | FractalRule.ConnectAgents(source, target, label) -> DynamicMetascriptRule.ConnectAgentsRule(source, target, label)
                    | FractalRule.EmitMetric(metric, value) -> DynamicMetascriptRule.EmitMetricRule(metric, value)
                    | FractalRule.RepeatPattern(name, depth) -> DynamicMetascriptRule.RepeatPatternRule(name, depth))

            { Name = name
              Rules = rules
              MaxDepth = block.MaxDepth
              SourcePath = sourcePath
              Expectations = expectations }

        let tryResolveDirectiveFile (key: string) (value: string) =
            let trimmed = value.Trim()
            if not (trimmed.StartsWith("@")) then
                None
            else
                let relativePath = trimmed.TrimStart('@').Trim()
                let resolvedPath =
                    if Path.IsPathRooted(relativePath) then
                        relativePath
                    else
                        Path.Combine(Directory.GetCurrentDirectory(), relativePath)
                        |> Path.GetFullPath

                if not (File.Exists(resolvedPath)) then
                    logger.LogWarning("DSL resource not found for key {Key}: {Path}", key, resolvedPath)
                    None
                else
                    try
                        let content = File.ReadAllText(resolvedPath)
                        Some(resolvedPath, content)
                    with
                    | ex ->
                        logger.LogError(ex, "Failed to load resource for key {Key}: {Path}", key, resolvedPath)
                        None

        let tryLoadDslResource (key: string) (value: string) =
            tryResolveDirectiveFile key value
            |> Option.map (fun (path, content) ->
                { Key = key
                  Path = path
                  Kind = determineResourceKind path
                  Content = content })

        let isSpecContractKey (key: string) =
            let lower = key.ToLowerInvariant()
            lower.Contains("spec_contract")
            || lower.Contains("feature_spec")
            || lower.Contains("specification")
            || lower.EndsWith(".spec")

        let tryLoadSpecContract (key: string) (value: string) =
            if not (isSpecContractKey key) then
                None
            else
                match tryResolveDirectiveFile key value with
                | None -> None
                | Some (path, content) ->
                    try
                        let summary = SpecKitParser.parse content (Path.GetExtension(path))
                        Some { Path = path; Summary = summary }
                    with
                    | ex ->
                        logger.LogWarning(
                            ex,
                            "Failed to parse Spec Kit contract for key {Key}: {Path}",
                            key,
                            path
                        )
                        None

        let tryGetSpecExpectations (context: Map<string, obj>) =
            match context |> Map.tryFind "dsl.spec.expectations" with
            | Some (:? Map<string, string> as expectations) -> Some expectations
            | _ -> None

        let splitKeyValuePair (segment: string) =
            let index = segment.IndexOf('=')
            if index < 0 then
                None
            else
                let key = segment.[0 .. index - 1].Trim()
                let value = segment.[index + 1 ..].Trim().Trim([|'"'; '\''|])
                if String.IsNullOrWhiteSpace(key) then None else Some(key, box value)

        let parseParameterSegments (segments: string array) =
            segments
            |> Array.choose splitKeyValuePair

        let formatOutput (output: obj option) =
            match output with
            | None -> "No output."
            | Some (:? string as text) when String.IsNullOrWhiteSpace(text) -> "Output produced an empty string."
            | Some (:? string as text) -> text
            | Some (:? GrammarEvolutionSummary as summary) ->
                let ranked =
                    summary.RankedGrammars
                    |> List.map (fun (name, score) ->
                        let formatted = score.ToString("F2", CultureInfo.InvariantCulture)
                        $"{name}:{formatted}")
                    |> fun entries -> String.Join(", ", entries)
                $"Grammar evolution • team={summary.TeamName} • goal=\"{summary.EvolutionGoal}\" • ranked={ranked}"
            | Some (:? CoordinationSummary as summary) ->
                let coverage =
                    summary.RoleCoverage
                    |> Map.toList
                    |> List.map (fun (role, count) -> $"{role}:{count}")
                    |> fun entries -> String.Join(", ", entries)
                $"Coordination • strategy={summary.Strategy} • score={summary.TeamScore:F2} • coverage={coverage}"
            | Some (:? FractalMetascriptArtifact as artifact) ->
                $"Fractal metascript • pattern={artifact.Pattern} • depth={artifact.Depth} • nodes={artifact.NodeCount}"
            | Some (:? DynamicMetascriptSummary as summary) ->
                let metricsText =
                    if summary.Metrics.IsEmpty then "none"
                    else
                        summary.Metrics
                        |> Map.toList
                        |> List.map (fun (key, value) -> $"{key}:{value:F2}")
                        |> fun items -> String.Join(", ", items)

                let spawnText =
                    summary.SpawnedAgents
                    |> List.map (fun (agentType, count, strategy) -> $"%A{agentType}×%d{count}@%A{strategy}")
                    |> fun items -> if items.IsEmpty then "none" else String.Join("; ", items)

                let connectionText =
                    summary.Connections
                    |> List.map (fun (source, target, label) -> $"{source}->{target} ({label})")
                    |> fun items -> if items.IsEmpty then "none" else String.Join("; ", items)

                let sourceLabel = summary.SourcePath |> Option.defaultValue "inline"

                $"Dynamic metascript • name={summary.Name} • rules={summary.RuleCount} • depth={summary.MaxDepth} • source={sourceLabel} • metrics=[{metricsText}] • spawn=[{spawnText}] • connections=[{connectionText}]"
            | Some other -> other.ToString()

        let toMetascriptResult (command: MetascriptClosureCommand) (closureResult: EvolutionaryClosureResult) =
            let baseEvolutionData = closureResult.EvolutionData
            let baseArtifacts = closureResult.GeneratedArtifacts
            let baseSteps = closureResult.NextEvolutionSteps

            let evolutionDataAfterDsl, artifactsAfterDsl, stepsAfterDsl =
                match closureResult.Output with
                | Some (:? DynamicMetascriptSummary as summary) ->
                    let evolutionDataWithMetrics =
                        summary.Metrics
                        |> Map.fold (fun acc key value -> acc |> Map.add($"dsl.metric.{key}") (box value)) baseEvolutionData

                    let artifactSummary =
                        $"Dynamic metascript summary recorded (%d{summary.SpawnedAgents.Length} spawn, %d{summary.Connections.Length} connections)"

                    let artifacts =
                        if baseArtifacts |> List.exists (fun entry -> entry = artifactSummary) then
                            baseArtifacts
                        else
                            baseArtifacts @ [ artifactSummary ]

                    let nextSteps =
                        baseSteps
                        |> List.append [ "Promote validated dynamic closure to registry if production-ready." ]
                        |> List.distinct

                    (evolutionDataWithMetrics, artifacts, nextSteps)

                | Some (:? FractalMetascriptArtifact as artifact) ->
                    let parsedBlock = tryParseFractalScript artifact.Script

                    let evolutionDataWithDsl =
                        baseEvolutionData
                        |> Map.add "dsl.metascript.pattern" (box artifact.Pattern)
                        |> Map.add "dsl.metascript.depth" (box artifact.Depth)
                        |> Map.add "dsl.metascript.node_count" (box artifact.NodeCount)
                        |> Map.add "dsl.metascript.branching_factor" (box artifact.BranchingFactor)
                        |> fun data ->
                            match parsedBlock with
                            | Some block ->
                                data
                                |> Map.add "dsl.metascript.rule_count" (box block.Rules.Length)
                                |> Map.add "dsl.metascript.max_depth" (box block.MaxDepth)
                            | None -> data

                    let artifactSummary =
                        match parsedBlock with
                        | Some block ->
                            $"Parsed fractal metascript '%s{block.Name}' with %d{block.Rules.Length} rules (max depth %d{block.MaxDepth})"
                        | None ->
                            $"Fractal metascript '%s{artifact.Pattern}' generated %d{artifact.NodeCount} nodes"

                    let artifacts =
                        if baseArtifacts |> List.exists (fun entry -> entry = artifactSummary) then baseArtifacts
                        else baseArtifacts @ [ artifactSummary ]

                    let nextSteps =
                        match parsedBlock with
                        | Some _ ->
                            baseSteps
                            |> List.append [ "Run DSL validator on generated metascript artifact."; "Integrate parsed metascript block into downstream coordination pipeline." ]
                            |> List.distinct
                        | None -> baseSteps

                    (evolutionDataWithDsl, artifacts, nextSteps)

                | Some (:? string as potentialDsl) when potentialDsl.Contains("SPAWN", StringComparison.OrdinalIgnoreCase) ->
                    let parsedBlock = tryParseFractalScript potentialDsl
                    match parsedBlock with
                    | Some block ->
                        let data =
                            baseEvolutionData
                            |> Map.add "dsl.metascript.rule_count" (box block.Rules.Length)
                            |> Map.add "dsl.metascript.max_depth" (box block.MaxDepth)

                        let artifacts =
                            baseArtifacts
                            |> List.append [ $"Parsed inline DSL block '%s{block.Name}' (%d{block.Rules.Length} rules)" ]

                        let steps =
                            baseSteps
                            |> List.append [ "Verify inline DSL block semantics against evolution constraints." ]
                            |> List.distinct

                        (data, artifacts, steps)
                    | None -> (baseEvolutionData, baseArtifacts, baseSteps)

                | _ -> (baseEvolutionData, baseArtifacts, baseSteps)

            let finalEvolutionData, finalArtifacts, finalSteps =
                match command.Context |> Map.tryFind "spec.contract.summary" with
                | Some (:? SpecKitSummary as summary) ->
                    let acceptanceCount =
                        summary.UserStories
                        |> List.collect (fun story -> story.AcceptanceCriteria)
                        |> List.length

                    let p1Count =
                        summary.UserStories
                        |> List.filter (fun story ->
                            story.Priority
                            |> Option.exists (fun priority -> priority.Trim().StartsWith("P1", StringComparison.OrdinalIgnoreCase)))
                        |> List.length

                    let status =
                        summary.Status
                        |> Option.map (fun s -> s.Trim())
                        |> Option.orElse (command.Context |> Map.tryFind "spec.contract.status" |> Option.map string)

                    let contractPath =
                        command.Context
                        |> Map.tryFind "spec.contract.path"
                        |> Option.map string

                    let dataWithSpec =
                        evolutionDataAfterDsl
                        |> Map.add "spec.contract.title" (box summary.Title)
                        |> Map.add "spec.contract.user_story_count" (box summary.UserStories.Length)
                        |> Map.add "spec.contract.acceptance_count" (box acceptanceCount)
                        |> Map.add "spec.contract.edge_case_count" (box summary.EdgeCases.Length)
                        |> Map.add "spec.contract.p1_story_count" (box p1Count)
                        |> (fun data ->
                            match status with
                            | Some s when not (String.IsNullOrWhiteSpace(s)) -> data |> Map.add "spec.contract.status" (box s)
                            | _ -> data)
                        |> (fun data ->
                            match contractPath with
                            | Some path -> data |> Map.add "spec.contract.path" (box path)
                            | None -> data)

                    let specArtifact =
                        $"Spec contract \"%s{summary.Title}\" captured (%d{summary.UserStories.Length} stories, %d{acceptanceCount} acceptance scenarios)"

                    let artifactsWithSpec =
                        if artifactsAfterDsl |> List.exists ((=) specArtifact) then artifactsAfterDsl
                        else artifactsAfterDsl @ [ specArtifact ]

                    let stepsWithSpec =
                        stepsAfterDsl
                        |> List.append [ "Validate acceptance scenarios defined in spec contract." ]
                        |> List.distinct

                    (dataWithSpec, artifactsWithSpec, stepsWithSpec)
                | _ -> (evolutionDataAfterDsl, artifactsAfterDsl, stepsAfterDsl)

            { CommandId = command.CommandId
              Success = closureResult.Success
              Output = closureResult.Output
              OutputSummary = formatOutput closureResult.Output
              Artifacts = finalArtifacts
              EvolutionData = finalEvolutionData
              NextSteps = finalSteps
              Error = closureResult.Error
              ExecutionTime = closureResult.ExecutionTime }

        let enqueueResult result =
            executionHistory.Enqueue(result)
            result

        member _.ParseClosureCommand(line: string) : MetascriptClosureCommand option =
            try
                let segments = line.Split(' ', StringSplitOptions.RemoveEmptyEntries)
                if segments.Length < 3 then
                    None
                else
                    let keyword = segments.[0].Trim().ToUpperInvariant()
                    if keyword <> "CLOSURE_CREATE" then
                        None
                    else
                        let closureType = segments.[1].Trim()
                        let name = segments.[2].Trim('"')
                        let rawParameters =
                            if segments.Length > 3 then parseParameterSegments segments.[3..] else Array.empty

                        let contextSeed: Map<string, obj> =
                            rawParameters
                            |> Array.choose (fun (key, value) ->
                                if key.StartsWith("context.", StringComparison.OrdinalIgnoreCase) then
                                    let contextKey = key.Substring("context.".Length).Trim()
                                    if String.IsNullOrWhiteSpace(contextKey) then None else Some(contextKey, value)
                                else
                                    None)
                            |> Map.ofArray

                        let executionMode =
                            rawParameters
                            |> Array.tryPick (fun (key, value) ->
                                if key.Equals("mode", StringComparison.OrdinalIgnoreCase) then
                                    let modeValue = string value
                                    if modeValue.Equals("async", StringComparison.OrdinalIgnoreCase)
                                       || modeValue.Equals("asynchronous", StringComparison.OrdinalIgnoreCase) then
                                        Some ExecutionMode.Asynchronous
                                    else
                                        Some ExecutionMode.Synchronous
                                else
                                    None)
                            |> Option.defaultValue ExecutionMode.Synchronous

                        let parameters, context =
                            rawParameters
                            |> Array.fold
                                (fun (paramAcc: Map<string, obj>, ctxAcc: Map<string, obj>) (key, value) ->
                                    if key.Equals("mode", StringComparison.OrdinalIgnoreCase)
                                       || key.StartsWith("context.", StringComparison.OrdinalIgnoreCase) then
                                        (paramAcc, ctxAcc)
                                    else
                                        match value with
                                        | :? string as str ->
                                            let mutable nextContext = ctxAcc
                                            let mutable nextValue: obj = box str

                                            let specContractOpt = tryLoadSpecContract key str

                                            match specContractOpt with
                                            | Some contract ->
                                                nextValue <- box contract.Summary
                                                logger.LogInformation(
                                                    "Loaded Spec Kit contract \"{Title}\" from {Path}",
                                                    contract.Summary.Title,
                                                    contract.Path
                                                )
                                                nextContext <-
                                                    let withBasic =
                                                        nextContext
                                                        |> Map.add "spec.contract.path" (box contract.Path)
                                                        |> Map.add "spec.contract.summary" (box contract.Summary)
                                                        |> Map.add "spec.contract.title" (box contract.Summary.Title)
                                                    let withStatus =
                                                        match contract.Summary.Status with
                                                        | Some status when not (String.IsNullOrWhiteSpace(status)) ->
                                                            withBasic |> Map.add "spec.contract.status" (box status)
                                                        | _ -> withBasic
                                                    withStatus

                                            | None ->
                                                match tryLoadDslResource key str with
                                                | Some resource ->
                                                    nextValue <- box resource.Content
                                                    let ctxWithResource =
                                                        nextContext
                                                        |> Map.add $"dsl.source.{key}" (box resource)

                                                    let ctxWithParsed =
                                                        if resource.Kind = "trsx"
                                                           || resource.Kind = "dsl"
                                                           || resource.Kind = "flux"
                                                           || key.ToLowerInvariant().Contains("metascript") then
                                                            match tryParseFractalScript resource.Content with
                                                            | Some block -> ctxWithResource |> Map.add $"dsl.parsed.{key}" (box block)
                                                            | None -> ctxWithResource
                                                        else
                                                            ctxWithResource

                                                    nextContext <- ctxWithParsed
                                                | None ->
                                                    if key.ToLowerInvariant().Contains("metascript") then
                                                        match tryParseFractalScript str with
                                                        | Some block ->
                                                            nextContext <- nextContext |> Map.add $"dsl.inline.{key}" (box block)
                                                        | None ->
                                                            ()

                                            (paramAcc |> Map.add key nextValue, nextContext)
                                        | _ ->
                                            (paramAcc |> Map.add key value, ctxAcc))
                                (Map.empty<string, obj>, contextSeed)

                        Some
                            { CommandId = Guid.NewGuid().ToString("N")[..7]
                              ClosureType = closureType
                              Name = name
                              Parameters = parameters
                              Context = context
                              Mode = executionMode }
            with
            | ex ->
                logger.LogError(ex, "Failed to parse metascript command: {Line}", line)
                None

        member this.ExecuteClosureCommand(command: MetascriptClosureCommand) =
            async {
                let command =
                    match command.Parameters |> Map.tryFind "spec" with
                    | Some (:? string as specPath) ->
                        let spec = MetascriptSpecLoader.loadFromFile specPath
                        let updatedContext =
                            command.Context
                            |> Map.add "dsl.spec.id" (box spec.Id)
                            |> Map.add "dsl.spec.path" (box spec.SourcePath)
                            |> Map.add "dsl.spec.expectations" (box spec.Expectations)
                            |> Map.add "dsl.spec.grammar" (box spec.Grammar)
                        { command with Context = updatedContext }
                    | _ -> command

                activeCommands.[command.CommandId] <- command
                logger.LogInformation(
                    "Executing metascript closure \"{Name}\" of type \"{Type}\" (mode {Mode})",
                    command.Name,
                    command.ClosureType,
                    command.Mode)

                let createPlaceholderResult message =
                    { CommandId = command.CommandId
                      Success = true
                      Output = None
                      OutputSummary = message
                      Artifacts = []
                      EvolutionData = Map.ofList [ ("status", box "pending") ]
                      NextSteps = [ "Await asynchronous completion." ]
                      Error = None
                      ExecutionTime = TimeSpan.Zero }

                let runClosure closure =
                    async {
                        try
                            try
                                let! closureResult = closure command.Parameters
                                let metascriptResult = toMetascriptResult command closureResult
                                logger.LogInformation(
                                    "Metascript closure \"{Name}\" ({Id}) completed with success={Success}",
                                    command.Name,
                                    command.CommandId,
                                    metascriptResult.Success)
                                return enqueueResult metascriptResult
                            with ex ->
                                logger.LogError(
                                    ex,
                                    "Error executing metascript closure \"{Name}\" ({Id})",
                                    command.Name,
                                    command.CommandId)
                                let errorResult =
                                    { CommandId = command.CommandId
                                      Success = false
                                      Output = None
                                      OutputSummary = ""
                                      Artifacts = []
                                      EvolutionData = Map.empty
                                      NextSteps = []
                                      Error = Some ex.Message
                                      ExecutionTime = TimeSpan.Zero }
                                return enqueueResult errorResult
                        finally
                            activeCommands.TryRemove(command.CommandId) |> ignore
                    }

                match this.MapToClosureType(command) with
                | None ->
                    activeCommands.TryRemove(command.CommandId) |> ignore
                    let result =
                        { CommandId = command.CommandId
                          Success = false
                          Output = None
                          OutputSummary = ""
                          Artifacts = []
                          EvolutionData = Map.empty
                          NextSteps = []
                          Error = Some $"Unsupported closure type \"{command.ClosureType}\"."
                          ExecutionTime = TimeSpan.Zero }
                    return enqueueResult result

                | Some closureType ->
                    let closure = closureFactory.CreateClosure(closureType, command.Name, command.Context)
                    match command.Mode with
                    | Synchronous ->
                        return! runClosure closure
                    | Asynchronous ->
                        let placeholder = createPlaceholderResult "Command scheduled for asynchronous execution."
                        let queuedPlaceholder = enqueueResult placeholder
                        do runClosure closure |> Async.Ignore |> Async.StartImmediate
                        return queuedPlaceholder
            }

        member _.GetExecutionHistory(?maximumCount: int) =
            let entries = executionHistory.ToArray()
            match maximumCount with
            | Some limit when limit > 0 -> entries |> Array.take (min limit entries.Length)
            | _ -> entries

        member private _.MapToClosureType(command: MetascriptClosureCommand) =
            match command.ClosureType.Trim().ToUpperInvariant() with
            | "GRAMMAR_EVOLUTION" ->
                let teamName =
                    command.Parameters
                    |> Map.tryFind "team_name"
                    |> Option.map string
                    |> Option.defaultValue "Unnamed Team"

                let goal =
                    command.Parameters
                    |> Map.tryFind "evolution_goal"
                    |> Option.map string
                    |> Option.defaultValue "Improve grammar quality"

                Some(GrammarEvolutionClosure(teamName, goal))

            | "MULTI_AGENT_COORDINATION" ->
                let strategy =
                    command.Parameters
                    |> Map.tryFind "strategy"
                    |> Option.map string
                    |> Option.map (fun value -> value.ToUpperInvariant())
                    |> Option.map (fun value ->
                        match value with
                        | "HIERARCHICAL" -> CoordinationStrategy.Hierarchical "leader"
                        | "DEMOCRATIC" -> CoordinationStrategy.Democratic
                        | "SPECIALIZED" -> CoordinationStrategy.Specialized
                        | "SWARM" -> CoordinationStrategy.Swarm
                        | "FRACTAL" -> CoordinationStrategy.FractalSelfOrganizing
                        | _ -> CoordinationStrategy.Hierarchical "leader")
                    |> Option.defaultValue (CoordinationStrategy.Hierarchical "leader")

                Some(MultiAgentCoordinationClosure(strategy))

            | "FRACTAL_METASCRIPT" ->
                let depth =
                    command.Parameters
                    |> Map.tryFind "depth"
                    |> Option.map (fun value -> int (string value))
                    |> Option.defaultValue 2

                let pattern =
                    command.Parameters
                    |> Map.tryFind "pattern"
                    |> Option.map string
                    |> Option.defaultValue "coordination"

                Some(FractalMetascriptGeneratorClosure(depth, pattern))

            | "DYNAMIC_METASCRIPT" ->
                let specExpectations = tryGetSpecExpectations command.Context
                let specFromMetadata =
                    command.Context
                    |> Map.tryFind "dsl.spec.grammar"
                    |> Option.bind (fun value ->
                        match tryParseFractalScript (string value) with
                        | Some block ->
                            let specName =
                                command.Context
                                |> Map.tryFind "dsl.spec.id"
                                |> Option.map string
                                |> Option.defaultValue command.Name
                            let sourcePath = command.Context |> Map.tryFind "dsl.spec.path" |> Option.map string
                            Some(toDynamicSpec specName block sourcePath specExpectations)
                        | None -> None)

                let parsedCandidates =
                    command.Context
                    |> Map.toList
                    |> List.choose (fun (key, value) ->
                        let keyLower = key.ToLowerInvariant()
                        if keyLower.StartsWith("dsl.parsed.") then
                            let token = key.Substring("dsl.parsed.".Length)
                            let block = value :?> FractalMetascriptBlock
                            let sourcePath =
                                command.Context
                                |> Map.tryFind($"dsl.source.{token}")
                                |> Option.map (fun info -> (info :?> DslResourceInfo).Path)
                            Some(block, sourcePath)
                        elif keyLower.StartsWith("dsl.inline.") then
                            let block = value :?> FractalMetascriptBlock
                            Some(block, None)
                        else
                            None)

                let dynamicSpecOpt =
                    match specFromMetadata with
                    | Some spec -> Some spec
                    | None ->
                        match parsedCandidates with
                        | (block, source) :: _ -> Some(toDynamicSpec command.Name block source specExpectations)
                        | [] ->
                            let grammarKeys = [ "grammar"; "metascript"; "spec"; "dsl" ]
                            grammarKeys
                            |> List.tryPick (fun key ->
                                command.Parameters
                                |> Map.tryFind key
                                |> Option.bind (fun value ->
                                    match value with
                                    | :? string as text -> tryParseFractalScript text
                                    | _ -> None))
                            |> Option.map (fun block -> toDynamicSpec command.Name block None specExpectations)

                match dynamicSpecOpt with
                | Some dynamicSpec -> Some(DynamicMetascriptClosure dynamicSpec)
                | None ->
                    logger.LogWarning("No parsed DSL content found for dynamic metascript closure '{Name}'.", command.Name)
                    None

            | _ ->
                None

        member _.TryGetActiveCommand(commandId: string) =
            match activeCommands.TryGetValue(commandId) with
            | true, command -> Some command
            | _ -> None

        member _.GetActiveCommands() =
            activeCommands.Values |> Seq.toList

        member _.FormatResultForTranscript(result: MetascriptClosureResult) =
            let builder = StringBuilder()
            let status = if result.Success then "SUCCESS" else "FAILED"
            builder.AppendLine $"Command %s{result.CommandId}: %s{status}" |> ignore
            builder.AppendLine $"  Output   : %s{result.OutputSummary}" |> ignore
            builder.AppendLine $"  Duration : %.0f{result.ExecutionTime.TotalMilliseconds} ms" |> ignore
            (result.Error |> Option.iter (fun error -> builder.AppendLine $"  Error    : %s{error}" |> ignore))

            if not result.Artifacts.IsEmpty then
                builder.AppendLine("  Artifacts:") |> ignore
                result.Artifacts |> List.iter (fun artifact -> builder.AppendLine $"    - %s{artifact}" |> ignore)

            if not result.EvolutionData.IsEmpty then
                builder.AppendLine("  Metrics  :") |> ignore
                result.EvolutionData
                |> Map.toList
                |> List.iter (fun (key, value) -> builder.AppendLine $"    - %s{key} = {value}" |> ignore)

            if not result.NextSteps.IsEmpty then
                builder.AppendLine("  Next Steps:") |> ignore
                result.NextSteps |> List.iter (fun step -> builder.AppendLine $"    - %s{step}" |> ignore)

            builder.ToString()

        member _.CreateSampleMetascript () =
            [ "#!/usr/bin/env trsx"
              "# Sample metascript generated by MetascriptClosureIntegrationService"
              $"# Generated: {DateTime.UtcNow:O}"
              ""
              "CLOSURE_CREATE GRAMMAR_EVOLUTION \"GrammarReview\" team_name=\"Systems Team\" evolution_goal=\"Reduce ambiguity\" grammars=\"@grammars.json\""
              "CLOSURE_CREATE MULTI_AGENT_COORDINATION \"Coordination\" strategy=\"Hierarchical\" agents=\"@agents.json\" mode=async context.session=\"demo\""
              "CLOSURE_CREATE FRACTAL_METASCRIPT \"FractalPlan\" depth=3 pattern=\"coordination\""
              "CLOSURE_CREATE DYNAMIC_METASCRIPT \"DynamicPlan\" grammar=\"@metascripts/fractal_coordination.trsx\""
              "" ]
            |> String.concat Environment.NewLine


