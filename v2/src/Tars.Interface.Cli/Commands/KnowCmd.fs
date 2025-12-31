module Tars.Interface.Cli.Commands.KnowCmd

open System
open System.IO
open System.Threading.Tasks
open Tars.Knowledge
open Spectre.Console
open System.Text.Json
open Tars.Llm

/// Options for the know command
type KnowOptions =
    { Command: string
      Subject: string option
      Predicate: string option
      Object: string option
      Path: string option
      Query: string option
      BeliefId: string option
      Depth: int
      UsePostgres: bool
      ShowPrompt: bool
      Verify: bool }

let defaultOptions =
    { Command = "status"
      Subject = None
      Predicate = None
      Object = None
      Path = None
      Query = None
      BeliefId = None
      Depth = 2
      UsePostgres = false
      ShowPrompt = false
      Verify = false }

let private createLedger (config: Tars.Core.TarsConfig) (usePostgres: bool) =
    if usePostgres then
        try
            let connStr =
                config.Memory.PostgresConnectionString
                |> Option.defaultValue (
                    Environment.GetEnvironmentVariable("TARS_POSTGRES_CONNECTION")
                    |> Option.ofObj
                    |> Option.defaultValue PostgresLedgerStorage.defaultConnectionString
                )


            let storage =
                PostgresLedgerStorage.createWithConnectionString connStr :> ILedgerStorage

            KnowledgeLedger(storage)
        with ex ->
            AnsiConsole.MarkupLine($"[yellow]⚠️ Postgres unavailable ({ex.Message}), using in-memory[/]")
            KnowledgeLedger.createInMemory ()
    else
        KnowledgeLedger.createInMemory ()

let private parseRelation (s: string) : RelationType =
    match s.ToLowerInvariant() with
    | "isa"
    | "is_a"
    | "is-a" -> IsA
    | "partof"
    | "part_of"
    | "part-of" -> PartOf
    | "hasproperty"
    | "has_property"
    | "has-property" -> HasProperty
    | "supports" -> Supports
    | "contradicts" -> Contradicts
    | "derivedfrom"
    | "derived_from"
    | "derived-from" -> DerivedFrom
    | "causes" -> Causes
    | "prevents" -> Prevents
    | "enables" -> Enables
    | "precedes" -> Precedes
    | "supersedes" -> Supersedes
    | "mentions" -> Mentions
    | "cites" -> Cites
    | "implements" -> Implements
    | other -> Custom other

let private printBelief (b: Belief) =
    let validity = if b.IsValid then "[green]✓[/]" else "[red]✗[/]"

    let confColor =
        if b.Confidence >= 0.8 then "green"
        elif b.Confidence >= 0.5 then "yellow"
        else "red"

    let idPart = $"[grey][[{b.Id}]][/] "
    let subjectPart = $"[cyan]{b.Subject.Value}[/] "
    let predicatePart = $"[magenta]{b.Predicate}[/] "
    let objectPart = $"[cyan]{b.Object.Value}[/] "
    let confPart = $"[{confColor}][[{b.Confidence:P0}]][/]"
    AnsiConsole.MarkupLine($"{idPart}{validity} {subjectPart}{predicatePart}{objectPart}{confPart}")

let private runStatus (ledger: KnowledgeLedger) =
    task {
        let stats = ledger.Stats()
        let table = Table().Border(TableBorder.Rounded)
        table.Title <- TableTitle("[bold blue]📊 TARS Knowledge Ledger Status[/]")
        table.AddColumn("[grey]Metric[/]") |> ignore
        table.AddColumn("[white]Value[/]") |> ignore

        table.AddRow("Valid Beliefs", stats.ValidBeliefs.ToString()) |> ignore
        table.AddRow("Total Beliefs", stats.TotalBeliefs.ToString()) |> ignore
        table.AddRow("Contradictions", $"[red]{stats.Contradictions}[/]") |> ignore
        table.AddRow("Avg Confidence", $"{stats.AverageConfidence:P1}") |> ignore
        table.AddRow("Unique Subjects", stats.UniqueSubjects.ToString()) |> ignore
        table.AddRow("Unique Objects", stats.UniqueObjects.ToString()) |> ignore

        AnsiConsole.Write(table)

        if not (List.isEmpty stats.ByPredicate) then
            let predTable = Table().Border(TableBorder.Minimal)
            predTable.AddColumn("[grey]Predicate[/]") |> ignore
            predTable.AddColumn("[white]Count[/]") |> ignore

            for (pred, count) in stats.ByPredicate do
                predTable.AddRow(pred.ToString(), count.ToString()) |> ignore

            AnsiConsole.Write(predTable)
    }

let private runAssert (ledger: KnowledgeLedger) (subject: string) (predicate: string) (obj: string) =
    task {
        let rel = parseRelation predicate
        let provenance = Provenance.FromUser()
        let! result = ledger.AssertTriple(subject, rel, obj, provenance, AgentId.User)

        match result with
        | Ok beliefId ->
            AnsiConsole.MarkupLine($"[green]✓ Asserted:[/] [white]{beliefId}[/]")
            AnsiConsole.MarkupLine($"  ([cyan]{subject}[/] [magenta]{rel}[/] [cyan]{obj}[/])")
        | Error e -> AnsiConsole.MarkupLine($"[red]✗ Failed:[/] {e}")
    }

let private runQuery
    (ledger: KnowledgeLedger)
    (subject: string option)
    (predicate: string option)
    (obj: string option)
    =
    task {
        let rel = predicate |> Option.map parseRelation

        let results =
            ledger.Query(?subject = subject, ?predicate = rel, ?obj = obj) |> Seq.toList

        if results.IsEmpty then
            AnsiConsole.MarkupLine("[yellow]No beliefs found matching query.[/]")
        else
            AnsiConsole.MarkupLine($"[blue]🔍 Found {results.Length} beliefs:[/]\n")

            for belief in results do
                printBelief belief
    }

let private runNeighborhood (ledger: KnowledgeLedger) (entity: string) (depth: int) =
    task {
        let results = ledger.GetNeighborhood(entity, depth) |> Seq.toList

        if results.IsEmpty then
            AnsiConsole.MarkupLine($"[yellow]No beliefs found around entity '{entity}'[/]")
        else
            AnsiConsole.MarkupLine(
                $"[blue]🌐 Neighborhood of '{entity}' (depth {depth}):[/] {results.Length} beliefs\n"
            )

            for belief in results do
                printBelief belief
    }

let private runContradictions (ledger: KnowledgeLedger) =
    task {
        let contradictions = ledger.GetContradictions() |> Seq.toList

        if contradictions.IsEmpty then
            AnsiConsole.MarkupLine("[green]✓ No contradictions found[/]")
        else
            AnsiConsole.MarkupLine($"[red]⚠️ Found {contradictions.Length} contradictions:[/]\n")

            for (b1, b2) in contradictions do
                AnsiConsole.MarkupLine("[bold red]Conflict discovered:[/]")
                AnsiConsole.Markup("  ")
                printBelief b1
                AnsiConsole.Markup("  ")
                printBelief b2
                AnsiConsole.WriteLine()
    }

let internal runIngest (ledger: KnowledgeLedger) (path: string) =
    task {
        if not (File.Exists path) then
            AnsiConsole.MarkupLine($"[red]File not found:[/] {path}")
        else
            AnsiConsole.MarkupLine($"[blue]📥 Ingesting from:[/] [white]{path}[/]")
            let lines = File.ReadAllLines(path)
            let mutable count = 0

            do!
                AnsiConsole
                    .Status()
                    .StartAsync(
                        "Ingesting beliefs...",
                        fun _ ->
                            task {
                                for line in lines do
                                    let line = line.Trim()

                                    if not (String.IsNullOrEmpty line) && not (line.StartsWith "#") then
                                        let parts = line.Split(',') |> Array.map (fun s -> s.Trim())

                                        if parts.Length >= 3 then
                                            let subject, predicate, obj = parts.[0], parts.[1], parts.[2]

                                            let confidence =
                                                if parts.Length >= 4 then
                                                    (match Double.TryParse(parts.[3]) with
                                                     | true, c -> c
                                                     | _ -> 1.0)
                                                else
                                                    1.0

                                            let rel = parseRelation predicate

                                            let provenance =
                                                { Provenance.FromExternal(Uri(Path.GetFullPath(path)), None, confidence) with
                                                    Confidence = confidence }

                                            let! result =
                                                ledger.AssertTriple(subject, rel, obj, provenance, AgentId "ingest")

                                            if result.IsOk then
                                                count <- count + 1

                                return ()
                            }
                    )

            AnsiConsole.MarkupLine($"\n[green]✓ Ingested {count} beliefs[/]")
    }

let internal tryParseBeliefId (ledger: KnowledgeLedger) (beliefIdStr: string) =
    match Guid.TryParse(beliefIdStr) with
    | true, guid -> Ok(BeliefId guid)
    | false, _ ->
        if beliefIdStr.StartsWith("b:", StringComparison.OrdinalIgnoreCase) then
            let matches =
                ledger.Query()
                |> Seq.filter (fun b -> b.Id.ToString().Equals(beliefIdStr, StringComparison.OrdinalIgnoreCase))
                |> Seq.toList

            match matches with
            | [ belief ] -> Ok belief.Id
            | [] -> Error $"No belief found with id {beliefIdStr}. Use a full GUID if available."
            | _ -> Error $"Multiple beliefs matched {beliefIdStr}. Use a full GUID."
        else
            Error $"Invalid belief ID: {beliefIdStr}"

let private runHistory (ledger: KnowledgeLedger) (beliefIdStr: string) =
    task {
        match tryParseBeliefId ledger beliefIdStr with
        | Ok beliefId ->
            let! events = ledger.GetHistory beliefId

            if events.IsEmpty then
                AnsiConsole.MarkupLine($"[yellow]No history found for belief {beliefId}[/]")
            else
                AnsiConsole.MarkupLine($"[blue]📜 History for {beliefId}:[/]\n")

                for entry in events do
                    let ts = entry.Timestamp.ToString("yyyy-MM-dd HH:mm:ss")
                    AnsiConsole.MarkupLine($"  [grey][{ts}][/] [white]{entry.Event}[/]")
        | Error message -> AnsiConsole.MarkupLine($"[red]{message}[/]")
    }

let private runFetch (ledger: KnowledgeLedger) (topic: string) =
    task {
        AnsiConsole.MarkupLine($"\n[blue]📚 Fetching Wikipedia summary for:[/] [white]{topic}[/]")
        use client = new System.Net.Http.HttpClient()
        client.DefaultRequestHeaders.Add("User-Agent", "TARS/2.0 (Autonomous Agent; Knowledge Ingestion Pipeline)")
        let encoded = Uri.EscapeDataString(topic.Replace(" ", "_"))
        let apiUrl = $"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"

        try
            let! response = client.GetAsync(apiUrl)

            if not response.IsSuccessStatusCode then
                AnsiConsole.MarkupLine($"[red]✗ Wikipedia article not found:[/] {topic}")
            else
                let! json = response.Content.ReadAsStringAsync()
                use doc = JsonDocument.Parse(json)
                let root = doc.RootElement

                let title =
                    let mutable prop = Unchecked.defaultof<JsonElement>

                    if root.TryGetProperty("title", &prop) then
                        prop.GetString()
                    else
                        topic

                let extract =
                    let mutable prop = Unchecked.defaultof<JsonElement>

                    if root.TryGetProperty("extract", &prop) then
                        prop.GetString()
                    else
                        ""

                let pageUrl =
                    let mutable urls = Unchecked.defaultof<JsonElement>

                    if root.TryGetProperty("content_urls", &urls) then
                        let mutable desktop = Unchecked.defaultof<JsonElement>

                        if urls.TryGetProperty("desktop", &desktop) then
                            let mutable page = Unchecked.defaultof<JsonElement>

                            if desktop.TryGetProperty("page", &page) then
                                page.GetString()
                            else
                                ""
                        else
                            ""
                    else
                        ""

                let sourceUrl =
                    if String.IsNullOrWhiteSpace pageUrl then
                        $"https://en.wikipedia.org/wiki/{encoded}"
                    else
                        pageUrl

                match ledger.Storage with
                | :? IEvidenceStorage as store ->
                    let hash =
                        use sha = System.Security.Cryptography.SHA256.Create()
                        let bytes = System.Text.Encoding.UTF8.GetBytes(json)

                        sha.ComputeHash(bytes)
                        |> fun b -> BitConverter.ToString(b).Replace("-", "").ToLower()

                    let candidate =
                        { Id = Guid.NewGuid()
                          SourceUrl = Uri(sourceUrl)
                          ContentHash = hash
                          FetchedAt = DateTime.UtcNow
                          RawContent = json
                          Segments = [ extract ]
                          ProposedAssertions = []
                          Status = Pending
                          Metadata = Map [ "title", title; "topic", topic ]
                          VerifiedAt = None
                          VerifiedBy = None
                          RejectionReason = None }

                    let! _ = store.SaveCandidate(candidate)
                    AnsiConsole.MarkupLine($"[grey]Evidence candidate saved ({candidate.Id})[/]")
                | _ -> ()

                AnsiConsole.Write(Panel(extract, Header = PanelHeader(title), Border = BoxBorder.Rounded))
                AnsiConsole.MarkupLine($"[grey]Source: {pageUrl}[/]")
        with ex ->
            AnsiConsole.MarkupLine($"[red]✗ Error fetching Wikipedia:[/] {ex.Message}")
    }

let private runPropose
    (ledger: KnowledgeLedger)
    (config: Tars.Core.TarsConfig)
    (topic: string)
    (showPrompt: bool)
    (verify: bool)
    =
    task {
        AnsiConsole.MarkupLine($"\n[blue]🧠 Extracting proposed beliefs for:[/] [white]{topic}[/]")
        use client = new System.Net.Http.HttpClient()
        client.DefaultRequestHeaders.Add("User-Agent", "TARS/2.0 (Autonomous Agent; Knowledge Ingestion Pipeline)")
        let encoded = Uri.EscapeDataString(topic.Replace(" ", "_"))
        let apiUrl = $"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
        let! response = client.GetAsync(apiUrl)

        if not response.IsSuccessStatusCode then
            AnsiConsole.MarkupLine($"[red]✗ Could not find text to extract from.[/]")
        else
            let! json = response.Content.ReadAsStringAsync()
            use doc = JsonDocument.Parse(json)

            let extract =
                let mutable prop = Unchecked.defaultof<JsonElement>

                if doc.RootElement.TryGetProperty("extract", &prop) then
                    prop.GetString()
                else
                    ""

            if String.IsNullOrWhiteSpace extract then
                AnsiConsole.MarkupLine("[yellow]! Text extract is empty.[/]")
            else
                let prompt =
                    $"""Extract a set of semantic triples (Subject, Predicate, Object) from the following text.
Text: {extract}
Return ONLY a JSON array of triples:
[ {{"subject": "...", "predicate": "...", "object": "...", "confidence": 0.9}} ]
"""

                if showPrompt then
                    AnsiConsole.MarkupLine("\n[grey]LLM PROMPT:[/]")
                    AnsiConsole.WriteLine(prompt)

                AnsiConsole.MarkupLine("[bold yellow]Consulting LLM...[/]")

                // Directly execute task
                use llmClient = new System.Net.Http.HttpClient()

                let llmService =
                    Tars.Llm.LlmService.DefaultLlmService(
                        llmClient,
                        { Routing = Tars.Llm.Routing.RoutingConfig.fromTarsConfig config }
                    )
                    :> Tars.Llm.ILlmService

                let req =
                    { ModelHint = None
                      Model = None
                      SystemPrompt = Some "Extract triples from text."
                      MaxTokens = Some 1000
                      Temperature = Some 0.0
                      Stop = []
                      Messages = [ { Role = Role.User; Content = prompt } ]
                      Tools = []
                      ToolChoice = None
                      ResponseFormat = Some ResponseFormat.Json
                      Stream = false
                      JsonMode = true
                      Seed = None
                      ContextWindow = None }

                let! res = llmService.CompleteAsync(req)

                // Debug: show raw LLM output
                // Only show simplified preview in grey
                let preview =
                    if res.Text.Length > 200 then
                        res.Text.Substring(0, 200) + "..."
                    else
                        res.Text

                AnsiConsole.MarkupLine($"[grey]LLM returned: {Markup.Escape(preview)}[/]")

                let tryGetPropertyInsensitive name (elem: JsonElement) =
                    if elem.ValueKind = JsonValueKind.Object then
                        elem.EnumerateObject()
                        |> Seq.tryFind (fun p -> p.Name.Equals(name, StringComparison.OrdinalIgnoreCase))
                        |> Option.map (fun p -> p.Value)
                    else
                        None

                let tryGetArray (elem: JsonElement) =
                    match elem.ValueKind with
                    | JsonValueKind.Array -> Some(elem, false) // (array, isSingleObject)
                    | JsonValueKind.Object ->
                        // First check for wrapper keys
                        let wrapperKeys =
                            [ "triples"; "proposals"; "items"; "data"; "results"; "facts"; "assertions" ]

                        match
                            wrapperKeys
                            |> List.tryPick (fun name ->
                                match tryGetPropertyInsensitive name elem with
                                | Some value when value.ValueKind = JsonValueKind.Array -> Some value
                                | _ -> None)
                        with
                        | Some arr -> Some(arr, false)
                        | None ->
                            // Check if this is a single triple object (has subject AND predicate AND object)
                            let hasSubject = tryGetPropertyInsensitive "subject" elem |> Option.isSome

                            let hasPredicate = tryGetPropertyInsensitive "predicate" elem |> Option.isSome

                            let hasObject = tryGetPropertyInsensitive "object" elem |> Option.isSome

                            if hasSubject && hasPredicate && hasObject then
                                Some(elem, true) // Treat as single-item
                            else
                                None
                    | _ -> None

                let getString name (elem: JsonElement) =
                    match tryGetPropertyInsensitive name elem with
                    | Some value when value.ValueKind = JsonValueKind.String -> value.GetString()
                    | Some value when value.ValueKind = JsonValueKind.Number -> value.GetRawText()
                    | _ -> ""

                let getDouble name (elem: JsonElement) =
                    match tryGetPropertyInsensitive name elem with
                    | Some value when value.ValueKind = JsonValueKind.Number -> value.GetDouble()
                    | Some value when value.ValueKind = JsonValueKind.String ->
                        match Double.TryParse(value.GetString()) with
                        | true, v -> v
                        | _ -> 0.5
                    | _ -> 0.5

                match JsonParsing.tryParseElement res.Text with
                | Ok root ->
                    match tryGetArray root with
                    | None -> AnsiConsole.MarkupLine($"[yellow]No triples array found. Root kind: {root.ValueKind}[/]")
                    | Some(arrayOrSingle, isSingleObject) ->
                        let table = Table().Border(TableBorder.Rounded)
                        table.AddColumn("[cyan]Subject[/]") |> ignore
                        table.AddColumn("[magenta]Predicate[/]") |> ignore
                        table.AddColumn("[cyan]Object[/]") |> ignore
                        table.AddColumn("[yellow]Conf.[/]") |> ignore

                        if verify then
                            table.AddColumn("[white]Verification[/]") |> ignore

                        // Get items as sequence
                        let items =
                            if isSingleObject then
                                seq { arrayOrSingle } // Single object
                            else
                                arrayOrSingle.EnumerateArray() |> Seq.cast<JsonElement>

                        match ledger.Storage with
                        | :? IEvidenceStorage as store ->
                            for item in items do
                                let s = getString "subject" item
                                let p = getString "predicate" item
                                let o = getString "object" item
                                let c = getDouble "confidence" item

                                if
                                    not (String.IsNullOrWhiteSpace s)
                                    && not (String.IsNullOrWhiteSpace p)
                                    && not (String.IsNullOrWhiteSpace o)
                                then
                                    let verificationStatus =
                                        if verify then
                                            let verifier = VerifierAgent(ledger)

                                            let result =
                                                // Create a temporary ProposedAssertion for verification
                                                let pA =
                                                    { Id = Guid.NewGuid()
                                                      Subject = s
                                                      Predicate = p
                                                      Object = o
                                                      Confidence = c
                                                      SourceSection = extract
                                                      ExtractorAgent = AgentId "wikipedia-extractor"
                                                      ExtractedAt = DateTime.UtcNow }

                                                verifier.Verify(pA) |> Async.RunSynchronously // We are in a task block, but Verify is async. Ideally await it properly.

                                            match result with
                                            | Accepted _ -> "[green]✓ Verified[/]"
                                            | Denied r -> $"[red]✗ {r}[/]"
                                            | Conflict(_, _) -> "[bold red]✗ Conflict[/]"
                                        else
                                            ""

                                    if verify then
                                        table.AddRow(
                                            Markup.Escape(s),
                                            Markup.Escape(p),
                                            Markup.Escape(o),
                                            $"{c:P0}",
                                            verificationStatus
                                        )
                                        |> ignore
                                    else
                                        table.AddRow(Markup.Escape(s), Markup.Escape(p), Markup.Escape(o), $"{c:P0}")
                                        |> ignore

                                    let verificationStatus =
                                        if verify then
                                            let verifier = VerifierAgent(ledger)

                                            let result =
                                                // Create a temporary ProposedAssertion for verification
                                                let pA =
                                                    { Id = Guid.NewGuid()
                                                      Subject = s
                                                      Predicate = p
                                                      Object = o
                                                      Confidence = c
                                                      SourceSection = extract
                                                      ExtractorAgent = AgentId "wikipedia-extractor"
                                                      ExtractedAt = DateTime.UtcNow }

                                                verifier.Verify(pA) |> Async.RunSynchronously // We are in a task block, but Verify is async. Ideally await it properly.

                                            match result with
                                            | Accepted _ -> "[green]✓ Verified[/]"
                                            | Denied r -> $"[red]✗ {r}[/]"
                                            | Conflict(_, _) -> "[bold red]✗ Conflict[/]"
                                        else
                                            ""

                                    if verify then
                                        table.AddRow(
                                            Markup.Escape(s),
                                            Markup.Escape(p),
                                            Markup.Escape(o),
                                            $"{c:P0}",
                                            verificationStatus
                                        )
                                        |> ignore
                                    else
                                        table.AddRow(Markup.Escape(s), Markup.Escape(p), Markup.Escape(o), $"{c:P0}")
                                        |> ignore

                                    let proposal: ProposedAssertion =
                                        { Id = Guid.NewGuid()
                                          Subject = s
                                          Predicate = p
                                          Object = o
                                          Confidence = c
                                          SourceSection = extract
                                          ExtractorAgent = AgentId "wikipedia-extractor"
                                          ExtractedAt = DateTime.UtcNow }

                                    let! _ = store.SaveProposal(proposal, None)
                                    ()
                        | _ ->
                            for item in items do
                                let s = getString "subject" item
                                let p = getString "predicate" item
                                let o = getString "object" item
                                let c = getDouble "confidence" item

                                if
                                    not (String.IsNullOrWhiteSpace s)
                                    && not (String.IsNullOrWhiteSpace p)
                                    && not (String.IsNullOrWhiteSpace o)
                                then
                                    table.AddRow(Markup.Escape(s), Markup.Escape(p), Markup.Escape(o), $"{c:P0}")
                                    |> ignore

                        AnsiConsole.Write(table)

                        AnsiConsole.MarkupLine(
                            "\n[grey]These are PROPOSALS. Use 'tars know assert' to add them to the ledger.[/]"
                        )
                | Error err -> AnsiConsole.MarkupLine($"[red]✗ Failed to parse LLM output:[/] {err}")

                return ()
    }

let private runIngestRun (ledger: KnowledgeLedger) (runIdOrPath: string) =
    task {
        let runDir = 
            if Directory.Exists runIdOrPath then runIdOrPath
            else Path.Combine(".wot", "runs", runIdOrPath)
            
        if not (Directory.Exists runDir) then
            AnsiConsole.MarkupLine($"[red]Run directory not found:[/] {runDir}")
        else
            let summaryPath = Path.Combine(runDir, "run_summary.json")
            let planPath = Path.Combine(runDir, "plan.json")
            
            if not (File.Exists summaryPath) || not (File.Exists planPath) then
                 AnsiConsole.MarkupLine("[red]Missing run_summary.json or plan.json in run directory.[/]")
            else
                AnsiConsole.MarkupLine($"[blue]📥 Ingesting run analysis from:[/] [white]{runDir}[/]")
                
                let parseJson path = JsonDocument.Parse(File.ReadAllText(path))
                use summaryDoc = parseJson summaryPath
                use planDoc = parseJson planPath
                
                let tryGetPropertyInsensitive name (elem: JsonElement) =
                    if elem.ValueKind = JsonValueKind.Object then
                        elem.EnumerateObject()
                        |> Seq.tryFind (fun p -> p.Name.Equals(name, StringComparison.OrdinalIgnoreCase))
                        |> Option.map (fun p -> p.Value)
                    else
                        None

                let rootSum = summaryDoc.RootElement
                let rootPlan = planDoc.RootElement
                
                let runId = 
                    match tryGetPropertyInsensitive "RunId" rootSum with
                    | Some p -> p.GetString()
                    | None -> Path.GetFileName(runDir)
                    
                let goal = 
                    match tryGetPropertyInsensitive "Goal" rootPlan with
                    | Some p -> p.GetString()
                    | None -> "unknown_task"
                    
                let model = 
                    match tryGetPropertyInsensitive "Reasoner" rootSum with
                    | Some r -> 
                        match tryGetPropertyInsensitive "Model" r with
                        | Some m -> m.GetString()
                        | None -> "unknown_model"
                    | None -> "unknown_model"
                    
                let passed = 
                    match tryGetPropertyInsensitive "VerifyPassed" rootSum with
                    | Some p -> if p.ValueKind = JsonValueKind.True then true else false
                    | None -> false

                let durationStr = 
                     match tryGetPropertyInsensitive "DurationMs" rootSum with
                     | Some p -> p.GetInt64().ToString()
                     | None -> "0"
                     
                let prov = Provenance.FromExternal(Uri(Path.GetFullPath(summaryPath)), None, 1.0)
                let agent = AgentId.User
                
                let mutable count = 0
                
                let! res1 = ledger.AssertTriple(runId, parseRelation "executed", goal, prov, agent)
                if res1.IsOk then count <- count + 1
                
                let! res2 = ledger.AssertTriple(goal, Custom "execution_time_ms", durationStr, prov, agent)
                if res2.IsOk then count <- count + 1
                
                if passed && model <> "unknown_model" then
                    let! res3 = ledger.AssertTriple(model, Custom "can_solve", goal, prov, agent)
                    if res3.IsOk then count <- count + 1
                    let! res4 = ledger.AssertTriple(runId, Custom "status", "SUCCESS", prov, agent) 
                    if res4.IsOk then count <- count + 1
                else
                    let! resFail = ledger.AssertTriple(runId, Custom "status", "FAILURE", prov, agent)
                    if resFail.IsOk then count <- count + 1
                    
                AnsiConsole.MarkupLine($"[green]✓ Ingested {count} facts from run {runId}[/]")
    }

let private printHelp () =
    AnsiConsole.Write(Rule("[bold blue]TARS Knowledge Ledger[/]"))
    AnsiConsole.MarkupLine("\n[bold]Knowledge Access:[/]")
    AnsiConsole.MarkupLine("  [green]status[/] [grey]- Show statistics[/]")
    AnsiConsole.MarkupLine("  [green]query[/] [grey]- Search beliefs[/]")
    AnsiConsole.MarkupLine("  [green]neighborhood[/] [grey]- Get entity subgraph[/]")
    AnsiConsole.MarkupLine("  [green]history[/] [grey]- show belief events[/]")
    AnsiConsole.MarkupLine("\n[bold]Assertion & Ingestion:[/]")
    AnsiConsole.MarkupLine("  [green]assert[/] [grey]- add single triple[/]")
    AnsiConsole.MarkupLine("  [green]ingest[/] [grey]- load from CSV[/]")
    AnsiConsole.MarkupLine("\n[bold]Research & Discovery:[/]")
    AnsiConsole.MarkupLine("  [green]fetch[/] [grey]- get wiki summary[/]")
    AnsiConsole.MarkupLine("  [green]propose[/] [grey]- extract proposals[/]")
    AnsiConsole.MarkupLine("  [green]ingest-run[/] [grey]- ingest .wot run artifacts[/]")

let run (config: Tars.Core.TarsConfig) (options: KnowOptions) : Task<int> =
    task {
        let ledger = createLedger config options.UsePostgres

        try
            do! ledger.Initialize()
        with ex ->
            AnsiConsole.MarkupLine($"[red]✗ Ledger initialization failed:[/] {ex.Message}")

        try
            match options.Command.ToLowerInvariant() with
            | "status"
            | "stats" -> do! runStatus ledger
            | "assert"
            | "add" ->
                match options.Subject, options.Predicate, options.Object with
                | Some s, Some p, Some o -> do! runAssert ledger s p o
                | _ -> AnsiConsole.MarkupLine("[yellow]Usage: tars know assert <subject> <predicate> <object>[/]")
            | "query"
            | "search" -> do! runQuery ledger options.Subject options.Predicate options.Object
            | "neighborhood" ->
                match options.Subject with
                | Some e -> do! runNeighborhood ledger e options.Depth
                | None -> AnsiConsole.MarkupLine("[yellow]Usage: tars know neighborhood <entity>[/]")
            | "contradictions" -> do! runContradictions ledger
            | "ingest" ->
                match options.Path with
                | Some p -> do! runIngest ledger p
                | None -> AnsiConsole.MarkupLine("[yellow]Usage: tars know ingest <file.csv>[/]")
            | "history" ->
                match options.BeliefId with
                | Some id -> do! runHistory ledger id
                | None -> AnsiConsole.MarkupLine("[yellow]Usage: tars know history <belief-id>[/]")
            | "fetch" ->
                match options.Query with
                | Some t -> do! runFetch ledger t
                | None -> AnsiConsole.MarkupLine("[yellow]Usage: tars know fetch <topic>[/]")
            | "propose" ->
                match options.Query with
                | Some t -> do! runPropose ledger config t options.ShowPrompt options.Verify
                | None -> AnsiConsole.MarkupLine("[yellow]Usage: tars know propose <topic>[/]")
            | "ingest-run" ->
                match options.Path with
                | Some p -> do! runIngestRun ledger p
                | None -> 
                    match options.Query with // Allow using query arg as runid if path not set
                    | Some q -> do! runIngestRun ledger q
                    | None -> AnsiConsole.MarkupLine("[yellow]Usage: tars know ingest-run <run-id>[/]")
            | _ -> printHelp ()

            return 0
        with ex ->
            AnsiConsole.WriteException(ex)
            return 1
    }

let parseArgs (args: string[]) : KnowOptions =
    let mutable options = defaultOptions
    let mutable i = 0

    if args.Length > 0 then
        options <- { options with Command = args.[0] }
        i <- 1

    while i < args.Length do
        match args.[i] with
        | "--subject"
        | "-s" when i + 1 < args.Length ->
            options <-
                { options with
                    Subject = Some args.[i + 1] }

            i <- i + 2
        | "--predicate"
        | "-p" when i + 1 < args.Length ->
            options <-
                { options with
                    Predicate = Some args.[i + 1] }

            i <- i + 2
        | "--object"
        | "-o" when i + 1 < args.Length ->
            options <-
                { options with
                    Object = Some args.[i + 1] }

            i <- i + 2
        | "--path"
        | "-f" when i + 1 < args.Length ->
            options <-
                { options with
                    Path = Some args.[i + 1] }

            i <- i + 2
        | "--depth"
        | "-d" when i + 1 < args.Length ->
            (match Int32.TryParse(args.[i + 1]) with
             | true, d -> options <- { options with Depth = d }
             | _ -> ())

            i <- i + 2
        | "--postgres"
        | "--pg" ->
            options <- { options with UsePostgres = true }
            i <- i + 1
        | "--show-prompt" ->
            options <- { options with ShowPrompt = true }
            i <- i + 1
        | "--verify" ->
            options <- { options with Verify = true }
            i <- i + 1
        | arg when not (arg.StartsWith "-") ->
            match options.Command with
            | "history" ->
                if options.BeliefId.IsNone then
                    options <- { options with BeliefId = Some arg }
            | "ingest" ->
                if options.Path.IsNone then
                    options <- { options with Path = Some arg }
            | "fetch"
            | "propose"
            | "ingest-run" ->
                if options.Query.IsNone then
                    options <- { options with Query = Some arg }
            | "neighborhood"
            | "query" ->
                if options.Subject.IsNone then
                    options <- { options with Subject = Some arg }
            | "assert" ->
                if options.Subject.IsNone then
                    options <- { options with Subject = Some arg }
                elif options.Predicate.IsNone then
                    options <- { options with Predicate = Some arg }
                elif options.Object.IsNone then
                    options <- { options with Object = Some arg }
            | _ -> ()

            i <- i + 1
        | _ -> i <- i + 1

    options
