namespace Tars.DSL.Wot

open System
open System.Text.RegularExpressions
open Tars.Core.WorkflowOfThought

type ParseError = { Line: int; Message: string }

// Used for checks parsing
type DslCheck =
    | NonEmpty of WotValueRef
    | Contains of WotValueRef * needle: string

module WotParser =

    // ---------------------------
    // Helpers
    // ---------------------------
    let private trim (s: string) = if isNull s then "" else s.Trim()

    let private isBlank (s: string) = String.IsNullOrWhiteSpace s

    let private startsWith (prefix: string) (s: string) =
        s.TrimStart().StartsWith(prefix, StringComparison.Ordinal)

    let private stripQuotes (s: string) =
        let t = trim s

        if
            t.Length >= 2
            && ((t.StartsWith("\"") && t.EndsWith("\""))
                || (t.StartsWith("'") && t.EndsWith("'")))
        then
            t.Substring(1, t.Length - 2)
        else
            t

    let private parseKeyValue (line: string) : (string * string) option =
        // Supports: key = "value"   OR  key = value
        let m = Regex.Match(line, @"^\s*([A-Za-z0-9_\-]+)\s*=\s*(.+?)\s*$")

        if m.Success then
            Some(m.Groups.[1].Value, stripQuotes m.Groups.[2].Value)
        else
            None

    let private parseStringList (value: string) : string list =
        // Supports: ["a", "b"] OR [ "a" ; "b" ]  (very forgiving)
        let t = trim value

        if not (t.StartsWith("[") && t.EndsWith("]")) then
            []
        else
            t.Trim('[', ']').Split([| ','; ';' |], StringSplitOptions.RemoveEmptyEntries)
            |> Array.map trim
            |> Array.map stripQuotes
            |> Array.filter (fun x -> x <> "")
            |> Array.toList

    let private parseChecksList (lines: string list) (startLineNo: int) : Result<WotCheck list, ParseError list> =
        // Expected each item as JSON-ish object on a single line:
        // { "type": "non_empty", "value": "${analysis_md}" }
        // { "type": "contains", "value": "${analysis_md}", "needle": "| Cyclomatic Complexity |" }
        //
        // We'll implement only non_empty + contains for v0.
        let errors = ResizeArray<ParseError>()
        let checks = ResizeArray<WotCheck>()

        let rxType = Regex(@"""type""\s*:\s*""([^""]+)""", RegexOptions.Compiled)
        let rxValue = Regex(@"""value""\s*:\s*""([^""]+)""", RegexOptions.Compiled)
        let rxNeedle = Regex(@"""needle""\s*:\s*""([^""]+)""", RegexOptions.Compiled)
        let rxPattern = Regex(@"""pattern""\s*:\s*""([^""]+)""", RegexOptions.Compiled)
        let rxSchema = Regex(@"""schema""\s*:\s*""([^""]+)""", RegexOptions.Compiled)
        let rxTool = Regex(@"""tool""\s*:\s*""([^""]+)""", RegexOptions.Compiled)
        let rxArgs = Regex(@"""args""\s*:\s*(\{.*?\})""", RegexOptions.Compiled)
        let rxCheck = Regex(@"""check""\s*:\s*""([^""]+)""", RegexOptions.Compiled)

        for i, line in List.indexed lines do
            let ln = startLineNo + i

            if isBlank line then
                ()
            elif line.Trim().StartsWith("//") then
                ()
            else
                let mt = rxType.Match(line)
                let mv = rxValue.Match(line)

                if not mt.Success then
                    errors.Add(
                        { Line = ln
                          Message = $"Invalid check line (missing type): {line}" }
                    )
                else
                    let ctype = mt.Groups.[1].Value

                    let vref =
                        if mv.Success then
                            mv.Groups.[1].Value |> trim |> (fun x -> x.Trim([| '$'; '{'; '}' |]))
                        else
                            ""

                    match ctype with
                    | "non_empty" ->
                        if String.IsNullOrEmpty vref then
                            errors.Add(
                                { Line = ln
                                  Message = "non_empty check missing value" }
                            )
                        else
                            checks.Add(WotCheck.NonEmpty vref)
                    | "contains" ->
                        let mn = rxNeedle.Match(line)

                        if not mn.Success then
                            errors.Add(
                                { Line = ln
                                  Message = "contains check missing needle" }
                            )
                        else
                            checks.Add(WotCheck.Contains(vref, mn.Groups.[1].Value))
                    | "regex" ->
                        let mp = rxPattern.Match(line)

                        if not mp.Success then
                            errors.Add(
                                { Line = ln
                                  Message = "regex check missing pattern" }
                            )
                        else
                            checks.Add(WotCheck.RegexMatch(vref, mp.Groups.[1].Value))
                    | "schema" ->
                        let ms = rxSchema.Match(line)

                        if not ms.Success then
                            errors.Add(
                                { Line = ln
                                  Message = "schema check missing schema" }
                            )
                        else
                            checks.Add(WotCheck.SchemaMatch(vref, ms.Groups.[1].Value))
                    | "tool_result" ->
                        let mtool = rxTool.Match(line)
                        let mcheck = rxCheck.Match(line)

                        if not mtool.Success || not mcheck.Success then
                            errors.Add(
                                { Line = ln
                                  Message = "tool_result check missing tool or check needle" }
                            )
                        else
                            // Very basic arg parsing: find "args": { "key": "value" }
                            let mutable args = Map.empty
                            let margs = rxArgs.Match(line)

                            if margs.Success then
                                let argsJson = margs.Groups.[1].Value
                                // Naive key-value extraction from the small JSON block
                                let pairRx = Regex(@"""([^""]+)""\s*:\s*""([^""]+)""")
                                let ms = pairRx.Matches(argsJson)

                                for m in ms do
                                    args <- args.Add(m.Groups.[1].Value, m.Groups.[2].Value)

                            checks.Add(WotCheck.ToolResult(mtool.Groups.[1].Value, args, mcheck.Groups.[1].Value))
                    | other ->
                        errors.Add(
                            { Line = ln
                              Message = $"Unsupported check type '{other}'" }
                        )

        if errors.Count > 0 then
            Error(List.ofSeq errors)
        else
            Ok(List.ofSeq checks)

    // ---------------------------
    // Main parser
    // ---------------------------

    type private Section =
        | NoSection
        | MetaSection
        | InputsSection
        | PolicySection
        | WorkflowSection

    type private NodeBlock =
        { Id: string
          Kind: string
          Goal: string option
          Output: string option
          Input: string list
          Invariants: string list
          Constraints: string list
          Tool: string option
          Args: Map<string, string>
          ChecksLines: string list
          Verdict: string option
          Agent: string option
          Condition: string option }

    let private emptyNode id =
        { Id = id
          Kind = ""
          Goal = None
          Output = None
          Input = []
          Invariants = []
          Constraints = []
          Tool = None
          Args = Map.empty
          ChecksLines = []
          Verdict = None
          Agent = None
          Condition = None }

    let parseLines (lines: string list) : Result<DslWorkflow, ParseError list> =
            let errors = ResizeArray<ParseError>()

            // Accumulators
            let mutable section = Section.NoSection
            let mutable name = ""
            let mutable version = ""
            let mutable risk = "low"
            let mutable description = ""
            let mutable domain = ""
            let mutable difficulty = ""

            let mutable inputs: Map<string, string> = Map.empty

            let mutable allowedTools: Set<string> = Set.empty
            let mutable maxToolCalls = 0
            let mutable maxTokens = 0
            let mutable maxTimeMs = 0

            let nodes = ResizeArray<NodeBlock>()
            let edges = ResizeArray<string * string>()
            let parallelGroups = ResizeArray<ParallelGroup>()

            // Node parsing state
            let mutable inWorkflow = false
            let mutable currentNode: NodeBlock option = None
            let mutable inChecksArray = false
            let mutable checksBuffer = ResizeArray<string>()
            let mutable checksStartLine = 0

            // Parallel block parsing state
            let mutable inParallelBlock = false
            let mutable parallelNodeIds = ResizeArray<string>()
            let mutable parallelGroupCounter = 0

            let flushChecksIfNeeded (lineNo: int) =
                match currentNode with
                | None -> ()
                | Some nb when inChecksArray ->
                    // still open, don't flush
                    ()
                | Some nb ->
                    if checksBuffer.Count > 0 then
                        // attach to node
                        let updated =
                            { nb with
                                ChecksLines = List.ofSeq checksBuffer }

                        currentNode <- Some updated
                        checksBuffer.Clear()

            let flushNode (lineNo: int) =
                flushChecksIfNeeded lineNo

                match currentNode with
                | Some nb ->
                    nodes.Add nb
                    // Track node ID if inside a parallel block
                    if inParallelBlock then
                        parallelNodeIds.Add(nb.Id)
                    currentNode <- None
                    inChecksArray <- false
                    checksBuffer.Clear()
                | None -> ()

            for idx = 0 to lines.Length - 1 do
                let lineNo = idx + 1
                let raw = lines[idx]
                let line = trim raw

                if isBlank line || line.StartsWith("//") then
                    () // ignore
                else if
                    // Section switches
                    startsWith "meta" line && line.EndsWith("{")
                then
                    section <- Section.MetaSection
                elif startsWith "inputs" line && line.EndsWith("{") then
                    section <- Section.InputsSection
                elif startsWith "policy" line && line.EndsWith("{") then
                    section <- Section.PolicySection
                elif startsWith "workflow" line && line.EndsWith("{") then
                    section <- Section.WorkflowSection
                    inWorkflow <- true
                elif line = "}" then
                    // Close any open node when leaving workflow blocks or node blocks
                    // Node blocks end with "}" too; we rely on state
                    if inChecksArray then
                        // end of checks? (only if checks were last thing)
                        inChecksArray <- false
                    else if
                        // could be end of node block
                        currentNode.IsSome
                    then
                        flushNode lineNo
                    else if inParallelBlock then
                        // Closing a parallel block - finalize the group
                        if parallelNodeIds.Count > 0 then
                            parallelGroups.Add(
                                { GroupId = $"parallel_{parallelGroupCounter}"
                                  NodeIds = List.ofSeq parallelNodeIds })
                        inParallelBlock <- false
                        parallelNodeIds.Clear()
                    // If we were in Meta/Inputs/Policy, closing brace returns to none unless workflow keeps us.
                    elif section <> Section.WorkflowSection then
                        section <- Section.NoSection
                else
                    match section with
                    | Section.MetaSection ->
                        match parseKeyValue line with
                        | Some("name", v) -> name <- v
                        | Some("version", v) -> version <- v
                        | Some("risk", v) -> risk <- v
                        | Some("description", v) -> description <- v
                        | Some("domain", v) -> domain <- v
                        | Some("difficulty", v) -> difficulty <- v
                        | Some _ -> () // ignore unknown meta fields v0
                        | None ->
                            errors.Add(
                                { Line = lineNo
                                  Message = $"Invalid meta line: {raw}" }
                            )

                    | Section.InputsSection ->
                        match parseKeyValue line with
                        | Some(k, v) -> inputs <- inputs.Add(k, v)
                        | None ->
                            errors.Add(
                                { Line = lineNo
                                  Message = $"Invalid inputs line: {raw}" }
                            )

                    | Section.PolicySection ->
                        match parseKeyValue line with
                        | Some("allowed_tools", v) -> allowedTools <- parseStringList v |> Set.ofList
                        | Some("max_tool_calls", v) ->
                            match Int32.TryParse v with
                            | true, n -> maxToolCalls <- n
                            | _ ->
                                errors.Add(
                                    { Line = lineNo
                                      Message = $"Invalid max_tool_calls: {raw}" }
                                )
                        | Some("max_tokens", v) ->
                            match Int32.TryParse v with
                            | true, n -> maxTokens <- n
                            | _ ->
                                errors.Add(
                                    { Line = lineNo
                                      Message = $"Invalid max_tokens: {raw}" }
                                )
                        | Some("max_time_ms", v) ->
                            match Int32.TryParse v with
                            | true, n -> maxTimeMs <- n
                            | _ ->
                                errors.Add(
                                    { Line = lineNo
                                      Message = $"Invalid max_time_ms: {raw}" }
                                )
                        | Some _ -> () // ignore v0
                        | None ->
                            errors.Add(
                                { Line = lineNo
                                  Message = $"Invalid policy line: {raw}" }
                            )

                    | Section.WorkflowSection ->
                        // PARALLEL block: parallel { ... }
                        if startsWith "parallel" line && line.EndsWith("{") then
                            flushNode lineNo
                            inParallelBlock <- true
                            parallelNodeIds.Clear()
                            parallelGroupCounter <- parallelGroupCounter + 1

                        // Flexible node header: node "id" kind="kind" agent="Agent" { OR node id {
                        elif startsWith "node" line && line.Contains("{") then
                            flushNode lineNo

                            // Try full header with agent: node "id" kind="kind" agent="agent" {
                            let mWithAgent =
                                Regex.Match(
                                    line,
                                    @"node\s+""?([^""\s]+)""?\s+kind\s*=\s*""?([^""\s]+)""?\s+agent\s*=\s*""?([^""\s]+)""?\s*\{"
                                )

                            if mWithAgent.Success then
                                let id = mWithAgent.Groups.[1].Value
                                let kind = mWithAgent.Groups.[2].Value
                                let agent = mWithAgent.Groups.[3].Value

                                currentNode <-
                                    Some
                                        { (emptyNode id) with
                                            Kind = kind
                                            Agent = Some agent }
                            else
                                // Try full header without agent: node "id" kind="kind" {
                                let mLong =
                                    Regex.Match(line, @"node\s+""?([^""\s]+)""?\s+kind\s*=\s*""?([^""\s]+)""?\s*\{")

                                if mLong.Success then
                                    let id = mLong.Groups.[1].Value
                                    let kind = mLong.Groups.[2].Value
                                    currentNode <- Some { (emptyNode id) with Kind = kind }
                                else
                                    // Try short header: node "id" {
                                    let mShort = Regex.Match(line, @"node\s+""?([^""\s\{\}]+)""?\s*\{")

                                    if mShort.Success then
                                        let id = mShort.Groups.[1].Value
                                        currentNode <- Some(emptyNode id) // Defaults to "reason"
                                    else
                                        errors.Add(
                                            { Line = lineNo
                                              Message = $"Invalid node header: {raw}" }
                                        )


                        // edge "a" -> "b" OR edge a -> b
                        elif startsWith "edge" line then
                            let m = Regex.Match(line, @"edge\s+""?([^""\s]+)""?\s*->\s*""?([^""\s]+)""?")

                            if not m.Success then
                                errors.Add(
                                    { Line = lineNo
                                      Message = $"Invalid edge: {raw}" }
                                )
                            else
                                edges.Add(m.Groups.[1].Value, m.Groups.[2].Value)

                        else
                            // inside a node block
                            match currentNode with
                            | None ->
                                // workflow-level lines we ignore for v0
                                ()
                            | Some nb ->
                                // checks array begins?
                                if startsWith "checks" line && line.EndsWith("[") then
                                    inChecksArray <- true
                                    checksStartLine <- lineNo + 1
                                elif inChecksArray then
                                    if line = "]" then
                                        inChecksArray <- false
                                        // attach checks to node now
                                        let updated =
                                            { nb with
                                                ChecksLines = List.ofSeq checksBuffer }

                                        currentNode <- Some updated
                                        checksBuffer.Clear()
                                    else
                                        checksBuffer.Add(raw) // keep raw for easier JSON-ish parsing
                                else
                                    // parse node key-values
                                    match parseKeyValue line with
                                    | Some("kind", v) -> currentNode <- Some { nb with Kind = v }
                                    | Some("goal", v) -> currentNode <- Some { nb with Goal = Some v }
                                    | Some("output", v) -> currentNode <- Some { nb with Output = Some v }
                                    | Some("input", v) ->
                                        // supports ["a","b"] or "a"
                                        let ins = if v.StartsWith("[") then parseStringList v else [ v ]
                                        currentNode <- Some { nb with Input = ins }
                                    | Some("invariants", v) ->
                                        currentNode <-
                                            Some
                                                { nb with
                                                    Invariants = parseStringList v }
                                    | Some("constraints", v) ->
                                        currentNode <-
                                            Some
                                                { nb with
                                                    Constraints = parseStringList v }
                                    | Some("tool", v) -> currentNode <- Some { nb with Tool = Some v }
                                    | Some("verdict", v) -> currentNode <- Some { nb with Verdict = Some v }
                                    | Some("condition", v) -> currentNode <- Some { nb with Condition = Some v }
                                    | Some("args", vRaw) ->
                                        // args = { "k": "v", "k2": "v2" }
                                        try
                                            let content = vRaw.Trim().TrimStart('{').TrimEnd('}').Trim()

                                            if content.Length > 0 then
                                                // Naive split by ',' but respecting quotes
                                                let pairs = ResizeArray<string * string>()
                                                let mutable currentPart = ""
                                                let mutable inQuote = false

                                                for c in content do
                                                    if c = '"' then
                                                        inQuote <- not inQuote
                                                        currentPart <- currentPart + string c
                                                    elif c = ',' && not inQuote then
                                                        // Split!
                                                        let p = currentPart.Trim()

                                                        if p.Length > 0 then
                                                            // kv pair: "k": "v"
                                                            let m = Regex.Match(p, @"""([^""]+)""\s*:\s*""([^""]+)""")

                                                            if m.Success then
                                                                pairs.Add(m.Groups.[1].Value, m.Groups.[2].Value)
                                                            else
                                                                errors.Add(
                                                                    { Line = lineNo
                                                                      Message = $"Invalid arg pair: {p}" }
                                                                )

                                                        currentPart <- ""
                                                    else
                                                        currentPart <- currentPart + string c

                                                // Last part
                                                let p = currentPart.Trim()

                                                if p.Length > 0 then
                                                    let m = Regex.Match(p, @"""([^""]+)""\s*:\s*""([^""]+)""")

                                                    if m.Success then
                                                        pairs.Add(m.Groups.[1].Value, m.Groups.[2].Value)
                                                    else
                                                        errors.Add(
                                                            { Line = lineNo
                                                              Message = $"Invalid arg pair: {p}" }
                                                        )

                                                // Update node
                                                let newArgs =
                                                    pairs
                                                    |> Seq.fold
                                                        (fun (acc: Map<string, string>) (k, v) -> acc.Add(k, v))
                                                        nb.Args

                                                currentNode <- Some { nb with Args = newArgs }
                                            else
                                                // Empty args {}
                                                ()
                                        with _ ->
                                            errors.Add(
                                                { Line = lineNo
                                                  Message = $"Failed to parse args object: {raw}" }
                                            )
                                    | Some _ -> () // ignore unknown in v0
                                    | None ->
                                        errors.Add(
                                            { Line = lineNo
                                              Message = $"Invalid node line: {raw}" }
                                        )

                    | Section.NoSection -> () // ignore stray lines v0

            // flush last node
            flushNode (lines.Length + 1)

            // Convert Nodes to DSL AST
            let wfNodes =
                nodes
                |> Seq.map (fun nb ->
                    // Convert checks lines -> WotCheck list
                    let checksResult =
                        if nb.ChecksLines.IsEmpty then
                            Ok []
                        else
                            parseChecksList nb.ChecksLines checksStartLine

                    match checksResult with
                    | Error es ->
                        es |> List.iter errors.Add
                        // still produce empty checks to keep going (errors will fail result)
                        { Id = nb.Id
                          Kind = if nb.Kind = "work" then NodeKind.Work else NodeKind.Reason
                          Name = nb.Id
                          Title = None
                          Inputs = nb.Input
                          Outputs = nb.Output |> Option.map (fun s -> [ SimpleOutput s ]) |> Option.defaultValue []
                          Tool = nb.Tool
                          Args =
                            if nb.Args.IsEmpty then
                                None
                            else
                                Some(nb.Args |> Map.map (fun _ v -> box v))
                          Checks = []
                          Goal = nb.Goal
                          Invariants = nb.Invariants
                          Constraints = nb.Constraints
                          Verdict = nb.Verdict
                          Condition = nb.Condition
                          Agent =
                            match nb.Agent with
                            | Some a -> ByRole a
                            | None -> Default
                          Transformation = None
                          StructuredOutput = None
                          RequiresEvidence = false
                          EvidenceRefs = []
                          Metadata = Map.empty }
                    | Ok ck ->
                        { Id = nb.Id
                          Kind = if nb.Kind = "work" then NodeKind.Work else NodeKind.Reason
                          Name = nb.Id
                          Title = None
                          Inputs = nb.Input
                          Outputs = nb.Output |> Option.map (fun s -> [ SimpleOutput s ]) |> Option.defaultValue []
                          Tool = nb.Tool
                          Args =
                            if nb.Args.IsEmpty then
                                None
                            else
                                Some(nb.Args |> Map.map (fun _ v -> box v))
                          Checks = ck
                          Goal = nb.Goal
                          Invariants = nb.Invariants
                          Constraints = nb.Constraints
                          Verdict = nb.Verdict
                          Condition = nb.Condition
                          Agent =
                            match nb.Agent with
                            | Some a -> ByRole a
                            | None -> Default
                          Transformation = None
                          StructuredOutput = None
                          RequiresEvidence = false
                          EvidenceRefs = []
                          Metadata = Map.empty })
                |> Seq.toList

            let wfEdges = edges |> Seq.map (fun (a, b) -> a, b) |> Seq.toList

            // Build graph
            let graph: DslWorkflow =
                { Name = if name = "" then "unnamed" else name
                  Version = if version = "" then "0.0.0" else version
                  Description = if description = "" then None else Some description
                  Domain = if domain = "" then None else Some domain
                  Difficulty = if difficulty = "" then None else Some difficulty
                  Risk = risk
                  Inputs = inputs
                  Policy =
                    { AllowedTools = allowedTools
                      MaxToolCalls = maxToolCalls
                      MaxTokens = maxTokens
                      MaxTimeMs = maxTimeMs }
                  Nodes = wfNodes
                  Edges = wfEdges
                  ParallelGroups = List.ofSeq parallelGroups }

            if errors.Count > 0 then
                Error(List.ofSeq errors)
            else
                Ok graph

    let parseFile (filePath: string) : Result<DslWorkflow, ParseError list> =
        if not (System.IO.File.Exists filePath) then
            Error
                [ { Line = 0
                    Message = $"File not found: {filePath}" } ]
        else
            let lines = System.IO.File.ReadAllLines(filePath) |> Array.toList
            parseLines lines
