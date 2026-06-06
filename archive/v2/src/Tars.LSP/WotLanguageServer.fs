namespace Tars.LSP

open System
open System.Collections.Concurrent
open System.Threading
open System.Threading.Tasks
open System.Text.RegularExpressions
open OmniSharp.Extensions.LanguageServer.Protocol
open OmniSharp.Extensions.LanguageServer.Protocol.Models
open OmniSharp.Extensions.LanguageServer.Protocol.Client.Capabilities
open OmniSharp.Extensions.LanguageServer.Protocol.Document
open OmniSharp.Extensions.LanguageServer.Protocol.Server
open MediatR
open Tars.DSL.Wot

// =============================================================================
// Shared document store
// =============================================================================
module DocumentStore =
    /// Maps document URI -> array of lines
    let documents = ConcurrentDictionary<string, string[]>()

    /// Stores last parse result per URI
    let parseResults = ConcurrentDictionary<string, Result<DslWorkflow, ParseError list>>()

// =============================================================================
// Diagnostics publisher helper
// =============================================================================
module DiagnosticsPublisher =

    let publish (textDocument: ITextDocumentLanguageServer) (uri: DocumentUri) (text: string) =
        let lines = text.Split([| '\n' |]) |> Array.map (fun l -> l.TrimEnd('\r'))
        DocumentStore.documents.[uri.ToString()] <- lines

        let linesList = lines |> Array.toList
        let result = WotParser.parseLines linesList
        DocumentStore.parseResults.[uri.ToString()] <- result

        let diagnostics =
            match result with
            | Ok _ -> Array.empty<Diagnostic>
            | Error errors ->
                errors
                |> List.map (fun e ->
                    let line = max 0 (e.Line - 1) // LSP lines are 0-based
                    let diag = Diagnostic()
                    diag.Range <- Range(Position(line, 0), Position(line, 1000))
                    diag.Severity <- DiagnosticSeverity.Error
                    diag.Source <- "tars-wot"
                    diag.Message <- e.Message
                    diag)
                |> List.toArray

        let publishParams = PublishDiagnosticsParams()
        publishParams.Uri <- uri
        publishParams.Diagnostics <- Container<Diagnostic>(diagnostics)
        textDocument.PublishDiagnostics(publishParams)

// =============================================================================
// Text Document Sync Handler
// =============================================================================
type WotTextDocumentSyncHandler(textDocument: ITextDocumentLanguageServer) =
    inherit TextDocumentSyncHandlerBase()

    let selector = TextDocumentSelector.ForPattern("**/*.wot.trsx")

    override _.GetTextDocumentAttributes(uri: DocumentUri) =
        TextDocumentAttributes(uri, "wot-trsx")

    override _.CreateRegistrationOptions(_capability: TextSynchronizationCapability, _clientCapabilities: ClientCapabilities) =
        let opts = TextDocumentSyncRegistrationOptions()
        opts.DocumentSelector <- selector
        opts.Change <- TextDocumentSyncKind.Full
        opts.Save <- SaveOptions(IncludeText = true)
        opts

    override _.Handle(request: DidOpenTextDocumentParams, _ct: CancellationToken) =
        let uri = request.TextDocument.Uri
        let text = request.TextDocument.Text
        DiagnosticsPublisher.publish textDocument uri text
        Task.CompletedTask

    override _.Handle(request: DidChangeTextDocumentParams, _ct: CancellationToken) =
        let uri = request.TextDocument.Uri
        let changes = request.ContentChanges |> Seq.toArray
        if changes.Length > 0 then
            let text = changes.[changes.Length - 1].Text
            DiagnosticsPublisher.publish textDocument uri text
        Task.CompletedTask

    override _.Handle(request: DidSaveTextDocumentParams, _ct: CancellationToken) =
        let uri = request.TextDocument.Uri
        match request.Text with
        | null -> ()
        | text -> DiagnosticsPublisher.publish textDocument uri text
        Task.CompletedTask

    override _.Handle(_request: DidCloseTextDocumentParams, _ct: CancellationToken) =
        Task.CompletedTask

// =============================================================================
// Completion data
// =============================================================================
module CompletionData =

    let topLevelKeywords =
        [| "meta"; "inputs"; "policy"; "workflow" |]

    let metaKeys =
        [| "name"; "description"; "domain"; "difficulty"; "version"; "risk" |]

    let policyKeys =
        [| "allowed_tools"; "max_tool_calls"; "max_tokens"; "max_time_ms" |]

    let nodeKeys =
        [| "kind"; "goal"; "output"; "input"; "tool"; "args"; "checks";
           "verdict"; "invariants"; "constraints"; "transformation"; "agent" |]

    let kindValues =
        [| "reason"; "work" |]

    let transformationValues =
        [| "generate"; "aggregate"; "refine"; "contradict"; "distill"; "backtrack"; "score" |]

    let checkTypes =
        [| "non_empty"; "contains"; "regex"; "schema"; "tool_result" |]

// =============================================================================
// Completion Handler
// =============================================================================
type WotCompletionHandler() =
    inherit CompletionHandlerBase()

    let selector = TextDocumentSelector.ForPattern("**/*.wot.trsx")

    /// Determine the context by scanning backwards from the cursor line
    let determineContext (lines: string[]) (line: int) =
        let mutable depth = 0
        let mutable context = "top"

        for i in line .. -1 .. 0 do
            let trimmed = lines.[i].Trim()
            for c in trimmed do
                if c = '}' then depth <- depth + 1
                elif c = '{' then depth <- depth - 1

            if depth <= 0 then
                if trimmed.StartsWith("meta") && trimmed.Contains("{") then
                    context <- "meta"
                elif trimmed.StartsWith("inputs") && trimmed.Contains("{") then
                    context <- "inputs"
                elif trimmed.StartsWith("policy") && trimmed.Contains("{") then
                    context <- "policy"
                elif trimmed.StartsWith("workflow") && trimmed.Contains("{") then
                    context <- "workflow"
                elif trimmed.StartsWith("node") && trimmed.Contains("{") then
                    context <- "node"

        context

    let makeItem (label: string) (detail: string) (kind: CompletionItemKind) =
        let item = CompletionItem()
        item.Label <- label
        item.Detail <- detail
        item.Kind <- kind
        item

    override _.CreateRegistrationOptions(_capability: CompletionCapability, _clientCapabilities: ClientCapabilities) =
        let opts = CompletionRegistrationOptions()
        opts.DocumentSelector <- selector
        opts.TriggerCharacters <- Container<string>([| "="; " "; "\"" |])
        opts.ResolveProvider <- false
        opts

    override _.Handle(request: CompletionParams, _ct: CancellationToken) =
        let uri = request.TextDocument.Uri.ToString()
        let line = request.Position.Line

        let items = ResizeArray<CompletionItem>()

        match DocumentStore.documents.TryGetValue(uri) with
        | false, _ -> ()
        | true, lines ->
            let currentLine = if line < lines.Length then lines.[line] else ""
            let context = determineContext lines line

            match context with
            | "top" ->
                for kw in CompletionData.topLevelKeywords do
                    items.Add(makeItem kw "Top-level section" CompletionItemKind.Keyword)
            | "meta" ->
                for k in CompletionData.metaKeys do
                    items.Add(makeItem (k + " = ") ("Meta field: " + k) CompletionItemKind.Property)
            | "inputs" ->
                items.Add(makeItem "variable_name = " "Input variable" CompletionItemKind.Variable)
            | "policy" ->
                for k in CompletionData.policyKeys do
                    items.Add(makeItem (k + " = ") ("Policy field: " + k) CompletionItemKind.Property)
            | "workflow" ->
                items.Add(makeItem "node " "Define a workflow node" CompletionItemKind.Function)
                items.Add(makeItem "edge " "Define an edge between nodes" CompletionItemKind.Reference)
            | "node" ->
                let trimmedLine = currentLine.TrimStart()
                if trimmedLine.Contains("kind") && trimmedLine.Contains("=") then
                    for v in CompletionData.kindValues do
                        items.Add(makeItem ("\"" + v + "\"") ("Node kind: " + v) CompletionItemKind.EnumMember)
                elif trimmedLine.Contains("transformation") && trimmedLine.Contains("=") then
                    for v in CompletionData.transformationValues do
                        items.Add(makeItem ("\"" + v + "\"") ("GoT transformation: " + v) CompletionItemKind.EnumMember)
                else
                    for k in CompletionData.nodeKeys do
                        items.Add(makeItem (k + " = ") ("Node field: " + k) CompletionItemKind.Property)
            | _ -> ()

        Task.FromResult(CompletionList(items))

    /// Resolve handler (required by CompletionHandlerBase) - returns item unchanged
    override _.Handle(request: CompletionItem, _ct: CancellationToken) =
        Task.FromResult(request)

// =============================================================================
// Hover data
// =============================================================================
module HoverData =

    let docs =
        dict [
            // Top-level sections
            "meta", "**meta** block\n\nDefines workflow metadata: name, description, domain, difficulty, version, and risk level."
            "inputs", "**inputs** block\n\nDeclares input variables that the workflow expects. Variables are referenced as `${variable_name}` in node goals and tools."
            "policy", "**policy** block\n\nSets execution constraints: allowed tools, max tool calls, token limits, and time limits."
            "workflow", "**workflow** block\n\nContains the graph of nodes and edges that define the execution flow."

            // Meta fields
            "name", "**name** (meta)\n\nThe unique identifier/name of this workflow."
            "description", "**description** (meta)\n\nHuman-readable description of what this workflow does."
            "domain", "**domain** (meta)\n\nThe problem domain (e.g., 'puzzle', 'incident-response', 'code-review')."
            "difficulty", "**difficulty** (meta)\n\nDifficulty rating (e.g., 'easy', 'medium', 'hard')."
            "version", "**version** (meta)\n\nSemantic version of this workflow definition."
            "risk", "**risk** (meta)\n\nRisk level: 'low', 'medium', or 'high'. Affects safety policy enforcement."

            // Node kinds
            "reason", "**reason** node\n\nA reasoning/thinking step. Uses LLM inference to analyze, plan, or decide. Has a `goal` that describes what to reason about, and optionally `invariants` and `constraints`."
            "work", "**work** node\n\nAn action/execution step. Calls a tool to perform concrete work. Requires `tool` and optionally `args`. Can have `checks` to validate output."

            // GoT Transformations
            "generate", "**generate** transformation\n\nCreates new thoughts/ideas from scratch. The starting point of a Graph-of-Thoughts chain."
            "aggregate", "**aggregate** transformation\n\nCombines multiple thoughts into a unified result. Merges outputs from parallel reasoning branches."
            "refine", "**refine** transformation\n\nImproves an existing thought iteratively. Takes previous output and enhances quality or accuracy."
            "contradict", "**contradict** transformation\n\nFinds contradictions or counterarguments. Used for adversarial reasoning and robustness checking."
            "distill", "**distill** transformation\n\nSummarizes or compresses information. Extracts key insights from verbose reasoning."
            "backtrack", "**backtrack** transformation\n\nUndoes a failed reasoning path and tries an alternative approach."
            "score", "**score** transformation\n\nEvaluates the quality of a thought using defined metrics."

            // Node fields
            "goal", "**goal** (node)\n\nThe objective for a reason node. Describes what the LLM should think about or analyze."
            "output", "**output** (node)\n\nThe variable name where this node stores its result. Referenced by downstream nodes as `${variable_name}`."
            "input", "**input** (node)\n\nList of variable names this node depends on. These must be outputs of upstream nodes."
            "tool", "**tool** (node)\n\nThe tool to invoke for a work node (e.g., 'read_file', 'web_search', 'execute_code')."
            "args", "**args** (node)\n\nArguments passed to the tool as a JSON-like object: `args = { \"key\": \"value\" }`."
            "checks", "**checks** (node)\n\nValidation rules applied to node output. Supported types: non_empty, contains, regex, schema, tool_result."
            "verdict", "**verdict** (node)\n\nThe final determination or decision variable for a reason node."
            "invariants", "**invariants** (node)\n\nConditions that must remain true throughout node execution."
            "constraints", "**constraints** (node)\n\nLimitations or boundaries on the node's reasoning or output."
            "transformation", "**transformation** (node)\n\nGoT transformation type: generate, aggregate, refine, contradict, distill, backtrack, or score."
            "agent", "**agent** (node)\n\nSpecifies which agent should execute this node. Routes by role name or agent ID."

            // Structural
            "edge", "**edge** declaration\n\nDefines a dependency between two nodes: `edge \"source\" -> \"target\"`. The target node executes after the source completes."
            "node", "**node** declaration\n\nDefines a workflow step. Syntax: `node \"id\" kind=\"reason\" { ... }` or `node \"id\" { ... }`."

            // Check types
            "non_empty", "**non_empty** check\n\nValidates that the output variable is not empty or null."
            "contains", "**contains** check\n\nValidates that the output contains a specific substring (needle)."
            "regex", "**regex** check\n\nValidates that the output matches a regular expression pattern."
            "schema", "**schema** check\n\nValidates that the output conforms to a JSON schema."
            "tool_result", "**tool_result** check\n\nRuns a tool and checks its output contains an expected value."
        ]

// =============================================================================
// Hover Handler
// =============================================================================
type WotHoverHandler() =
    inherit HoverHandlerBase()

    let selector = TextDocumentSelector.ForPattern("**/*.wot.trsx")

    override _.CreateRegistrationOptions(_capability: HoverCapability, _clientCapabilities: ClientCapabilities) =
        let opts = HoverRegistrationOptions()
        opts.DocumentSelector <- selector
        opts

    override _.Handle(request: HoverParams, _ct: CancellationToken) =
        let uri = request.TextDocument.Uri.ToString()
        let line = request.Position.Line
        let col = request.Position.Character

        let result =
            match DocumentStore.documents.TryGetValue(uri) with
            | false, _ -> null
            | true, lines ->
                if line >= lines.Length then null
                else
                    let lineText = lines.[line]
                    // Extract the word under the cursor
                    let mutable wordStart = col
                    let mutable wordEnd = col

                    while wordStart > 0 && (Char.IsLetterOrDigit(lineText.[wordStart - 1]) || lineText.[wordStart - 1] = '_') do
                        wordStart <- wordStart - 1

                    while wordEnd < lineText.Length && (Char.IsLetterOrDigit(lineText.[wordEnd]) || lineText.[wordEnd] = '_') do
                        wordEnd <- wordEnd + 1

                    if wordStart >= wordEnd then null
                    else
                        let word = lineText.Substring(wordStart, wordEnd - wordStart)
                        match HoverData.docs.TryGetValue(word) with
                        | true, doc ->
                            let hover = Hover()
                            hover.Contents <- MarkedStringsOrMarkupContent(MarkupContent(Kind = MarkupKind.Markdown, Value = doc))
                            hover.Range <- Range(Position(line, wordStart), Position(line, wordEnd))
                            hover
                        | false, _ -> null

        Task.FromResult(result)

// =============================================================================
// Document Symbol Handler
// =============================================================================
type WotDocumentSymbolHandler() =
    inherit DocumentSymbolHandlerBase()

    let selector = TextDocumentSelector.ForPattern("**/*.wot.trsx")

    override _.CreateRegistrationOptions(_capability: DocumentSymbolCapability, _clientCapabilities: ClientCapabilities) =
        let opts = DocumentSymbolRegistrationOptions()
        opts.DocumentSelector <- selector
        opts

    override _.Handle(request: DocumentSymbolParams, _ct: CancellationToken) =
        let uri = request.TextDocument.Uri.ToString()
        let symbols = ResizeArray<SymbolInformationOrDocumentSymbol>()

        match DocumentStore.documents.TryGetValue(uri) with
        | false, _ -> ()
        | true, lines ->
            for i in 0 .. lines.Length - 1 do
                let trimmed = lines.[i].Trim()
                let lineLen = lines.[i].Length

                // Meta section
                if trimmed.StartsWith("meta") && trimmed.Contains("{") then
                    let sym = DocumentSymbol()
                    sym.Name <- "meta"
                    sym.Kind <- SymbolKind.Namespace
                    sym.Range <- Range(Position(i, 0), Position(i, lineLen))
                    sym.SelectionRange <- Range(Position(i, 0), Position(i, 4))
                    symbols.Add(SymbolInformationOrDocumentSymbol(sym))

                // Workflow section
                elif trimmed.StartsWith("workflow") && trimmed.Contains("{") then
                    let sym = DocumentSymbol()
                    sym.Name <- "workflow"
                    sym.Kind <- SymbolKind.Module
                    sym.Range <- Range(Position(i, 0), Position(i, lineLen))
                    sym.SelectionRange <- Range(Position(i, 0), Position(i, 8))
                    symbols.Add(SymbolInformationOrDocumentSymbol(sym))

                // Node declarations
                elif trimmed.StartsWith("node") && trimmed.Contains("{") then
                    let m = Regex.Match(trimmed, @"node\s+""?([^""\s\{]+)""?")
                    if m.Success then
                        let nodeId = m.Groups.[1].Value
                        let kindMatch = Regex.Match(trimmed, @"kind\s*=\s*""?(\w+)""?")
                        let detail = if kindMatch.Success then kindMatch.Groups.[1].Value else "node"
                        let sym = DocumentSymbol()
                        sym.Name <- nodeId
                        sym.Detail <- detail
                        sym.Kind <- SymbolKind.Function
                        sym.Range <- Range(Position(i, 0), Position(i, lineLen))
                        sym.SelectionRange <- Range(Position(i, 0), Position(i, lineLen))
                        symbols.Add(SymbolInformationOrDocumentSymbol(sym))

                // Edge declarations
                elif trimmed.StartsWith("edge") then
                    let m = Regex.Match(trimmed, @"edge\s+""?([^""\s]+)""?\s*->\s*""?([^""\s]+)""?")
                    if m.Success then
                        let label = sprintf "%s -> %s" m.Groups.[1].Value m.Groups.[2].Value
                        let sym = DocumentSymbol()
                        sym.Name <- label
                        sym.Kind <- SymbolKind.Property
                        sym.Range <- Range(Position(i, 0), Position(i, lineLen))
                        sym.SelectionRange <- Range(Position(i, 0), Position(i, lineLen))
                        symbols.Add(SymbolInformationOrDocumentSymbol(sym))

                // Policy section
                elif trimmed.StartsWith("policy") && trimmed.Contains("{") then
                    let sym = DocumentSymbol()
                    sym.Name <- "policy"
                    sym.Kind <- SymbolKind.Namespace
                    sym.Range <- Range(Position(i, 0), Position(i, lineLen))
                    sym.SelectionRange <- Range(Position(i, 0), Position(i, 6))
                    symbols.Add(SymbolInformationOrDocumentSymbol(sym))

                // Inputs section
                elif trimmed.StartsWith("inputs") && trimmed.Contains("{") then
                    let sym = DocumentSymbol()
                    sym.Name <- "inputs"
                    sym.Kind <- SymbolKind.Namespace
                    sym.Range <- Range(Position(i, 0), Position(i, lineLen))
                    sym.SelectionRange <- Range(Position(i, 0), Position(i, 6))
                    symbols.Add(SymbolInformationOrDocumentSymbol(sym))

        Task.FromResult(SymbolInformationOrDocumentSymbolContainer(symbols))
