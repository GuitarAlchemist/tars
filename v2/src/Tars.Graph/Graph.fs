namespace Tars.Graph

open Tars.Core
open Tars.Connectors
open Tars.Tools
open System.Threading.Tasks
open System.Text
open System
open Tars.Llm
open Tars.Llm.LlmService
open Tars.Kernel

module ToolGrammar =
    let spec =
        GrammarDistill.fromJsonExamples [ """{"name":"tool_name","arguments":{"param":"value"}}""" ]

module PromptBuilder =
    /// Formats tool descriptions for the system prompt
    let private formatTools (tools: Tool list) =
        if tools.IsEmpty then
            ""
        else
            let sb = StringBuilder()
            sb.AppendLine("\n## Available Tools") |> ignore
            sb.AppendLine("You have access to the following tools:\n") |> ignore

            for tool in tools do
                sb.AppendLine($"### {tool.Name}") |> ignore
                sb.AppendLine($"  {tool.Description}") |> ignore

            sb.AppendLine("\n## Tool Call Format") |> ignore
            sb.AppendLine("To use a tool, respond with a JSON block:") |> ignore
            sb.AppendLine("```tool") |> ignore

            sb.AppendLine("""{"name": "tool_name", "arguments": {"arg1": "value1"}}""")
            |> ignore

            sb.AppendLine("```") |> ignore
            sb.AppendLine("Or use the simple format: TOOL:name:input") |> ignore

            sb.AppendLine("\nAfter receiving tool results, incorporate them into your response.\n")
            |> ignore

            sb.ToString()

    let buildSystemPrompt (agent: Agent) (history: Message list) =
        let sb = StringBuilder()
        sb.AppendLine(agent.SystemPrompt) |> ignore

        // Add tool descriptions if agent has tools
        if not agent.Tools.IsEmpty then
            sb.Append(formatTools agent.Tools) |> ignore

        sb.AppendLine("\n## Conversation History") |> ignore

        for msg in history do
            let sourceName =
                match msg.Sender with
                | MessageEndpoint.System -> "System"
                | MessageEndpoint.User -> "User"
                | MessageEndpoint.Agent _ -> "Assistant"
                | MessageEndpoint.Alias name -> name

            let performative = sprintf "[%A]" msg.Performative
            sb.AppendLine $"{sourceName} {performative}: {msg.Content}" |> ignore

        if agent.Tools.IsEmpty then
            sb.AppendLine("\nInstructions: Reply to the user directly.") |> ignore
        else
            sb.AppendLine("\nInstructions: Reply to the user. Use tools when needed to accomplish tasks.")
            |> ignore

            sb.AppendLine(
                ToolGrammar.spec.PromptHint
                + " Wrap tool calls in ```tool``` fenced JSON with fields \"name\" and \"arguments\"."
            )
            |> ignore

        sb.AppendLine("\n## Communication Protocol") |> ignore

        sb.AppendLine(
            "You can perform specific speech acts by starting your response with ACT: <PERFORMATIVE>: <CONTENT>"
        )
        |> ignore

        sb.AppendLine("Valid performatives: REQUEST, INFORM, QUERY, PROPOSE, REFUSE, FAILURE, EVENT")
        |> ignore

        sb.AppendLine("Example: ACT: PROPOSE: I can optimize this function for $5.")
        |> ignore

        sb.ToString()

module ResponseParser =
    open System.Text.Json
    open System.Text.RegularExpressions

    type ParsedResponse =
        | ToolCall of name: string * input: string
        | MultiToolCall of calls: (string * string) list
        | TextResponse of text: string
        | SpeechAct of performative: Performative * content: string

    /// Parses tool calls in multiple formats:
    /// 1. TOOL:name:input (legacy format)
    /// 2. ```tool\n{"name": "...", "input": "..."}\n``` (JSON block)
    /// 3. <tool_call>{"name": "...", "arguments": {...}}</tool_call> (XML-style)
    /// 4. {"tool": "name", "args": {...}} (inline JSON)
    let parseWithValidator (response: string) (validatorOpt: (string -> bool) option) =
        let text = response.Trim()

        let isValid =
            match validatorOpt with
            | Some v -> v
            | None -> fun _ -> true

        let parsePerformative (s: string) =
            match s.ToUpperInvariant() with
            | "REQUEST" -> Some Performative.Request
            | "INFORM" -> Some Performative.Inform
            | "QUERY" -> Some Performative.Query
            | "PROPOSE" -> Some Performative.Propose
            | "REFUSE" -> Some Performative.Refuse
            | "FAILURE" -> Some Performative.Failure
            | "NOTUNDERSTOOD" -> Some Performative.NotUnderstood
            | "EVENT" -> Some Performative.Event
            | _ -> None

        // Try Speech Act format: ACT: <PERFORMATIVE>: <CONTENT>
        if text.StartsWith("ACT:") then
            let parts = text.Split(':', 3)

            if parts.Length = 3 then
                let performativeStr = parts[1].Trim()
                let content = parts[2].Trim()

                match parsePerformative performativeStr with
                | Some p -> SpeechAct(p, content)
                | None -> TextResponse text
            else
                TextResponse text
        elif text.StartsWith("TOOL:") then
            let parts = text.Split(':', 3)

            if parts.Length = 3 then
                ToolCall(parts[1].Trim(), parts[2].Trim())
            else
                TextResponse text
        else
            // Try to find JSON tool call patterns
            let toolCallPattern = @"<tool_call>\s*(\{[^}]+\})\s*</tool_call>"
            let jsonBlockPattern = @"```(?:tool|json)\s*(\{[^`]+\})\s*```"

            let inlineJsonPattern =
                @"\{[^{}]*""(?:tool|name|function)""\s*:\s*""([^""]+)""[^{}]*(?:""(?:args|arguments|input|parameters)""\s*:\s*(\{[^{}]*\}|""[^""]*""))?[^{}]*\}"

            let tryParseToolJson (json: string) =
                try
                    let doc = JsonDocument.Parse(json)
                    let root = doc.RootElement
                    let mutable nameProp = Unchecked.defaultof<JsonElement>
                    let mutable inputProp = Unchecked.defaultof<JsonElement>

                    let name =
                        if root.TryGetProperty("name", &nameProp) then
                            nameProp.GetString()
                        elif root.TryGetProperty("tool", &nameProp) then
                            nameProp.GetString()
                        elif root.TryGetProperty("function", &nameProp) then
                            nameProp.GetString()
                        else
                            null

                    let input =
                        if root.TryGetProperty("input", &inputProp) then
                            inputProp.GetRawText()
                        elif root.TryGetProperty("arguments", &inputProp) then
                            inputProp.GetRawText()
                        elif root.TryGetProperty("args", &inputProp) then
                            inputProp.GetRawText()
                        elif root.TryGetProperty("parameters", &inputProp) then
                            inputProp.GetRawText()
                        else
                            "{}"

                    if not (isNull name) && isValid json then
                        Some(name, input)
                    else
                        None
                with _ ->
                    None

            // Try XML-style tool_call
            let xmlMatch = Regex.Match(text, toolCallPattern, RegexOptions.Singleline)

            if xmlMatch.Success then
                match tryParseToolJson (xmlMatch.Groups.[1].Value) with
                | Some(name, input) -> ToolCall(name, input)
                | None -> TextResponse text
            else
                // Try ```tool block
                let blockMatch = Regex.Match(text, jsonBlockPattern, RegexOptions.Singleline)

                if blockMatch.Success then
                    match tryParseToolJson (blockMatch.Groups.[1].Value) with
                    | Some(name, input) -> ToolCall(name, input)
                    | None -> TextResponse text
                else
                    // Try to find inline JSON with tool/name/function key
                    let inlineMatches = Regex.Matches(text, inlineJsonPattern, RegexOptions.Singleline)

                    if inlineMatches.Count > 0 then
                        let calls =
                            inlineMatches
                            |> Seq.cast<Match>
                            |> Seq.choose (fun m ->
                                let fullMatch = m.Value
                                tryParseToolJson fullMatch)
                            |> Seq.toList

                        match calls with
                        | [] -> TextResponse text
                        | [ (name, input) ] -> ToolCall(name, input)
                        | multiple -> MultiToolCall multiple
                    else
                        TextResponse text

    let parse (response: string) = parseWithValidator response None

module GraphRuntime =
    type GraphContext =
        { Registry: IAgentRegistry
          Llm: ILlmService
          MaxSteps: int
          BudgetGovernor: BudgetGovernor option
          OutputGuard: IOutputGuard option
          Logger: string -> unit }

    let private createMessage (source: MessageEndpoint) (content: string) =
        { Id = Guid.NewGuid()
          CorrelationId = CorrelationId(Guid.NewGuid()) // ideally propagate from context
          Sender = source
          Receiver = Some MessageEndpoint.System // or generic target
          Performative = Performative.Inform
          Intent = None // Default to None, or infer?
          Constraints = SemanticConstraints.Default
          Ontology = None
          Language = "text"
          Content = content
          Timestamp = DateTime.UtcNow
          Metadata = Map.empty }

    let private handleIdle (agent: Agent) =
        match agent.Memory with
        | [] ->
            { agent with
                State = WaitingForUser "How can I help you?" }
        | msgs -> { agent with State = Thinking msgs }
        |> Success
        |> Task.FromResult

    let private handleThinking (agent: Agent) (history: Message list) (ctx: GraphContext) =
        task {
            // 1. Construct Prompt
            ctx.Logger $"[Thinking] Agent {agent.Name} is thinking..."
            let basePrompt = PromptBuilder.buildSystemPrompt agent history

            let mutable currentPromptMessages =
                [ { Role = Role.User
                    Content = basePrompt } ]

            let mutable attempts = 0
            let maxAttempts = 3
            let mutable finalResponseText = ""
            let mutable guardPassed = false
            let mutable failureReason = ""

            // 1.5 Check Budget
            let canSpend =
                match ctx.BudgetGovernor with
                | Some governor -> governor.CanAfford { Cost.Zero with Tokens = 100<token> }
                | None -> true

            if not canSpend then
                return Failure [ PartialFailure.Error "Budget Exhausted" ]
            else
                while attempts < maxAttempts && not guardPassed do
                    attempts <- attempts + 1

                    // 2. Call LLM
                    let req =
                        { ModelHint = Some agent.Model
                          Model = None
                          SystemPrompt = None
                          MaxTokens = Some 1024
                          Temperature = Some 0.7
                          Stop = []
                          Messages = currentPromptMessages
                          Tools = []
                          ToolChoice = None
                          ResponseFormat = None
                          Stream = false
                          JsonMode = false
                          Seed = None }

                    let! response = ctx.Llm.CompleteAsync req

                    // Record Usage if BudgetGovernor is present
                    match ctx.BudgetGovernor with
                    | Some governor ->
                        let tokens =
                            match response.Usage with
                            | Some u -> u.TotalTokens
                            | None -> (basePrompt.Length + response.Text.Length) / 4 // Fallback estimation

                        governor.Consume
                            { Cost.Zero with
                                Tokens = tokens * 1<token> }
                        |> ignore
                    | None -> ()

                    // 2.5 Apply Output Guard
                    match ctx.OutputGuard with
                    | None ->
                        finalResponseText <- response.Text
                        guardPassed <- true
                    | Some guard ->
                        let guardInput =
                            { ResponseText = response.Text
                              Grammar = None // Could extract from agent tools if needed
                              ExpectedJsonFields = None
                              RequireCitations = false
                              Citations = None
                              AllowExtraFields = true
                              Metadata = Map.empty }

                        let! guardResult = guard.Evaluate guardInput |> Async.StartAsTask

                        match guardResult.Action with
                        | GuardAction.Accept ->
                            finalResponseText <- response.Text
                            guardPassed <- true
                        | GuardAction.RetryWithHint hint ->
                            ctx.Logger $"[Guard] Retry attempt {attempts}: {hint}"

                            currentPromptMessages <-
                                currentPromptMessages
                                @ [ { Role = Role.Assistant
                                      Content = response.Text }
                                    { Role = Role.User
                                      Content = $"Output rejected. Hint: {hint}" } ]
                        | GuardAction.AskForEvidence hint ->
                            ctx.Logger $"[Guard] Asking for evidence: {hint}"

                            currentPromptMessages <-
                                currentPromptMessages
                                @ [ { Role = Role.Assistant
                                      Content = response.Text }
                                    { Role = Role.User
                                      Content = $"Please provide evidence: {hint}" } ]
                        | GuardAction.Reject reason ->
                            ctx.Logger $"[Guard] Rejected: {reason}"
                            failureReason <- reason
                            attempts <- maxAttempts // Stop loop
                        | GuardAction.Fallback text ->
                            ctx.Logger $"[Guard] Fallback triggered"
                            finalResponseText <- text
                            guardPassed <- true

                if not guardPassed then
                    let reason =
                        if String.IsNullOrEmpty failureReason then
                            "Max retry attempts exceeded"
                        else
                            failureReason

                    return Failure [ PartialFailure.Error reason ]
                else
                    // 3. Parse Response
                    let parsed =
                        if agent.Tools.IsEmpty then
                            ResponseParser.parse finalResponseText
                        else
                            ResponseParser.parseWithValidator finalResponseText (Some ToolGrammar.spec.Validator)

                    match parsed with
                    | ResponseParser.ToolCall(name, input) ->
                        match agent.Tools |> List.tryFind (fun t -> t.Name = name) with
                        | Some tool ->
                            // Record the tool call in memory
                            let msg = createMessage (MessageEndpoint.Agent agent.Id) finalResponseText
                            let newMemory = agent.Memory @ [ msg ]

                            return
                                { agent with
                                    Memory = newMemory
                                    State = Acting(tool, input) }
                                |> Success
                        | None ->
                            // Tool not found - try to create it dynamically!
                            match ToolFactory.tryCreateTool name input [] with
                            | ToolFactory.Created tool ->
                                // Add the new tool to the agent and use it
                                let msg = createMessage (MessageEndpoint.Agent agent.Id) finalResponseText
                                let newMemory = agent.Memory @ [ msg ]

                                let agentWithNewTool =
                                    { agent with
                                        Tools = agent.Tools @ [ tool ] }

                                return
                                    { agentWithNewTool with
                                        Memory = newMemory
                                        State = Acting(tool, input) }
                                    |> Success
                            | ToolFactory.CreationFailed reason ->
                                return
                                    { agent with
                                        State = WaitingForUser $"Error: Could not create tool {name}: {reason}" }
                                    |> Success
                    | ResponseParser.MultiToolCall calls ->
                        // For now, just take the first one. Future: support parallel tool calls.
                        match calls with
                        | (name, input) :: _ ->
                            match agent.Tools |> List.tryFind (fun t -> t.Name = name) with
                            | Some tool ->
                                let msg = createMessage (MessageEndpoint.Agent agent.Id) finalResponseText
                                let newMemory = agent.Memory @ [ msg ]

                                return
                                    { agent with
                                        Memory = newMemory
                                        State = Acting(tool, input) }
                                    |> Success
                            | None ->
                                return
                                    { agent with
                                        State = WaitingForUser $"Error: Tool %s{name} not found." }
                                    |> Success
                        | [] ->
                            return
                                { agent with
                                    State = WaitingForUser "Error: Empty multi-tool call." }
                                |> Success
                    | ResponseParser.TextResponse responseText ->
                        // Record the assistant's response in memory so the conversation history is preserved
                        let msg = createMessage (MessageEndpoint.Agent agent.Id) responseText
                        let newMemory = agent.Memory @ [ msg ]
                        return
                            { agent with
                                Memory = newMemory
                                State = WaitingForUser responseText }
                            |> Success
                    | ResponseParser.SpeechAct(perf, content) ->
                        let msg =
                            { createMessage (MessageEndpoint.Agent agent.Id) content with
                                Performative = perf }

                        let newMemory = agent.Memory @ [ msg ]

                        return
                            { agent with
                                Memory = newMemory
                                State = WaitingForUser $"ACT: {perf}: {content}" }
                            |> Success
        }

    let private handleActing (agent: Agent) (tool: Tool) (input: string) (ctx: GraphContext) =
        task {
            ctx.Logger $"[Acting] Agent {agent.Name} is executing tool {tool.Name} with input: {input}"
            let! (result: Result<string, string>) = tool.Execute input |> Async.StartAsTask

            match result with
            | Result.Ok output ->
                return
                    { agent with
                        State = Observing(tool, output) }
                    |> Success
            | Result.Error err ->
                // Tool execution error is treated as a PartialSuccess so the agent can recover
                let nextAgent =
                    { agent with
                        State = Observing(tool, $"Error: {err}") }

                return PartialSuccess(nextAgent, [ PartialFailure.ToolError(tool.Name, err) ])
        }

    let private handleObserving (agent: Agent) (tool: Tool) (output: string) (ctx: GraphContext) =
        // Record the observation
        ctx.Logger $"[Observing] Agent {agent.Name} observed output from {tool.Name}"
        let msg = createMessage MessageEndpoint.System $"Result of {tool.Name}: {output}"
        let newMemory = agent.Memory @ [ msg ]

        { agent with
            Memory = newMemory
            State = Idle }
        |> Success
        |> Task.FromResult

    let step (agent: Agent) (ctx: GraphContext) =
        task {
            match agent.State with
            | Idle -> return! handleIdle agent
            | Thinking history -> return! handleThinking agent history ctx
            | Acting(tool, input) -> return! handleActing agent tool input ctx
            | Observing(tool, output) -> return! handleObserving agent tool output ctx
            | WaitingForUser _ -> return Success agent
            | AgentState.Error _ -> return Success agent
        }
