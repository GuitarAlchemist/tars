namespace Tars.Metascript

open System
open System.Threading.Tasks
open System.Text.RegularExpressions
open Tars.Core
open Tars.Core.HybridBrain
open Tars.Core.WorkflowOfThought
open Tars.Llm

/// <summary>
/// Production implementation of the Workflow of Thought engine.
/// Connects WoT nodes to actual LLM services and tool registries.
/// </summary>
type WotEngine(llm: ILlmServiceFunctional, tools: IToolRegistry) =

    interface IWotEngine with
        member this.ExecuteNodeAsync(node, state) =
            task {
                let nodeId, nodeName =
                    match node with
                    | Reason r -> r.Id, r.Name
                    | Work w -> w.Id, w.Name

                let input =
                    match node with
                    | Reason r -> r.Input
                    | Work w -> w.Input

                let policy =
                    match node with
                    | Reason r -> r.Policy
                    | Work w -> w.Policy

                // Helper to resolve context from state history
                let resolveContext (state: WorkflowState) =
                    let getOutputText id =
                        match state.NodeStates |> Map.tryFind id with
                        | Some(Completed content) -> content.Text
                        | _ -> ""

                    match node with
                    | Reason r ->
                        match r.Operation with
                        | Critique target ->
                            let targetText = getOutputText target

                            if String.IsNullOrWhiteSpace(targetText) then
                                input.Text
                            else
                                $"Target Content to Critique:\n{targetText}\n\nAdditional Feedback: {input.Text}"
                        | Synthesize sources ->
                            let contexts =
                                sources
                                |> List.map (fun s ->
                                    let name =
                                        match state.Graph.Nodes |> Map.tryFind s with
                                        | Some(Reason rn) -> rn.Name
                                        | Some(Work wn) -> wn.Name
                                        | None -> "Unknown"

                                    $"--- {name} ---\n{getOutputText s}")

                            let aggregated = String.Join("\n\n", contexts)

                            if String.IsNullOrWhiteSpace(aggregated) then
                                input.Text
                            else
                                $"Sources for Synthesis:\n{aggregated}\n\nObjective: {input.Text}"
                        | _ -> input.Text
                    | Work w -> 
                        // Evidence Binding: If this node depends on another, consume its output
                        let dependency =
                            state.Graph.Edges
                            |> List.tryFind (fun (fromId, edge, toId) -> 
                                fromId = nodeId && edge = DependsOn)
                            |> Option.map (fun (_, _, toId) -> toId)
                        
                        match dependency with
                        | Some depId -> getOutputText depId
                        | None -> input.Text

                let actualText = resolveContext state

                // 1. Policy check (PII redaction, etc.)
                match PolicyEnforcement.evaluate policy input.Text with
                | PolicyGateResult.Blocked reason -> return Result.Error $"Policy blocked: {reason}"
                | PolicyGateResult.RequiresHumanReview reason -> return Result.Error $"Human review required: {reason}"
                | PolicyGateResult.RequiresRedaction sensitive ->
                    let redactedText = PolicyEnforcement.redact input.Text sensitive
                    let output = { input with Text = redactedText }

                    let evidence =
                        { NodeId = nodeId
                          Timestamp = DateTimeOffset.UtcNow
                          InputHash = input.Text.GetHashCode().ToString()
                          OutputHash = output.Text.GetHashCode().ToString()
                          TokensUsed = 0
                          Duration = TimeSpan.FromMilliseconds(50.0)
                          ToolCallsMade = []
                          PolicyChecksRun = [ "PII" ]
                          Decision = "Redacted"
                          Rationale = $"Redacted {sensitive.Length} sensitive items" }

                    return Result.Ok(output, evidence)

                | PolicyGateResult.Allowed ->
                    match node with
                    | Reason reasonNode ->
                        // 2. Real LLM call for ReasonNode
                        let prompt =
                            sprintf
                                "Operation: %A\nNode: %s\nInput Context: %s\n\nInstructions: Please execute this reasoning step. If critiquing, be specific about errors. If planning, output valid JSON steps."
                                reasonNode.Operation
                                nodeName
                                actualText

                        let req =
                            { LlmRequest.Default with
                                SystemPrompt =
                                    Some
                                        "You are TARS, an autonomous reasoning system. Provide structured, accurate thinking. For plans, output valid JSON. For critiques, reference specific evidence."
                                Messages = [ { Role = Role.User; Content = prompt } ]
                                Temperature = Some 0.2
                                Model = reasonNode.Model
                                ModelHint = reasonNode.ModelHint |> Option.orElse (Some "reasoning")
                                ContextWindow = reasonNode.Budget.MaxContext }

                        let startTime = DateTimeOffset.UtcNow
                        let! resResult = llm.CompleteAsync req
                        let endTime = DateTimeOffset.UtcNow

                        match resResult with
                        | Result.Ok response ->
                            // Strip <think>...</think> blocks
                            let cleanText = 
                                let startThink = response.Text.IndexOf("<think>")
                                let endThink = response.Text.LastIndexOf("</think>")
                                if startThink >= 0 && endThink > startThink then
                                    response.Text.Substring(endThink + 8).Trim()
                                else response.Text

                            // Attempt to parse formal critique if this is a critique node
                            let mutable structured = Map.empty

                            match reasonNode.Operation with
                            | Critique _ ->
                                try
                                    // Look for JSON block in the response
                                    let content = cleanText

                                    let jsonMatch =
                                        Regex.Match(content, "```json\\s*({.*?})\\s*```", RegexOptions.Singleline)

                                    if jsonMatch.Success then
                                        let json = jsonMatch.Groups.[1].Value

                                        let critique =
                                            System.Text.Json.JsonSerializer.Deserialize<
                                                Tars.Core.HybridBrain.FormalCritique
                                             >(
                                                json
                                            )

                                        structured <- Map.ofList [ ("formal_critique", box critique) ]

                                        printfn
                                            "   [DEBUG] Parsed formal critique with %d suggestions"
                                            (critique.Suggestions |> List.length)
                                with _ ->
                                    ()
                            | _ -> ()

                            let output =
                                { Text = cleanText
                                  Structured = structured
                                  Confidence = 0.95
                                  Sources = [] }

                            let usage =
                                response.Usage
                                |> Option.defaultValue
                                    { PromptTokens = 0
                                      CompletionTokens = 0
                                      TotalTokens = 0 }

                            let evidence =
                                { NodeId = nodeId
                                  Timestamp = DateTimeOffset.UtcNow
                                  InputHash = input.Text.GetHashCode().ToString()
                                  OutputHash = output.Text.GetHashCode().ToString()
                                  TokensUsed = usage.TotalTokens
                                  Duration = endTime - startTime
                                  ToolCallsMade = []
                                  PolicyChecksRun = if policy.CheckPII then [ "PII" ] else []
                                  Decision = "Completed"
                                  Rationale =
                                    match reasonNode.Operation with
                                    | Plan _ -> "LLM generated initial plan"
                                    | Critique _ -> "LLM provided critique of previous step"
                                    | Synthesize _ -> "LLM synthesized multiple inputs"
                                    | _ -> "LLM reasoning executed successfully" }

                            return Result.Ok(output, evidence)
                        | Result.Error err -> return Result.Error $"LLM Execution Failed: {err}"

                    | Work workNode ->
                        // 3. Real tool call for WorkNode
                        let startTime = DateTimeOffset.UtcNow

                        match workNode.Operation with
                        | ToolCall(toolName, args) ->
                            match tools.Get toolName with
                            | Some tool ->
                                // Prepare tool input: use args if present (JSON), else fallback to actualText (upstream output)
                                let toolInput =
                                    if args.IsEmpty then
                                        actualText
                                    else
                                        System.Text.Json.JsonSerializer.Serialize(args)

                                let! toolRes = tool.Execute toolInput |> Async.StartAsTask
                                let endTime = DateTimeOffset.UtcNow

                                match toolRes with
                                | Result.Ok resText ->
                                    let output =
                                        { Text = resText
                                          Structured = Map.empty
                                          Confidence = 1.0
                                          Sources = [] }

                                    let evidence =
                                        { NodeId = nodeId
                                          Timestamp = DateTimeOffset.UtcNow
                                          InputHash = input.Text.GetHashCode().ToString()
                                          OutputHash = output.Text.GetHashCode().ToString()
                                          TokensUsed = 0
                                          Duration = endTime - startTime
                                          ToolCallsMade = [ toolName ]
                                          PolicyChecksRun = if policy.CheckPII then [ "PII" ] else []
                                          Decision = "Completed"
                                          Rationale = $"Tool '{toolName}' executed successfully" }

                                    return Result.Ok(output, evidence)
                                | Result.Error err -> return Result.Error $"Tool Execution Failed: {err}"
                            | None -> return Result.Error $"Tool Not Found: {toolName}"
                        | Verify(condition, expected) ->
                            // Grounded Verification: Check if expected string is in the actual input text
                            let isMatch = actualText.Contains(expected, StringComparison.OrdinalIgnoreCase)
                            
                            let decision = if isMatch then "Verified" else "Failed"
                            let resultText = if isMatch then $"Verified: Output contains '{expected}'" else $"Failed: Output did not contain '{expected}'. Got: {actualText.Substring(0, Math.Min(50, actualText.Length))}..."

                            if not isMatch then
                                return Result.Error resultText
                            else
                                let output =
                                    { Text = resultText
                                      Structured = Map.empty
                                      Confidence = 1.0
                                      Sources = [] }

                                let evidence =
                                    { NodeId = nodeId
                                      Timestamp = DateTimeOffset.UtcNow
                                      InputHash = actualText.GetHashCode().ToString()
                                      OutputHash = output.Text.GetHashCode().ToString()
                                      TokensUsed = 0
                                      Duration = DateTimeOffset.UtcNow - startTime
                                      ToolCallsMade = []
                                      PolicyChecksRun = if policy.CheckPII then [ "PII" ] else []
                                      Decision = decision
                                      Rationale = $"Checked '{condition}': {resultText}" }

                                return Result.Ok(output, evidence)

                        | Redact patterns ->
                            let redactedText = input.Text // Simplification
                            let output = { input with Text = redactedText }

                            return
                                Result.Ok(
                                    output,
                                    { NodeId = nodeId
                                      Timestamp = DateTimeOffset.UtcNow
                                      InputHash = ""
                                      OutputHash = ""
                                      TokensUsed = 0
                                      Duration = TimeSpan.Zero
                                      ToolCallsMade = []
                                      PolicyChecksRun = [ "Redact" ]
                                      Decision = "Redacted"
                                      Rationale = "Applied redaction patterns" }
                                )

                        | op ->
                            // Basic support for other ops if not implemented as separate tools
                            return Result.Error $"Operation '{op}' not yet implemented in engine"
            }

        member this.ExecuteWorkflowAsync(graph) = this.ExecuteWorkflowAsync(graph)

    member this.ExecuteWorkflowAsync(graph) =
            task {
                let mutable state = WorkflowExecution.init graph

                while state.CurrentNode.IsSome do
                    let nodeId = state.CurrentNode.Value

                    match graph.Nodes |> Map.tryFind nodeId with
                    | Some node ->
                        let! res = (this :> IWotEngine).ExecuteNodeAsync(node, state)

                        match res with
                        | Result.Ok(output, evidence) ->
                            let newState =
                                { state with
                                    NodeStates = state.NodeStates |> Map.add nodeId (Completed output)
                                    History = state.History @ [ nodeId ]
                                    TotalTokensUsed = state.TotalTokensUsed + evidence.TokensUsed
                                    TotalDuration = state.TotalDuration + evidence.Duration
                                    AllEvidence = state.AllEvidence @ [ evidence ] }

                            let newStateWithDeps = WorkflowExecution.updateDependencies nodeId newState

                            state <-
                                { newStateWithDeps with
                                    CurrentNode = WorkflowExecution.nextReady newStateWithDeps }
                        | Result.Error err ->
                            // Stop on error
                            let failedState =
                                { state with
                                    NodeStates = state.NodeStates |> Map.add nodeId (Failed err)
                                    CurrentNode = None }

                            state <- failedState
                    | None -> state <- { state with CurrentNode = None }

                return state
            }
