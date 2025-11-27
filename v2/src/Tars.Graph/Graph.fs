namespace Tars.Graph

open Tars.Core
open Tars.Connectors
open System.Threading.Tasks
open System.Text
open System
open Tars.Llm
open Tars.Llm.LlmService
open Tars.Kernel

module PromptBuilder =
    let buildSystemPrompt (agent: Agent) (history: Message list) =
        let sb = StringBuilder()
        sb.AppendLine(agent.SystemPrompt) |> ignore
        sb.AppendLine("History:") |> ignore

        for msg in history do
            let sourceName =
                match msg.Sender with
                | MessageEndpoint.System -> "System"
                | MessageEndpoint.User -> "User"
                | MessageEndpoint.Agent _ -> "Assistant"
                | MessageEndpoint.Alias name -> name

            sb.AppendLine $"{sourceName}: {msg.Content}" |> ignore

        sb.AppendLine(
            "Instructions: Reply to the user. If you want to use a tool, format it as TOOL:Name:Input. Otherwise just reply."
        )
        |> ignore

        sb.ToString()

module ResponseParser =
    type ParsedResponse =
        | ToolCall of name: string * input: string
        | TextResponse of text: string

    let parse (response: string) =
        let text = response.Trim()

        if text.StartsWith("TOOL:") then
            let parts = text.Split(':', 3)

            if parts.Length = 3 then
                ToolCall(parts[1], parts[2])
            else
                TextResponse text
        else
            TextResponse text

module GraphRuntime =
    type GraphContext =
        { Kernel: KernelContext
          Llm: ILlmService
          MaxSteps: int
          BudgetGovernor: BudgetGovernor option }

    let private createMessage (source: MessageEndpoint) (content: string) =
        { Id = Guid.NewGuid()
          CorrelationId = CorrelationId(Guid.NewGuid()) // ideally propagate from context
          Sender = source
          Receiver = Some MessageEndpoint.System // or generic target
          Performative = Performative.Inform
          Constraints = SemanticConstraints.Default
          Content = content
          Timestamp = DateTime.UtcNow
          Metadata = Map.empty }

    let private handleIdle (agent: Agent) =
        match agent.Memory with
        | [] ->
            { agent with
                State = WaitingForUser "How can I help you?" }
        | msgs -> { agent with State = Thinking msgs }
        |> Task.FromResult

    let private handleThinking (agent: Agent) (history: Message list) (ctx: GraphContext) =
        task {
            // 1. Construct Prompt
            let prompt = PromptBuilder.buildSystemPrompt agent history

            // 1.5 Check Budget
            let correlationId =
                match history |> List.tryLast with
                | Some msg ->
                    match msg.CorrelationId with
                    | CorrelationId id -> id
                | None -> Guid.Empty

            let canSpend =
                match ctx.BudgetGovernor with
                | Some governor -> governor.CanSpend(correlationId, 100<token>) // Estimate 100 tokens for prompt?
                | None -> true

            if not canSpend then
                return
                    { agent with
                        State = AgentState.Error "Budget Exhausted" }
            else

                // 2. Call LLM
                let req =
                    { ModelHint = Some agent.Model
                      MaxTokens = Some 1024
                      Temperature = Some 0.7
                      Messages = [ { Role = Role.User; Content = prompt } ] }

                let! response = ctx.Llm.CompleteAsync req

                // Record Usage if BudgetGovernor is present
                match ctx.BudgetGovernor with
                | Some governor ->
                    // Estimate usage: 1 token per 4 chars of response + prompt
                    let tokens = (prompt.Length + response.Text.Length) / 4
                    governor.RecordUsage(correlationId, tokens * 1<token>)
                | None -> ()

                // 3. Parse Response
                match ResponseParser.parse response.Text with
                | ResponseParser.ToolCall(name, input) ->
                    match agent.Tools |> List.tryFind (fun t -> t.Name = name) with
                    | Some tool ->
                        // Record the tool call in memory
                        let msg = createMessage (MessageEndpoint.Agent agent.Id) response.Text
                        let newMemory = agent.Memory @ [ msg ]

                        return
                            { agent with
                                Memory = newMemory
                                State = Acting(tool, input) }
                    | None ->
                        return
                            { agent with
                                State = WaitingForUser $"Error: Tool %s{name} not found." }
                | ResponseParser.TextResponse responseText ->
                    return
                        { agent with
                            State = WaitingForUser responseText }
        }

    let private handleActing (agent: Agent) (tool: Tool) (input: string) =
        task {
            let! (result: Result<string, string>) = tool.Execute input |> Async.StartAsTask

            match result with
            | Result.Ok output ->
                return
                    { agent with
                        State = Observing(tool, output) }
            | Result.Error err ->
                return
                    { agent with
                        State = AgentState.Error err }
        }

    let private handleObserving (agent: Agent) (tool: Tool) (output: string) =
        // Record the observation
        let msg = createMessage MessageEndpoint.System $"Result of {tool.Name}: {output}"
        let newMemory = agent.Memory @ [ msg ]

        Task.FromResult
            { agent with
                Memory = newMemory
                State = Idle }

    let step (agent: Agent) (ctx: GraphContext) =
        task {
            match agent.State with
            | Idle -> return! handleIdle agent
            | Thinking history -> return! handleThinking agent history ctx
            | Acting(tool, input) -> return! handleActing agent tool input
            | Observing(tool, output) -> return! handleObserving agent tool output
            | WaitingForUser _ -> return agent
            | AgentState.Error _ -> return agent
        }
