namespace Tars.Graph

open Tars.Core
open Tars.Connectors
open System.Threading.Tasks
open System.Text
open System

module PromptBuilder =
    let buildSystemPrompt (agent: Agent) (history: Message list) =
        let sb = StringBuilder()
        sb.AppendLine(agent.SystemPrompt) |> ignore
        sb.AppendLine("History:") |> ignore

        for msg in history do
            let sourceName =
                match msg.Source with
                | System -> "System"
                | User -> "User"
                | Agent _ -> "Assistant"
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

module Graph =
    type GraphContext =
        { Kernel: KernelContext; MaxSteps: int }

    let private createMessage (source: MessageEndpoint) (content: string) =
        { Id = Guid.NewGuid()
          CorrelationId = CorrelationId(Guid.NewGuid()) // ideally propagate from context
          Source = source
          Target = System // or generic target
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

    let private handleThinking (agent: Agent) (history: Message list) =
        task {
            // 1. Construct Prompt
            let prompt = PromptBuilder.buildSystemPrompt agent history

            // 2. Call LLM
            let! response = Llm.generate agent.Model prompt

            match response with
            | Result.Ok text ->
                match ResponseParser.parse text with
                | ResponseParser.ToolCall(name, input) ->
                    match agent.Tools |> List.tryFind (fun t -> t.Name = name) with
                    | Some tool ->
                        // Record the tool call in memory
                        let msg = createMessage (Agent agent.Id) text
                        let newMemory = agent.Memory @ [msg]
                        
                        return
                            { agent with
                                Memory = newMemory
                                State = Acting(tool, input) }
                    | None ->
                        return
                            { agent with
                                State = WaitingForUser $"Error: Tool %s{name} not found." }
                | ResponseParser.TextResponse responseText ->
                    // For final response, we might want to add it to memory too, 
                    // but currently the loop ends here.
                    return
                        { agent with
                            State = WaitingForUser responseText }
            | Result.Error err ->
                return
                    { agent with
                        State = AgentState.Error $"LLM Error: %s{err}" }
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
        let msg = createMessage System $"Result of {tool.Name}: {output}"
        let newMemory = agent.Memory @ [msg]
        
        Task.FromResult
            { agent with
                Memory = newMemory
                State = Idle }

    let step (agent: Agent) (ctx: GraphContext) =
        task {
            match agent.State with
            | Idle -> return! handleIdle agent
            | Thinking history -> return! handleThinking agent history
            | Acting(tool, input) -> return! handleActing agent tool input
            | Observing(tool, output) -> return! handleObserving agent tool output
            | WaitingForUser _ -> return agent
            | AgentState.Error _ -> return agent
        }
