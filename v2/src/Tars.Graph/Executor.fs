namespace Tars.Graph

open System
open System.Threading.Tasks
open Tars.Core
open Tars.Core.AgentWorkflow
open Tars.Kernel

/// Executor that runs an agent loop until completion
type GraphExecutor
    (
        registry: IAgentRegistry,
        llm: Tars.Llm.LlmService.ILlmService,
        budget: BudgetGovernor option,
        outputGuard: IOutputGuard option,
        logger: string -> unit
    ) =

    // Default constructor for backward compatibility
    new(registry, llm, budget) = GraphExecutor(registry, llm, budget, None, fun _ -> ())
    new(registry, llm, budget, logger) = GraphExecutor(registry, llm, budget, None, logger)

    /// Helper to run the agent loop until it produces a response or errors
    member this.RunAgentLoop
        (agent: Agent, maxSteps: int, ?cancellationToken: System.Threading.CancellationToken)
        : Task<ExecutionOutcome<Agent * string * string list>> =
        task {
            let token = defaultArg cancellationToken System.Threading.CancellationToken.None
            let mutable currentAgent = agent
            let mutable stepCount = 0
            let mutable finished = false
            let mutable trace = []
            let mutable resultOutput = ""
            let mutable success = false
            let mutable cancelled = false

            let graphCtx: GraphRuntime.GraphContext =
                { Registry = registry
                  Llm = llm
                  MaxSteps = maxSteps
                  BudgetGovernor = budget
                  OutputGuard = outputGuard
                  CancellationToken = token
                  Logger = logger }

            while not finished && stepCount < maxSteps do
                if token.IsCancellationRequested then
                    cancelled <- true
                    finished <- true
                else
                    trace <- trace @ [ $"Step %d{stepCount}: %A{currentAgent.State}" ]
                    let! outcome = GraphRuntime.step currentAgent graphCtx

                    match outcome with
                    | Success next -> currentAgent <- next
                    | PartialSuccess(next, warnings) ->
                        trace <- trace @ (warnings |> List.map (fun w -> $"Warning: %A{w}"))
                        currentAgent <- next
                    | Failure errs ->
                        let errStr = String.concat "; " (errs |> List.map (fun e -> $"%A{e}"))
                        trace <- trace @ [ $"Execution Failure: %s{errStr}" ]
                        finished <- true
                        success <- false
                        resultOutput <- errStr

                    stepCount <- stepCount + 1

                    match currentAgent.State with
                    | WaitingForUser response ->
                        trace <- trace @ [ $"Response: %s{response}" ]
                        resultOutput <- response
                        success <- true
                        finished <- true
                    | AgentState.Error err ->
                        trace <- trace @ [ $"Error: %s{err}" ]
                        resultOutput <- err
                        success <- false
                        finished <- true
                    | _ -> ()

            let isError =
                match currentAgent.State with
                | AgentState.Error _ -> true
                | _ -> false

            if cancelled then
                let warnings = [ Timeout("AgentLoop", TimeSpan.Zero) ]
                return Failure warnings
            elif not finished then
                // Provide more context about what happened
                let lastState =
                    match currentAgent.State with
                    | Idle -> "idle (never started)"
                    | Thinking _ -> "thinking (LLM call may have stalled)"
                    | Acting(tool, _) -> $"acting on tool '%s{tool.Name}'"
                    | Observing(tool, output) ->
                        let preview =
                            if output.Length > 50 then
                                output.Substring(0, 50) + "..."
                            else
                                output

                        $"observing '%s{tool.Name}' result: %s{preview}"
                    | WaitingForUser _ -> "waiting for user"
                    | AgentState.Error e -> $"error: %s{e}"

                let lastTraceEntry =
                    if trace.Length > 0 then
                        trace.[trace.Length - 1]
                    else
                        "no trace"

                resultOutput <-
                    $"Task incomplete - agent was %s{lastState}. Last trace: %s{lastTraceEntry}. Consider increasing iterations or simplifying the task."

                let warnings = [ Timeout("AgentLoop", TimeSpan.Zero) ]
                return PartialSuccess((currentAgent, resultOutput, trace), warnings)
            elif isError then
                return Failure [ PartialFailure.Error resultOutput ]
            else
                return Success(currentAgent, resultOutput, trace)
        }

    interface IAgentExecutor with
        member this.Execute(agentId, taskSpec) =
            async {
                // 1. Retrieve Agent
                let! agentOpt = registry.GetAgent(agentId)

                match agentOpt with
                | None -> return Failure [ PartialFailure.Error $"Agent {agentId} not found" ]
                | Some agent ->
                    // 2. Send Task Message
                    // We need to construct a message.
                    // Ideally we should use a helper or the agent's input schema.
                    // For now, we assume a simple text request.
                    let msg =
                        { Id = Guid.NewGuid()
                          CorrelationId = CorrelationId(Guid.NewGuid())
                          Sender = MessageEndpoint.System
                          Receiver = Some(MessageEndpoint.Agent agent.Id)
                          Performative = Performative.Request
                          Intent = None
                          Constraints = SemanticConstraints.Default
                          Ontology = None
                          Language = "text"
                          Content = taskSpec
                          Timestamp = DateTime.UtcNow
                          Metadata = Map.empty }

                    let agentWithMsg = agent.ReceiveMessage(msg)

                    // 3. Run Loop
                    let! result = this.RunAgentLoop(agentWithMsg, 20) |> Async.AwaitTask

                    match result with
                    | Success(_, output, _) -> return Success output
                    | PartialSuccess((_, output, _), warnings) -> return PartialSuccess(output, warnings)
                    | Failure errors -> return Failure errors
            }
