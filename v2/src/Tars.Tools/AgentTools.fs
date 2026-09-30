namespace Tars.Tools.Standard

open System
open Tars.Tools
open Tars.Core

module AgentTools =

    /// Agent registry reference (set at startup)
    let mutable private agentRegistry: IAgentRegistry option = None

    /// Set the agent registry reference
    let setRegistry (registry: IAgentRegistry) = agentRegistry <- Some registry

    [<TarsToolAttribute("delegate_task",
                        "Looks up an agent to hand a task to. It cannot run another agent: it says whether the agent is registered, and that nothing was started. Input JSON: { \"agent\": \"Reviewer\", \"task\": \"Review this code for bugs\", \"context\": \"optional context\" }")>]
    let delegateTask (args: string) =
        task {
            try
                let doc = System.Text.Json.JsonDocument.Parse(args)
                let root = doc.RootElement

                let agent = root.GetProperty("agent").GetString()
                let taskDesc = root.GetProperty("task").GetString()

                let mutable contextProp = Unchecked.defaultof<System.Text.Json.JsonElement>

                let context =
                    if root.TryGetProperty("context", &contextProp) then
                        contextProp.GetString()
                    else
                        ""

                printfn $"📤 DELEGATING to %s{agent}: %s{taskDesc.Substring(0, min 50 taskDesc.Length)}"

                match agentRegistry with
                | Some reg ->
                    let agents = reg.GetAllAgents() |> Async.RunSynchronously

                    let target =
                        agents |> List.tryFind (fun a -> a.Name.ToLower().Contains(agent.ToLower()))

                    match target with
                    | Some a ->
                        // Nothing here can run another agent. This used to answer "Task
                        // delegated ... Agent-to-agent execution initiated via registry",
                        // and the caller moved on as if someone were doing the task.
                        return
                            $"❌ Not delegated: %s{a.Name} (%s{a.Version}) is registered, but this tool cannot run another agent, so nothing was started.\nTask: %s{taskDesc}"
                    | None ->
                        return $"❌ Agent '%s{agent}' not found in registry. Use list_agents to see available targets."
                | None -> return "Error: AgentRegistry not initialized."
            with ex ->
                return "delegate_task error: " + ex.Message
        }

    [<TarsToolAttribute("request_review",
                        "Returns a code review checklist for the given code. No reviewer is asked. Input JSON: { \"code\": \"code to review\", \"focus\": \"bugs|style|performance\" }")>]
    let requestReview (args: string) =
        task {
            try
                let doc = System.Text.Json.JsonDocument.Parse(args)
                let root = doc.RootElement

                let code = root.GetProperty("code").GetString()

                let mutable focusProp = Unchecked.defaultof<System.Text.Json.JsonElement>

                let focus =
                    if root.TryGetProperty("focus", &focusProp) then
                        focusProp.GetString()
                    else
                        "general"

                printfn $"🔍 REQUESTING REVIEW (focus: %s{focus})"

                let codePreview =
                    if code.Length > 100 then
                        code.Substring(0, 100) + "..."
                    else
                        code

                // No reviewer is asked: this is a checklist for the caller to work through.
                let review =
                    "Review checklist (no reviewer was asked):\n"
                    + "  Focus: "
                    + focus
                    + "\n"
                    + "  Code preview: "
                    + codePreview
                    + "\n\n"
                    + "Review Checklist:\n"
                    + "  [ ] Check for bugs and edge cases\n"
                    + "  [ ] Verify error handling\n"
                    + "  [ ] Review code style\n"
                    + "  [ ] Check performance implications\n"
                    + "  [ ] Ensure tests are adequate\n\n"
                    + "Use analyze_code or run_tests for deeper analysis."

                return review
            with ex ->
                return "request_review error: " + ex.Message
        }

    [<TarsToolAttribute("query_agent",
                        "Cannot reach another agent: it says the question was not asked. Use list_agents to see what is registered. Input JSON: { \"agent\": \"Curriculum\", \"question\": \"What tasks are pending?\" }")>]
    let queryAgent (args: string) =
        task {
            try
                let doc = System.Text.Json.JsonDocument.Parse(args)
                let root = doc.RootElement

                let agent = root.GetProperty("agent").GetString()
                let question = root.GetProperty("question").GetString()

                printfn $"❓ QUERYING %s{agent}: %s{question.Substring(0, min 40 question.Length)}"

                // Nothing here can reach another agent. The answers used to be written
                // in advance, one per agent name, and returned whatever the question was
                // - presented as that agent's reply.
                return
                    $"❌ Not asked: this tool cannot reach another agent, so %s{agent} was not asked \"%s{question}\". Use list_agents to see what is registered."
            with ex ->
                return "query_agent error: " + ex.Message
        }

    [<TarsToolAttribute("list_agents", "Lists all available agents and their capabilities. No input required.")>]
    let listAgents (_: string) =
        task {
            printfn "📋 LISTING AGENTS"

            match agentRegistry with
            | Some reg ->
                let! agents = reg.GetAllAgents()

                if agents.IsEmpty then
                    return "No agents registered."
                else
                    let lines =
                        agents
                        |> List.mapi (fun i a ->
                            let caps =
                                a.Capabilities |> List.map (fun c -> c.Kind.ToString()) |> String.concat ", "

                            $"{i + 1}. {a.Name} (v{a.Version}) - Status: {a.State}\n   Capabilities: {caps}")

                    return "Available Agents:\n\n" + (String.Join("\n\n", lines))
            | None -> return "Error: AgentRegistry not initialized."
        }

    [<TarsToolAttribute("agent_status",
                        "Gets the current status of an agent. Input: agent name (Curriculum, Executor, or Reviewer)")>]
    let agentStatus (args: string) =
        task {
            let agentName = ToolHelpers.parseStringArg args "agent"

            let name =
                if String.IsNullOrWhiteSpace(agentName) then
                    ""
                else
                    agentName.Trim()

            printfn $"📊 STATUS: %s{name}"

            match agentRegistry with
            | Some reg ->
                let! agents = reg.GetAllAgents()

                let target =
                    agents |> List.tryFind (fun a -> a.Name.ToLower().Contains(name.ToLower()))

                match target with
                | Some a ->
                    let caps =
                        a.Capabilities |> List.map (fun c -> c.Kind.ToString()) |> String.concat ", "

                    return
                        $"Agent: %s{a.Name}\nVersion: %s{a.Version}\nState: %A{a.State}\nCapabilities: %s{caps}\nMemory size: %d{a.Memory.Length}"
                | None ->
                    if String.IsNullOrWhiteSpace(name) then
                        return "Please specify an agent name. Use list_agents to see available agents."
                    else
                        return $"Agent '%s{name}' not found."
            | None -> return "Error: AgentRegistry not initialized."
        }
