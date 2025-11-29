namespace Tars.Cortex

open System
open System.Threading.Tasks
open Tars.Core
open Tars.Llm
open Tars.Llm.LlmService

/// <summary>
/// Stores and retrieves agent capabilities using semantic search.
/// Allows finding agents that can perform a specific task based on their capability descriptions.
/// </summary>
type CapabilityStore(vectorStore: IVectorStore, llm: ILlmService) =

    let collectionName = "agent_capabilities"

    /// <summary>
    /// Registers a capability for a specific agent.
    /// </summary>
    member this.RegisterAsync(agentId: AgentId, capability: Capability) =
        task {
            // Create embedding for the capability description
            let! embedding = llm.EmbedAsync capability.Description

            let (AgentId guid) = agentId

            // Serialize payload
            let payload =
                Map
                    [ "agent_id", string guid
                      "kind", string capability.Kind
                      "description", capability.Description
                      "input_schema", defaultArg capability.InputSchema ""
                      "output_schema", defaultArg capability.OutputSchema "" ]

            // Generate a unique ID for this registration
            let id = Guid.NewGuid().ToString()

            do! vectorStore.SaveAsync(collectionName, id, embedding, payload)
        }

    /// <summary>
    /// Finds agents with capabilities matching the query.
    /// Returns a list of (AgentId, Capability, Score) tuples.
    /// </summary>
    member this.FindAgentsAsync(query: string, limit: int) =
        task {
            let! embedding = llm.EmbedAsync query

            let! (results: (string * float32 * Map<string, string>) list) =
                vectorStore.SearchAsync(collectionName, embedding, limit)

            return
                results
                |> List.choose (fun (id, score, payload) ->
                    let agentIdOpt =
                        payload
                        |> Map.tryFind "agent_id"
                        |> Option.bind (fun s ->
                            match Guid.TryParse s with
                            | true, g -> Some(AgentId g)
                            | _ -> None)

                    let kindStr = payload |> Map.tryFind "kind" |> Option.defaultValue "Custom"
                    let desc = payload |> Map.tryFind "description" |> Option.defaultValue ""

                    let input =
                        payload |> Map.tryFind "input_schema" |> Option.filter (fun s -> s <> "")

                    let output =
                        payload |> Map.tryFind "output_schema" |> Option.filter (fun s -> s <> "")

                    let kind =
                        match kindStr with
                        | "Summarization" -> Summarization
                        | "WebSearch" -> WebSearch
                        | "CodeGeneration" -> CodeGeneration
                        | "DataAnalysis" -> DataAnalysis
                        | "Planning" -> Planning
                        | "TaskExecution" -> TaskExecution
                        | "Reasoning" -> Reasoning
                        | s -> Custom s

                    let capability =
                        { Kind = kind
                          Description = desc
                          InputSchema = input
                          OutputSchema = output }

                    match agentIdOpt with
                    | Some agentId -> Some(agentId, capability, score)
                    | None -> None)
        }

    /// <summary>
    /// Records usage metrics for a capability (placeholder for future implementation).
    /// </summary>
    member this.TrackUsageAsync(agentId: AgentId, capabilityKind: CapabilityKind, success: bool) =
        task {
            // TODO: Implement metrics tracking (Phase 6.5.4)
            return ()
        }
