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
            let (AgentId guid) = agentId

            // Generate deterministic ID
            let id =
                let raw = guid.ToString() + capability.Kind.ToString() + capability.Description
                use sha = System.Security.Cryptography.SHA256.Create()
                let bytes = System.Text.Encoding.UTF8.GetBytes(raw)
                sha.ComputeHash(bytes) |> Convert.ToHexString |> (fun s -> s.ToLowerInvariant())

            // TODO: Check if exists to avoid re-embedding (requires IVectorStore extension)

            // Create embedding for the capability description
            // This might take time if model needs to be pulled
            let! embedding = llm.EmbedAsync capability.Description

            // Serialize payload
            let payload =
                Map
                    [ "agent_id", string guid
                      "kind", string capability.Kind
                      "description", capability.Description
                      "input_schema", defaultArg capability.InputSchema ""
                      "output_schema", defaultArg capability.OutputSchema ""
                      "confidence", capability.Confidence |> Option.map string |> Option.defaultValue ""
                      "reputation", capability.Reputation |> Option.map string |> Option.defaultValue "" ]

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

                    let confidence =
                        payload
                        |> Map.tryFind "confidence"
                        |> Option.bind (fun s ->
                            match Double.TryParse s with
                            | true, v -> Some(float v)
                            | _ -> None)

                    let reputation =
                        payload
                        |> Map.tryFind "reputation"
                        |> Option.bind (fun s ->
                            match Double.TryParse s with
                            | true, v -> Some(float v)
                            | _ -> None)

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
                          OutputSchema = output
                          Confidence = confidence
                          Reputation = reputation }

                    // Blend vector similarity with reputation/confidence to bias towards reliable agents.
                    let repScore = reputation |> Option.defaultValue 0.5
                    let confScore = confidence |> Option.defaultValue 0.5
                    let adjustedScore = score + (float32 repScore * 0.1f) + (float32 confScore * 0.05f)

                    match agentIdOpt with
                    | Some agentId -> Some(agentId, capability, float adjustedScore)
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

    interface ICapabilityStore with
        member this.FindAgentsAsync(query, limit) = this.FindAgentsAsync(query, limit)
