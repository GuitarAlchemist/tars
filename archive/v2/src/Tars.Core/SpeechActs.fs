namespace Tars.Core

open System

/// Interaction protocols derived from FIPA standards
type InteractionProtocol =
    | RequestResponse // Ask -> Tell/Error
    | ContractNet // Propose -> Accept/Reject
    | Notification // Event/Tell -> (No response expected locally)
    | Command // Act -> (Execution result observed)
    | Terminal // End of a flow (Accept/Reject/ACK)

/// Helper module for managing semantic interactions (Speech Acts)
module SpeechActs =

    /// Determine the protocol required for a given intent
    let determineProtocol (intent: AgentIntent) : InteractionProtocol =
        match intent with
        | Ask _ -> RequestResponse
        | Propose _ -> ContractNet
        | Tell _
        | Event _
        | Error _ -> Notification
        | Act _ -> Command
        | Accept _
        | Reject _ -> Terminal

    /// Validates if a reply message logically follows the original message
    /// Enforces FIPA-style transition rules.
    let validateFlow (original: AgentMessage) (reply: AgentMessage) : Result<unit, string> =
        if reply.CorrelationId <> original.CorrelationId then
            Result.Error $"CorrelationId mismatch: {original.CorrelationId} vs {reply.CorrelationId}"
        else
            match original.Intent, reply.Intent with
            // Ask -> Tell | Error
            | Ask _, Tell _ -> Ok()
            | Ask _, Error _ -> Ok()
            | Ask _, _ -> Result.Error "Invalid transition: Ask must be followed by Tell or Error"

            // Propose -> Accept | Reject | Error
            | Propose _, Accept refId ->
                if refId = original.Id then
                    Ok()
                else
                    Result.Error $"Accept ID mismatch: Expected {original.Id}, got {refId}"
            | Propose _, Reject(refId, _) ->
                if refId = original.Id then
                    Ok()
                else
                    Result.Error $"Reject ID mismatch: Expected {original.Id}, got {refId}"
            | Propose _, Error _ -> Ok()
            | Propose _, _ -> Result.Error "Invalid transition: Propose must be followed by Accept, Reject, or Error"

            // Tell/Act -> usually don't have direct semantic replies in this restricted flow checker,
            // unless we count ACK. For now, assume they terminate or start new headers.
            | Tell _, _ ->
                Result.Error
                    "Tell is a terminal/notification act, no semantic reply expected in this protocol (start new flow)"
            | Act _, _ ->
                Result.Error "Act is a command, results are observed via Event/Tell, not direct reply logic yet"

            | _ -> Result.Error $"Unsupported transition from {original.Intent} to {reply.Intent}"

    /// Creates a reply envelope with the correct correlation/semantic headers
    let createReply (original: AgentMessage) (intent: AgentIntent) (content: string) (senderId: AgentId) =
        { AgentMessage.Create(
              MessageEndpoint.Agent senderId,
              intent,
              content,
              ?receiver = Some(original.From),
              correlationId = original.CorrelationId
          ) with
            Domain = original.Domain }

    /// Formats an intent and content into the standard TARS wire format: "ACT: <IntentName>: <Content>"
    let format (intent: AgentIntent) (content: string) =
        let name =
            match intent with
            | Ask _ -> "Ask"
            | Tell _ -> "Tell"
            | Propose _ -> "Propose"
            | Accept _ -> "Accept"
            | Reject _ -> "Reject"
            | Act _ -> "Act"
            | Event _ -> "Event"
            | Error _ -> "Error"

        $"ACT: {name}: {content}"

    /// Attempts to parse an intent from a wire format string.
    /// Format: "ACT: <IntentName>: <Content>"
    let tryParse (text: string) : (AgentIntent * string) option =
        if not (text.StartsWith("ACT:")) then
            None
        else
            let parts = text.Split(':', 3)

            if parts.Length < 3 then
                None
            else
                let intentName = parts.[1].Trim()
                let content = parts.[2].Trim()

                match intentName.ToLowerInvariant() with
                | "ask" | "request" | "query" -> Some(Ask content, content)
                | "tell" | "inform" -> Some(Tell content, content)
                | "propose" -> Some(AgentIntent.Propose content, content)
                | "accept" | "verify" ->
                    match Guid.TryParse(content) with
                    | (true, id) -> Some(Accept id, content)
                    | _ -> Some(Accept Guid.Empty, content) // Fallback for lax parsing
                | "reject" -> Some(Reject(Guid.Empty, content), content)
                | "act" -> Some(Act("unknown", content), content)
                | "event" -> Some(AgentIntent.Event("general", content), content)
                | "error" -> Some(AgentIntent.Error content, content)
                | _ -> None

    /// Maps a SemanticMessage to an AgentMessage for validation
    let fromSemantic (msg: SemanticMessage<string>) : AgentMessage =
        // Map Performative to AgentIntent if not already present
        let intent =
            match tryParse msg.Content with
            | Some(i, _) -> i
            | None ->
                match msg.Performative with
                | Performative.Request -> Ask msg.Content
                | Performative.Inform -> Tell msg.Content
                | Performative.Query -> Ask msg.Content
                | Performative.Propose -> AgentIntent.Propose msg.Content
                | Performative.Refuse -> Reject(Guid.Empty, msg.Content)
                | Performative.Failure -> AgentIntent.Error msg.Content
                | Performative.NotUnderstood -> AgentIntent.Error "NotUnderstood"
                | Performative.Event -> AgentIntent.Event("system", msg.Content)

        { Id = msg.Id
          CorrelationId = msg.CorrelationId
          From = msg.Sender
          To = msg.Receiver
          Intent = intent
          Domain = msg.Ontology
          Content = msg.Content
          Timestamp = msg.Timestamp
          Metadata = msg.Metadata }
