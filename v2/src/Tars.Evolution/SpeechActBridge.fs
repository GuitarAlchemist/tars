/// Speech Act Bridge for Evolution Engine
/// Provides semantic message wrappers for agent-to-agent communication
namespace Tars.Evolution

open System
open Tars.Core

/// Speech act helpers for Evolution loop
module SpeechActBridge =

    /// Create a Request message for task assignment
    let requestTask
        (curriculumId: AgentId)
        (executorId: AgentId)
        (task: TaskDefinition)
        : SemanticMessage<TaskDefinition> =
        { Id = Guid.NewGuid()
          CorrelationId = CorrelationId task.Id
          Sender = MessageEndpoint.Agent curriculumId
          Receiver = Some(MessageEndpoint.Agent executorId)
          Performative = Performative.Request
          Intent = Some AgentDomain.Coding
          Constraints = SemanticConstraints.Default
          Ontology = Some "coding"
          Language = "fsharp"
          Content = task
          Timestamp = DateTime.UtcNow
          Metadata = Map.empty }

    /// Create an Inform message for task completion
    let informResult
        (executorId: AgentId)
        (curriculumId: AgentId)
        (correlationId: Guid)
        (result: TaskResult)
        : SemanticMessage<TaskResult> =
        { Id = Guid.NewGuid()
          CorrelationId = CorrelationId correlationId
          Sender = MessageEndpoint.Agent executorId
          Receiver = Some(MessageEndpoint.Agent curriculumId)
          Performative =
            if result.Success then
                Performative.Inform
            else
                Performative.Failure
          Intent = Some AgentDomain.Coding
          Constraints = SemanticConstraints.Default
          Ontology = Some "coding_result"
          Language = "fsharp"
          Content = result
          Timestamp = DateTime.UtcNow
          Metadata = Map.empty }

    /// Create a Refuse message when task cannot be accepted
    let refuseTask
        (executorId: AgentId)
        (curriculumId: AgentId)
        (correlationId: Guid)
        (reason: string)
        : SemanticMessage<string> =
        { Id = Guid.NewGuid()
          CorrelationId = CorrelationId correlationId
          Sender = MessageEndpoint.Agent executorId
          Receiver = Some(MessageEndpoint.Agent curriculumId)
          Performative = Performative.Refuse
          Intent = Some AgentDomain.Planning
          Constraints = SemanticConstraints.Default
          Ontology = Some "resource_management"
          Language = "en"
          Content = reason
          Timestamp = DateTime.UtcNow
          Metadata = Map.empty }

    /// Format speech act for logging
    let formatForLog (msg: SemanticMessage<'T>) : string =
        let senderStr =
            match msg.Sender with
            | MessageEndpoint.System -> "System"
            | MessageEndpoint.User -> "User"
            | MessageEndpoint.Agent aid -> sprintf "Agent:%O" aid
            | MessageEndpoint.Alias name -> sprintf "Alias:%s" name

        let receiverStr =
            match msg.Receiver with
            | Some(MessageEndpoint.Agent aid) -> sprintf "Agent:%O" aid
            | Some(MessageEndpoint.Alias name) -> sprintf "Alias:%s" name
            | Some MessageEndpoint.User -> "User"
            | Some MessageEndpoint.System -> "System"
            | None -> "Broadcast"

        sprintf "[%A] %s -> %s" msg.Performative senderStr receiverStr

    /// Log a speech act message
    let logSpeechAct (logger: string -> unit) (msg: SemanticMessage<'T>) =
        let formatted = formatForLog msg
        logger (sprintf "[SpeechAct] %s" formatted)
